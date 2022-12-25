from __future__ import annotations

import math
import os
from dataclasses import dataclass
from threading import Lock

import numpy as np
import tensorrt
import torch
import torch.nn.functional as F
import vapoursynth as vs
from functorch.compile import memory_efficient_fusion
from torch_tensorrt.fx import LowerSetting
from torch_tensorrt.fx.lower import Lowerer
from torch_tensorrt.fx.utils import LowerPrecision

from .femasr_arch import FeMaSRNet, VectorQuantizer
from .network_swinir import SwinTransformerBlock

__version__ = '1.0.0'

package_dir = os.path.dirname(os.path.realpath(__file__))


class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class CUDAGraphs:
        graph: list[torch.cuda.CUDAGraph]
        static_input: list[torch.Tensor]
        static_output: list[torch.Tensor]

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


@torch.inference_mode()
def femasr(
    clip: vs.VideoNode,
    device_index: int | None = None,
    num_streams: int = 1,
    nvfuser: bool = False,
    cuda_graphs: bool = False,
    trt: bool = False,
    trt_max_workspace_size: int = 1 << 25,
    trt_cache_path: str = package_dir,
    model: int = 0,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 16,
) -> vs.VideoNode:
    """Real-World Blind Super-Resolution via Feature Matching with Implicit High-Resolution Priors

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param nvfuser:                 Enable fusion through nvFuser. Not allowed in TensorRT. (experimental)
    :param cuda_graphs:             Use CUDA Graphs to remove CPU overhead associated with launching CUDA kernels
                                    sequentially. Not allowed in TensorRT.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_max_workspace_size:  Maximum workspace size for TensorRT engine.
    :param trt_cache_path:          Path for TensorRT engine file. Engine will be cached when it's built for the first
                                    time. Note each engine is created for specific settings such as model path/name,
                                    precision, workspace etc, and specific GPUs and it's not portable.
    :param model:                   Model to use.
                                    0 = FeMaSR_SRX2_model_g
                                    1 = FeMaSR_SRX4_model_g
    :param tile_w:                  Tile width. As too large images result in the out of GPU memory issue, so this tile
                                    option will first crop input images into tiles, and then process each of them.
                                    Finally, they will be merged into one image. 0 denotes for do not use tile.
    :param tile_h:                  Tile height.
    :param tile_pad:                Pad size for each tile, to remove border artifacts.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('femasr: this is not a clip')

    if clip.format.id not in (vs.RGBH, vs.RGBS):
        raise vs.Error('femasr: only RGBH and RGBS formats are supported')

    if not torch.cuda.is_available():
        raise vs.Error('femasr: CUDA is not available')

    if num_streams < 1:
        raise vs.Error('femasr: num_streams must be at least 1')

    if num_streams > vs.core.num_threads:
        raise vs.Error('femasr: setting num_streams greater than `core.num_threads` is useless')

    if trt:
        if nvfuser:
            raise vs.Error('femasr: nvfuser and trt are mutually exclusive')

        if cuda_graphs:
            raise vs.Error('femasr: cuda_graphs and trt are mutually exclusive')

    if model not in range(2):
        raise vs.Error('femasr: model must be 0 or 1')

    if os.path.getsize(os.path.join(package_dir, 'FeMaSR_SRX2_model_g.pth')) == 0:
        raise vs.Error("femasr: model files have not been downloaded. run 'python -m vsfemasr' first")

    torch.backends.cuda.matmul.allow_tf32 = True

    fp16 = clip.format.bits_per_sample == 16

    device = torch.device('cuda', device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    match model:
        case 0:
            model_name = 'FeMaSR_SRX2_model_g.pth'
            scale = 2
            modulo = 32
        case 1:
            model_name = 'FeMaSR_SRX4_model_g.pth'
            scale = 4
            modulo = 16

    model_path = os.path.join(package_dir, model_name)

    module = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=True, scale_factor=scale)
    module.load_state_dict(torch.load(model_path, map_location='cpu')['params'], strict=False)
    module.eval().to(device, memory_format=torch.channels_last)

    if fp16:
        module.half()
        torch.set_default_tensor_type(torch.HalfTensor)

    if tile_w > 0 and tile_h > 0:
        pad_w = ((min(tile_w + 2 * tile_pad, clip.width) - 1) // modulo + 1) * modulo
        pad_h = ((min(tile_h + 2 * tile_pad, clip.height) - 1) // modulo + 1) * modulo
    else:
        pad_w = ((clip.width - 1) // modulo + 1) * modulo
        pad_h = ((clip.height - 1) // modulo + 1) * modulo

    if nvfuser:
        module = memory_efficient_fusion(module)

    if cuda_graphs:
        graph: list[torch.cuda.CUDAGraph] = []
        static_input: list[torch.Tensor] = []
        static_output: list[torch.Tensor] = []

        for i in range(num_streams):
            static_input.append(torch.empty(1, 3, pad_h, pad_w, device=device, memory_format=torch.channels_last))

            torch.cuda.synchronize(device=device)
            stream[i].wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(stream[i]):
                module(static_input[i])
            torch.cuda.current_stream(device=device).wait_stream(stream[i])
            torch.cuda.synchronize(device=device)

            graph.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graph[i], stream=stream[i]):
                static_output.append(module(static_input[i]))

        backend = Backend.CUDAGraphs(graph, static_input, static_output)
    elif trt:
        device_name = torch.cuda.get_device_name(device)
        trt_version = tensorrt.__version__
        dimensions = f'{pad_w}x{pad_h}'
        precision = 'fp16' if fp16 else 'fp32'
        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_path),
            (
                f'{model_name}'
                + f'_{device_name}'
                + f'_trt-{trt_version}'
                + f'_{dimensions}'
                + f'_{precision}'
                + f'_workspace-{trt_max_workspace_size}'
                + '.pt'
            ),
        )

        if not os.path.isfile(trt_engine_path):
            lower_setting = LowerSetting(
                lower_precision=LowerPrecision.FP16 if fp16 else LowerPrecision.FP32,
                min_acc_module_size=1,
                leaf_module_list={SwinTransformerBlock, VectorQuantizer},
                max_workspace_size=trt_max_workspace_size,
                dynamic_batch=False,
                tactic_sources=1 << int(tensorrt.TacticSource.EDGE_MASK_CONVOLUTIONS)
                | 1 << int(tensorrt.TacticSource.JIT_CONVOLUTIONS),
            )
            lowerer = Lowerer.create(lower_setting=lower_setting)
            module = lowerer(
                module, [torch.empty(1, 3, pad_h, pad_w, device=device, memory_format=torch.channels_last)]
            )
            torch.save(module, trt_engine_path)

        del module
        torch.cuda.empty_cache()
        module = [torch.load(trt_engine_path) for _ in range(num_streams)]
        backend = Backend.TensorRT(module)
    else:
        backend = Backend.Eager(module)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img = frame_to_tensor(f[0], device)

            if tile_w > 0 and tile_h > 0:
                output = tile_process(img, scale, tile_w, tile_h, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                img = F.pad(img, (0, pad_w - w, 0, pad_h - h), 'reflect')

                if cuda_graphs:
                    static_input[local_index].copy_(img)
                    graph[local_index].replay()
                    output = static_output[local_index]
                elif trt:
                    output = module[local_index](img)
                else:
                    output = module(img)

                output = output[:, :, : h * scale, : w * scale]

            return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], inference), clip_src=[clip, new_clip]
    )


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def tile_process(
    img: torch.Tensor,
    scale: int,
    tile_w: int,
    tile_h: int,
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Eager | Backend.CUDAGraphs | Backend.TensorRT,
    index: int,
) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_w)
    tiles_y = math.ceil(height / tile_h)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_w
            ofs_y = y * tile_h

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_w, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_h, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            h, w = input_tile.shape[2:]
            input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), 'reflect')

            # process tile
            if isinstance(backend, Backend.CUDAGraphs):
                backend.static_input[index].copy_(input_tile)
                backend.graph[index].replay()
                output_tile = backend.static_output[index]
            elif isinstance(backend, Backend.TensorRT):
                output_tile = backend.module[index](input_tile)
            else:
                output_tile = backend.module(input_tile)

            output_tile = output_tile[:, :, : h * scale, : w * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output
