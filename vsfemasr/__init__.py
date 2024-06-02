from __future__ import annotations

import math
import os
import warnings
from enum import IntEnum
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .femasr_arch import FeMaSRNet

__version__ = "1.2.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "At pre-dispatch tracing")
warnings.filterwarnings("ignore", "Attempted to insert a get_attr Node with no underlying reference")
warnings.filterwarnings("ignore", "Node _run_on_acc_0_engine target _run_on_acc_0_engine _run_on_acc_0_engine of")
warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class FeMaSRModel(IntEnum):
    FeMaSR_SRX2_model_g = 0
    FeMaSR_SRX4_model_g = 1


@torch.inference_mode()
def femasr(
    clip: vs.VideoNode,
    device_index: int = 0,
    num_streams: int = 1,
    model: FeMaSRModel = FeMaSRModel.FeMaSR_SRX2_model_g,
    trt: bool = False,
    trt_debug: bool = False,
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Real-World Blind Super-Resolution via Feature Matching with Implicit High-Resolution Priors

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param model:                   Model to use.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("femasr: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("femasr: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("femasr: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("femasr: num_streams must be at least 1")

    if model not in FeMaSRModel:
        raise vs.Error("femasr: model must be one of the members in FeMaSRModel")

    if os.path.getsize(os.path.join(model_dir, "FeMaSR_SRX2_model_g.pth")) == 0:
        raise vs.Error("femasr: model files have not been downloaded. run 'python -m vsfemasr' first")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    match model:
        case FeMaSRModel.FeMaSR_SRX2_model_g:
            scale = 2
            modulo = 32
            downscale = 4
        case FeMaSRModel.FeMaSR_SRX4_model_g:
            scale = 4
            modulo = 16
            downscale = 2

    w = clip.width
    h = clip.height
    pad_w = math.ceil(w / modulo) * modulo
    pad_h = math.ceil(h / modulo) * modulo
    padding = (0, pad_w - w, 0, pad_h - h)

    model_name = f"{FeMaSRModel(model).name}.pth"
    state_dict = torch.load(os.path.join(model_dir, model_name), map_location=device, mmap=True)["params"]

    module = FeMaSRNet(
        input_resolution=(pad_h // downscale, pad_w // downscale),
        codebook_params=[[32, 1024, 512]],
        LQ_stage=True,
        scale_factor=scale,
    )
    module.load_state_dict(state_dict, strict=False)
    module.eval().to(device)
    if fp16:
        module.half()

    if trt:
        import tensorrt
        import torch_tensorrt

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_{pad_w}x{pad_h}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ".ep"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            inputs = [torch.zeros((1, 3, pad_h, pad_w), dtype=dtype, device=device)]

            module = torch_tensorrt.compile(
                module,
                ir="dynamo",
                inputs=inputs,
                enabled_precisions={dtype},
                debug=trt_debug,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
                truncate_double=True,
                device=device,
            )

            torch_tensorrt.save(module, trt_engine_path, inputs=inputs)

        module = [torch.export.load(trt_engine_path).module() for _ in range(num_streams)]

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
            img = F.pad(img, padding, "replicate")

            if trt:
                output = module[local_index](img)
            else:
                output = module(img)

            return tensor_to_frame(output[:, :, : h * scale, : w * scale], f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], inference), clip_src=[clip, new_clip]
    )


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return (
        torch.stack([torch.from_numpy(np.asarray(frame[plane])).to(device) for plane in range(frame.format.num_planes)])
        .unsqueeze(0)
        .clamp(0.0, 1.0)
    )


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane])
    return frame
