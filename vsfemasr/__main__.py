import os

import requests
from tqdm import tqdm


def download_model(url: str) -> None:
    filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), filename), 'wb') as f:
        with tqdm(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=filename,
            total=int(r.headers.get('content-length', 0)),
        ) as pbar:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))


if __name__ == '__main__':
    url = 'https://github.com/HolyWu/vs-femasr/releases/download/model/'
    models = ['FeMaSR_SRX2_model_g', 'FeMaSR_SRX4_model_g']
    for model in models:
        download_model(url + model + '.pth')
