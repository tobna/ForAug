"""Use the LaMa (*LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions*) model to infill the background of an image.

Based on: https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py
"""

import abc
import hashlib
import os
import sys
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import torch
from torch.hub import download_url_to_file, get_dir

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")


class LDMSampler(str, Enum):
    ddim = "ddim"
    plms = "plms"


class SDSampler(str, Enum):
    ddim = "ddim"
    pndm = "pndm"
    k_lms = "k_lms"
    k_euler = "k_euler"
    k_euler_a = "k_euler_a"
    dpm_plus_plus = "dpm++"
    uni_pc = "uni_pc"


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def undo_pad_to_mod(img: np.ndarray, height: int, width: int):
    return img[:height, :width, :]


class InpaintModel:
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
        """
        self.device = device
        self.init_model(device, **kwargs)

    @abc.abstractmethod
    def init_model(self, device, **kwargs): ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool: ...

    @abc.abstractmethod
    def forward(self, image, mask):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W, 1] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    def _pad_forward(self, image, mask):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size)
        pad_mask = pad_img_to_modulo(mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size)

        result = self.forward(pad_image, pad_mask)
        # result = result[0:origin_height, 0:origin_width, :]

        # result, image, mask = self.forward_post_process(result, image, mask)

        mask = mask[:, :, np.newaxis]
        result = result[0].permute(1, 2, 0).cpu().numpy()  # 400 x 504 x 3
        result = undo_pad_to_mod(result, origin_height, origin_width)
        assert result.shape[:2] == image.shape[:2], f"{result.shape[:2]} != {image.shape[:2]}"
        # result = result * 255
        mask = (mask > 0) * 255
        result = result * mask + image * (1 - (mask / 255))
        result = np.clip(result, 0, 255).astype("uint8")
        return result

    def forward_post_process(self, result, image, mask):
        return result, image, mask


def resize_np_img(np_img, size, interpolation="bicubic"):
    assert interpolation in [
        "nearest",
        "bilinear",
        "bicubic",
    ], f"Unsupported interpolation: {interpolation}, use nearest, bilinear or bicubic."
    torch_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float() / 255
    interp_img = torch.nn.functional.interpolate(torch_img, size=size, mode=interpolation, align_corners=True)
    return (interp_img.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def resize_max_size(np_img, size_limit: int, interpolation="bicubic") -> np.ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return resize_np_img(np_img, size=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_model(url, model_md5: str = None):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 == _md5:
                print(f"Download model success, md5: {_md5}")
            else:
                try:
                    os.remove(cached_file)
                    print(
                        f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart"
                        " lama-cleaner.If you still have errors, please try download model manually first"
                        " https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
                    )
                except:
                    print(
                        f"Model md5: {_md5}, expected md5: {model_md5}, please delete {cached_file} and restart"
                        " lama-cleaner."
                    )
                exit(-1)

    return cached_file


def load_jit_model(url_or_path, device, model_md5: str):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    print(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    model.eval()
    return model


class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8

    @torch.no_grad()
    def __call__(self, image, mask):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        # boxes = boxes_from_mask(mask)
        inpaint_result = self._pad_forward(image, mask)

        return inpaint_result

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def forward(self, image, mask):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)
        return inpainted_image
