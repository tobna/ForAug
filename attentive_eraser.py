from copy import copy

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from PIL import Image, ImageFilter
from torchvision.transforms.functional import gaussian_blur, to_tensor


def _preprocess_image(image, device, dtype=torch.float16):
    image = to_tensor(image)
    image = image.unsqueeze_(0).float() * 2 - 1  # [0,1] --> [-1,1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (1024, 1024))
    return image.to(dtype).to(device)


def _preprocess_mask(mask, device, dtype=torch.float16):
    mask = to_tensor(mask.convert("L"))
    mask = mask.unsqueeze_(0).float()  # 0 or 1
    mask = F.interpolate(mask, (1024, 1024))
    mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    return mask.to(dtype).to(device)


class AttentiveEraser:
    """Attentive Eraser Pipeline + Pre- and Post-Processing."""

    prompt = ""
    dtype = torch.float16

    def __init__(self, num_steps=50, device=None):
        """Create the attentive eraser.

        Args:
            num_steps (int, optional): Number of steps in the diffusion process. Will start at 20%. Defaults to 100.
            device (_type_, optional): Device to run on. Defaults to 'cuda' if available, else 'cpu'.

        """
        self.num_steps = num_steps

        self.device = device if device else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
        )
        self.model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            custom_pipeline="./pipelines/pipeline_stable_diffusion_xl_attentive_eraser.py",
            scheduler=scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_model_cpu_offload()

    def __call__(self, image, mask):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        prep_img = _preprocess_image(image, self.device, self.dtype)
        prep_mask = _preprocess_mask(mask, self.device, self.dtype)
        orig_shape = image.size

        diff_img = (
            self.pipeline(
                prompt=self.prompt,
                image=prep_img,
                mask_image=prep_mask,
                height=1024,
                width=1024,
                AAS=True,  # enable AAS
                strength=0.8,  # inpainting strength
                rm_guidance_scale=9,  # removal guidance scale
                ss_steps=9,  # similarity suppression steps
                ss_scale=0.3,  # similarity suppression scale
                AAS_start_step=0,  # AAS start step
                AAS_start_layer=34,  # AAS start layer
                AAS_end_layer=70,  # AAS end layer
                num_inference_steps=self.num_steps,  # number of inference steps # AAS_end_step = int(strength*num_inference_steps)
                guidance_scale=1,
            )
            .images[0]
            .resize(orig_shape)
        )

        paste_mask = Image.fromarray(np.array(mask) > 0 * 255)
        paste_mask = paste_mask.convert("RGB").filter(ImageFilter.GaussianBlur(radius=10)).convert("L")
        out_img = copy(image)
        out_img.paste(diff_img, (0, 0), paste_mask)
        return out_img
