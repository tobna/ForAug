import torch
import torchvision
from torchvision.transforms import (
    Resize,
)
from torchvision.transforms import functional as F

_image_and_target_transforms = [
    torchvision.transforms.RandomCrop,
    torchvision.transforms.RandomHorizontalFlip,
    torchvision.transforms.CenterCrop,
    torchvision.transforms.RandomRotation,
    torchvision.transforms.RandomAffine,
    torchvision.transforms.RandomResizedCrop,
    torchvision.transforms.RandomRotation,
]


def apply_dense_transforms(x, y, transforms: torchvision.transforms.transforms.Compose):
    """Apply some transfomations to both image and target.

    Args:
        x (torch.Tensor): image
        y (torch.Tensor): target (image)
        transforms (torchvision.transforms.transforms.Compose): transformations to apply

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x, y) with applyed transformations

    """
    for trans in transforms.transforms:
        if isinstance(trans, torchvision.transforms.RandomResizedCrop):
            params = trans.get_params(x, trans.scale, trans.ratio)
            x = F.resized_crop(x, *params, trans.size, trans.interpolation, antialias=trans.antialias)
            y = F.resized_crop(y.unsqueeze(0), *params, trans.size, 0).squeeze(0)  # nearest neighbor interpolation
        elif isinstance(trans, Resize):
            pre_shape = x.shape
            x = trans(x)
            if x.shape != pre_shape:
                y = F.resize(y.unsqueeze(0), trans.size, 0, trans.max_size, trans.antialias).squeeze(
                    0
                )  # nearest neighbor interpolation
        elif any(isinstance(trans, simul_transform) for simul_transform in _image_and_target_transforms):
            xy = torch.cat([x, y.unsqueeze(0).float()], dim=0)
            xy = trans(xy)
            x, y = xy[:-1], xy[-1].long()
        elif isinstance(trans, torchvision.transforms.ToTensor):
            if not isinstance(x, torch.Tensor):
                x = trans(x)
        else:
            x = trans(x)

    return x, y
