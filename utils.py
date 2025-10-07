import os

import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
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


class ImageFolderWithKey(ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return dict(image=sample, key=path.split(os.sep)[-1], label=target)


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


def save_img(img: Image, img_name: str, base_dir: str, img_class: str = None, format="PNG", img_version=None):
    """Save an image to a directory.

    Args:
        img (PIL.Image): Image to save.
        img_name (str): Relative path to the image.
        base_dir (str): Base directory to save images in.
        img_class (str, optional): Image class, if not given try to extract it from the image name in ImageNet train format. Defaults to None.
        format (str, optional): Format to save the image in. Defaults to "PNG".
        img_version (int, optional): Version of the image. Will be appended to the path. Defaults to None.

    """
    if not img_name.endswith(f".{format}"):
        img_name = f"{img_name.split('.')[0]}.{format}"
    if img_class is None:
        img_class = img_name.split("_")[0]
    if not os.path.exists(os.path.join(base_dir, img_class)):
        os.makedirs(os.path.join(base_dir, img_class), exist_ok=True)
    if img_version is not None:
        img_name = f"{img_name.split('.')[0]}_v{img_version}.{format}"
    img.save(os.path.join(base_dir, img_class, img_name), format.lower())


def already_segmented(img_name: str, base_dir: str, img_class: str = None):
    """Check if an image was already segmented.

    Args:
        img_name (str): Relative path to the image.
        base_dir (str): Base directory to save images in.
        img_class (str, optional): Image class, if not given try to extract it from the image name in ImageNet train format. Defaults to None.

    Returns:
        bool: Image was segmented already.

    """
    img_base_name = ".".join(img_name.split(".")[:-1]) if "." in img_name else img_name
    if img_class is None:
        img_class = img_name.split("_")[0]
    if not os.path.exists(os.path.join(base_dir, img_class)):
        return False
    return any(
        file.startswith(img_base_name + "_v") or file.startswith(img_base_name + ".")
        for file in os.listdir(os.path.join(base_dir, img_class))
    )
