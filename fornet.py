import json
import logging
import os
import zipfile
from io import BytesIO
from math import floor

import numpy as np
import PIL

try:
    from datadings.torch import Compose
except ImportError:
    from torchvision.transforms import Compose
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms as T

from utils import apply_dense_transforms


class ForNet(Dataset):
    """Recombine ImageNet forgrounds and backgrounds.

    Note:
        This dataset has exactly the ImageNet classes.

    """

    _back_combs = ["same", "all", "original"]
    _bg_transforms = {T.RandomCrop, T.CenterCrop, T.Resize, T.RandomResizedCrop}

    def __init__(
        self,
        root,
        transform=None,
        train=True,
        target_transform=None,
        background_combination="all",
        fg_scale_jitter=0.3,
        fg_transform=None,
        pruning_ratio=0.8,
        return_fg_masks=False,
        fg_size_mode="range",
        fg_bates_n=1,  # uniform distribution
        paste_pre_transform=True,
        mask_smoothing_sigma=4.0,
        rel_jut_out=0.0,
        fg_in_nonant=None,
        size_fact=1.0,
        orig_img_prob=0.0,
        orig_ds=None,
        _orig_ds_file_type="JPEG",
        epochs=0,
    ):
        """Create RecombinationNet dataset.

        Args:
            root (str): Root folder for the dataset.
            transform (T.Collate | list, optional): Transform to apply to the image. Defaults to None.
            train (bool, optional): On the train set (False -> val set). Defaults to True.
            target_transform (T.Collate | list, optional): Transform to apply to the target values. Defaults to None.
            background_combination (str, optional): Which backgrounds to combine with foregrounds. Defaults to "same".
            fg_scale_jitter (tuple, optional): How much should the size of the foreground be changed (random ratio). Defaults to (0.1, 0.8).
            fg_transform (_type_, optional): Transform to apply to the foreground before applying to the background. This is supposed to be a random rotation, mainly. Defaults to None.
            pruning_ratio (float, optional): For pruning backgrounds, with (foreground size/background size) >= <pruning_ratio>. Backgrounds from images that contain very large foreground objects are mostly computer generated and therefore relatively unnatural. Defaults to full dataset.
            return_fg_masks (bool, optional): Return the foreground masks. Defaults to False.
            fg_size_mode (str, optional): How to determine the size of the foreground, based on the foreground sizes of the foreground and background images. Defaults to "max".
            fg_bates_n (int, optional): Bates parameter for the distribution of the object position in the foreground. Defaults to 1 (uniform distribution). The higher the value, the more likely the object is in the center. For fg_bates_n = 0, the object is always in the center.
            paste_pre_transform (bool, optional): Paste the foreground onto the background before applying the transform. If false, the background will be cropped and resized before pasting the foreground. Defaults to False.
            mask_smoothing_sigma (float, optional): Sigma for the Gaussian blur of the mask edge. Defaults to 0.0. Try 2.0 or 4.0?
            rel_jut_out (float, optional): How much is the foreground allowed to stand/jut out of the background (and then cut off). Defaults to 0.0.
            fg_in_nonant (int, optional): If not None, the foreground will be placed in a specific nonant (0-8) of the image. Defaults to None.
            size_fact (float, optional): Factor to multiply the size of the foreground with. For size bias calculation. Defaults to 1.0.
            orig_img_prob (float | str, optional): Probability to use the original image, instead of the fg-bg recombinations. Or probability strategy: cos, linear, revlinear. Defaults to 0.0.
            orig_ds (Dataset | str, optional): Original dataset (without transforms) to use for the original images. Or path to the image folders. Defaults to None.
            _orig_ds_file_type (str, optional): File type of the original dataset. Defaults to "JPEG".
            epochs (int, optional): Number of epochs to train on. Used for linear increase of orig_img_prob.

        Note:
            For more information on the bates distribution, see https://en.wikipedia.org/wiki/Bates_distribution.
            For fg_bats_n < 0, we take extend the bates dirstribution to focus more and more on the edges. This is done by sampling B ~ Bates(|fg_bates_n|) and then passing through f(x) = x + 0.5 - floor(x + 0.5).

            For the list of transformations that will be applied to the background only (if paste_pre_transform=False), see RecombinationNet._bg_transforms.

            A nonant in this case refers to a square in a 3x3 grid dividing the image.

        """
        self.init_dataset_loading(root, train, pruning_ratio)

        assert (
            background_combination in self._back_combs
        ), f"background_combination={background_combination} is not supported. Use one of {self._back_combs}"

        self.root = root
        self.train = train
        self.background_combination = background_combination
        self.fg_scale_jitter = fg_scale_jitter
        self.fg_transform = fg_transform
        self.return_fg_masks = return_fg_masks
        self.paste_pre_transform = paste_pre_transform
        self.mask_smoothing_sigma = mask_smoothing_sigma
        self.rel_jut_out = rel_jut_out
        self.size_fact = size_fact
        self.fg_in_nonant = fg_in_nonant
        assert fg_in_nonant is None or -1 <= fg_in_nonant < 9, f"fg_in_nonant={fg_in_nonant} not in [0, 8] or None"

        self.orig_img_prob = orig_img_prob
        if orig_img_prob != 0.0:
            assert (isinstance(orig_img_prob, float) and orig_img_prob > 0.0) or orig_img_prob in [
                "linear",
                "cos",
                "revlinear",
            ]
            assert orig_ds is not None, "orig_ds must be provided if orig_img_prob > 0.0"
            assert not return_fg_masks, "can't provide fg masks for original images (yet)"

            self.epochs = epochs
            self._epoch = 0

        assert fg_size_mode in [
            "max",
            "min",
            "mean",
            "range",
        ], f"fg_size_mode={fg_size_mode} not supported; use one of ['max', 'min', 'mean', 'range']"
        self.fg_size_mode = fg_size_mode
        self.fg_bates_n = fg_bates_n

        # do cropping and resizing mainly on background; paste foreground on top later
        if not paste_pre_transform:
            if isinstance(transform, (T.Compose, Compose)):
                transform = transform.transforms
            elif transform is None:
                transform = []

            for tf in transform:
                self.bg_transform = []
                self.join_transform = []
                if isinstance(tf, tuple(self._bg_transforms)) and not paste_pre_transform:
                    self.bg_transform.append(tf)
                else:
                    self.join_transform.append(tf)

                self.bg_transform = T.Compose(self.bg_transform)
                self.join_transform = T.Compose(self.join_transform)
        else:
            self.join_transform = transform

        self.trgt_map = {cls: i for i, cls in enumerate(self.classes)}

        self.target_transform = target_transform

        self.cls_to_allowed_bg = {}
        for bg_file in self.backgrounds:
            if background_combination == "same":
                bg_cls = bg_file.split("/")[-2]
                if bg_cls not in self.cls_to_allowed_bg:
                    self.cls_to_allowed_bg[bg_cls] = []
                self.cls_to_allowed_bg[bg_cls].append(bg_file)

        if background_combination == "same":
            for cls_code in self.classes:
                if cls_code not in self.cls_to_allowed_bg or len(self.cls_to_allowed_bg[cls_code]) == 0:
                    self.cls_to_allowed_bg[cls_code] = [self.all_backgrounds[cls_code]]
                    print(f"Warning: no background for class {cls_code}, using {self.all_backgrounds[cls_code]}")

        self._zf = {}

    def init_dataset_loading(self, root, train, pruning_ratio=1.1, orig_img_prob=0.0, orig_ds=None):
        """Initilaize everything thats needed for loading images (both foregrounds and backgrounds).

        Sets:
            - self.foregrounds: List of allowed foregrounds. Required for len(self)
            - self.backgrounds: List of allowed backgrounds (depends on the pruning ratio). Required for certain recombination modes.
            - self.all_backgrounds: List of all possible backgrounds (in case there is a class where everything would get pruned). Required for certain recombination modes in case we prune too much.
            - self.classes: List of classes. Nice to have/expose for other code.
            - self.fg_bg_ratios: Stores foreground ratios for the imagenet images to use in the recombination step. Depends on the implementation of image loading
            - self.orig_ds: To load the original dataset. Depends on the implementation of image loading.
        """
        if (not os.path.exists(f"{root}/backgrounds_{'train' if train else 'val'}.zip")) and os.path.exists(
            os.path.join(root, "train" if train else "val", "backgrounds")
        ):
            self._mode = "folder"
        else:
            self._mode = "zip"

        if self._mode == "zip":
            try:
                with zipfile.ZipFile(f"{root}/backgrounds_{'train' if train else 'val'}.zip", "r") as bg_zip:
                    self.backgrounds = [f for f in bg_zip.namelist() if f.endswith(".JPEG")]
                with zipfile.ZipFile(f"{root}/foregrounds_{'train' if train else 'val'}.zip", "r") as fg_zip:
                    self.foregrounds = [f for f in fg_zip.namelist() if f.endswith(".WEBP")]
            except FileNotFoundError as e:
                print(
                    f"RecombinationNet: {e}. Make sure to have the background and foreground zips in the root"
                    f" directory: found {os.listdir(root)}"
                )
                raise e
            classes = set([f.split("/")[-2] for f in self.foregrounds])
        else:
            classes = set(os.listdir(os.path.join(root, "train" if train else "val", "foregrounds")))
            foregrounds = []
            backgrounds = []
            for cls in classes:
                foregrounds.extend(
                    [
                        f"{cls}/{f}"
                        for f in os.listdir(os.path.join(root, "train" if train else "val", "foregrounds", cls))
                    ]
                )
                backgrounds.extend(
                    [
                        f"{cls}/{f}"
                        for f in os.listdir(os.path.join(root, "train" if train else "val", "backgrounds", cls))
                    ]
                )
            self.foregrounds = foregrounds
            self.backgrounds = backgrounds

        self.classes = sorted(list(classes), key=lambda x: int(x[1:]))

        assert os.path.exists(f"{root}/fg_bg_ratios_{'train' if train else 'val'}.json"), (
            f"{root}/fg_bg_ratios_{'train' if train else 'val'}.json not found, provide the information or set"
            " pruning_ratio=1.0"
        )
        with open(f"{root}/fg_bg_ratios_{'train' if train else 'val'}.json", "r") as f:
            self.fg_bg_ratios = json.load(f)
            if self._mode == "folder":
                self.fg_bg_ratios = {"/".join(key.split("/")[-2:]): val for key, val in self.fg_bg_ratios.items()}
                print(f"Renamed fg_bg_ratios keys to {list(self.fg_bg_ratios.keys())[:3]}...")

        if pruning_ratio <= 1.0:
            self.all_backgrounds = {}
            for bg_file in self.backgrounds:
                bg_cls = bg_file.split("/")[-2]
                self.all_backgrounds[bg_cls] = bg_file
            self.backgrounds = [
                bg for bg in self.backgrounds if bg in self.fg_bg_ratios and self.fg_bg_ratios[bg] < pruning_ratio
            ]
            # print(
            #     f"RecombinationNet: keep {len(self.backgrounds)} of {len(self.fg_bg_ratios)} backgrounds with pr {pruning_ratio}"
            # )
        if orig_img_prob != 0.0:
            assert os.path.exists(os.path.join(root, f"{'train' if train else 'val'}_indices.json")) or isinstance(
                orig_ds, str
            ), f"{root}/{'train' if train else 'val'}_indices.json must be provided if orig_ds is a dataset"
            if not isinstance(orig_ds, str):
                with open(os.path.join(root, f"{'train' if train else 'val'}_indices.json"), "r") as f:
                    self.key_to_orig_idx = json.load(f)
            else:
                if not (
                    orig_ds.endswith("train" if train else "val") or orig_ds.endswith("train/" if train else "val/")
                ):
                    orig_ds = f"{orig_ds}/{'train' if train else 'val'}"
                self.key_to_orig_idx = None
                self._orig_ds_file_type = _orig_ds_file_type
            self.orig_ds = orig_ds

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        assert value < self.epochs, f"epoch={value} must be < epochs={self.epochs}"
        self._epoch = value

    def __len__(self):
        """Size of the dataset.

        Returns:
            int: number of foregrounds

        """
        return len(self.foregrounds)

    def num_classes(self):
        return len(self.classes)

    def _get_fg(self, idx):
        worker_id = self._wrkr_info()

        fg_file = self.foregrounds[idx]
        with self._zf[worker_id]["fg"].open(fg_file) as f:
            fg_data = BytesIO(f.read())
        return Image.open(fg_data)

    def _wrkr_info(self):
        worker_id = get_worker_info().id if get_worker_info() else 0

        if worker_id not in self._zf and self._mode == "zip":
            self._zf[worker_id] = {
                "bg": zipfile.ZipFile(f"{self.root}/backgrounds_{'train' if self.train else 'val'}.zip", "r"),
                "fg": zipfile.ZipFile(f"{self.root}/foregrounds_{'train' if self.train else 'val'}.zip", "r"),
            }
        return worker_id

    def recombine(self, fg_img, bg_img, fg_bg_ratio_foreground: float, fg_bg_ratio_background: float):
        """Recombine foreground and background to a new image.

        Args:
            fg_img (Image): Image of foreground object with alpha channel
            bg_img (Image): Background image to paste the foreground on (mode: RGB)
            fg_bg_ratio_foreground (float): Size of the foreground image (total area ignoring alpha channel) relative to area of the original image this foreground object came from.
            fg_bg_ratio_background (float): Size of the object that was originally in the background image relative to area of the background image. See fg_bg_ratio_foreground.

        Returns:
            tuple(Image, Tensor | None): Combined image and foreground mask if self.return_fg_masks is set. Otherwise the mask will just be None.
        """
        orig_fg_ratio = fg_bg_ratio_foreground
        bg_fg_ratio = fg_bg_ratio_background
        bg_size = bg_img.size

        if self.fg_transform:
            fg_img = self.fg_transform(fg_img)

        # How much does the foreground fill it's image? How much of the foreground image is empty?
        fg_size_factor = T.ToTensor()(fg_img.split()[-1]).mean().item()

        # total area of the background image
        bg_area = bg_size[0] * bg_size[1]
        if self.fg_in_nonant is not None:
            bg_area = bg_area / 9

        # How much of the output image should be filled by the foreground object? Depends on the size mode.
        if self.fg_size_mode == "max":
            goal_fg_ratio_lower = goal_fg_ratio_upper = max(orig_fg_ratio, bg_fg_ratio)
        elif self.fg_size_mode == "min":
            goal_fg_ratio_lower = goal_fg_ratio_upper = min(orig_fg_ratio, bg_fg_ratio)
        elif self.fg_size_mode == "mean":
            goal_fg_ratio_lower = goal_fg_ratio_upper = (orig_fg_ratio + bg_fg_ratio) / 2
        else:
            # range
            goal_fg_ratio_lower = min(orig_fg_ratio, bg_fg_ratio)
            goal_fg_ratio_upper = max(orig_fg_ratio, bg_fg_ratio)

        fg_scale = (
            np.random.uniform(
                goal_fg_ratio_lower * (1 - self.fg_scale_jitter), goal_fg_ratio_upper * (1 + self.fg_scale_jitter)
            )
            / fg_size_factor
            * self.size_fact
        )

        # scale the foreground image accordingly, while keeping the relative side length
        goal_shape_y = round(np.sqrt(bg_area * fg_scale * fg_img.size[1] / fg_img.size[0]))
        goal_shape_x = round(np.sqrt(bg_area * fg_scale * fg_img.size[0] / fg_img.size[1]))

        fg_img = fg_img.resize((goal_shape_x, goal_shape_y))

        if fg_img.size[0] > bg_size[0] or fg_img.size[1] > bg_size[1]:
            # random crop to fit
            goal_w, goal_h = (min(fg_img.size[0], bg_size[0]), min(fg_img.size[1], bg_size[1]))
            fg_img = T.RandomCrop((goal_h, goal_w))(fg_img) if self.train else T.CenterCrop((goal_h, goal_w))(fg_img)

        # paste the resized foreground onto the background
        # select the relative position in the image
        z1, z2 = (
            (
                np.random.uniform(0, 1, abs(self.fg_bates_n)).mean(),  # bates distribution n=1 => uniform
                np.random.uniform(0, 1, abs(self.fg_bates_n)).mean(),
            )
            if self.fg_bates_n != 0
            else (0.5, 0.5)
        )
        if self.fg_bates_n < 0:
            z1 = z1 + 0.5 - floor(z1 + 0.5)
            z2 = z2 + 0.5 - floor(z2 + 0.5)

        # get allowed position borders (depending on the sizes of foreground and background image)
        x_min = -self.rel_jut_out * fg_img.size[0]
        x_max = bg_size[0] - fg_img.size[0] * (1 - self.rel_jut_out)
        y_min = -self.rel_jut_out * fg_img.size[1]
        y_max = bg_size[1] - fg_img.size[1] * (1 - self.rel_jut_out)

        if self.fg_in_nonant is not None and self.fg_in_nonant >= 0:
            x_min = (self.fg_in_nonant % 3) * bg_size[0] / 3
            x_max = ((self.fg_in_nonant % 3) + 1) * bg_size[0] / 3 - fg_img.size[0]
            y_min = (self.fg_in_nonant // 3) * bg_size[1] / 3
            y_max = ((self.fg_in_nonant // 3) + 1) * bg_size[1] / 3 - fg_img.size[1]

        if x_min > x_max:
            x_min = x_max = (x_min + x_max) / 2
        if y_min > y_max:
            y_min = y_max = (y_min + y_max) / 2

        # convert to absolute position
        offs_x = round(z1 * (x_max - x_min) + x_min)
        offs_y = round(z2 * (y_max - y_min) + y_min)

        # paste the foreground object onto the background image
        paste_mask = fg_img.split()[-1]
        if self.mask_smoothing_sigma > 0.0:
            sigma = (np.random.rand() * 0.9 + 0.1) * self.mask_smoothing_sigma
            paste_mask = paste_mask.filter(ImageFilter.GaussianBlur(radius=sigma))
            paste_mask = paste_mask.point(lambda p: 2 * p - 255 if p > 128 else 0)

        bg_img.paste(fg_img.convert("RGB"), (offs_x, offs_y), paste_mask)
        bg_img = bg_img.convert("RGB")

        # handling for also returning the foreground mask for the new image
        if self.return_fg_masks:
            fg_mask = Image.new("L", bg_size, 0)
            fg_mask.paste(paste_mask, (offs_x, offs_y))

            fg_mask = T.ToTensor()(fg_mask)[0]
            return bg_img, fg_mask

        return bg_img, None

    def __getitem__(self, idx):
        """Get the foreground at index idx and combine it with a (random) background.

        Args:
            idx (int): foreground index

        Returns:
            torch.Tensor, torch.Tensor: image, target

        """
        worker_id = self._wrkr_info()
        fg_file = self.foregrounds[idx]
        trgt_cls = fg_file.split("/")[-2]

        if (
            (self.orig_img_prob == "linear" and np.random.rand() < self._epoch / self.epochs)
            or (self.orig_img_prob == "revlinear" and np.random.rand() < (self._epoch - self.epochs) / self.epochs)
            or (self.orig_img_prob == "cos" and np.random.rand() > np.cos(np.pi * self._epoch / (2 * self.epochs)))
            or (
                isinstance(self.orig_img_prob, float)
                and self.orig_img_prob > 0.0
                and np.random.rand() < self.orig_img_prob
            )
        ):
            # return original image
            data_key = f"{trgt_cls}/{fg_file.split('/')[-1].split('.')[0]}"

            if isinstance(self.orig_ds, str):
                image_file = os.path.join(self.orig_ds, f"{data_key}.{self._orig_ds_file_type}")
                orig_img = Image.open(image_file).convert("RGB")
            else:
                orig_data = self.orig_ds[self.key_to_orig_idx[data_key]]
                orig_img = orig_data["image"] if isinstance(orig_data, dict) else orig_data[0]

            if self.bg_transform:
                orig_img = self.bg_transform(orig_img)
            if self.join_transform:
                orig_img = self.join_transform(orig_img)
            trgt = self.trgt_map[trgt_cls]
            if self.target_transform:
                trgt = self.target_transform(trgt)
            return orig_img, trgt

        # return ForNet image
        if self._mode == "zip":
            with self._zf[worker_id]["fg"].open(fg_file) as f:
                fg_data = BytesIO(f.read())
                try:
                    fg_img = Image.open(fg_data).convert("RGBA")
                except PIL.UnidentifiedImageError as e:
                    logging.error(f"Error with idx={idx}, file={self.foregrounds[idx]}")
                    raise e
        else:
            fg_img = Image.open(
                os.path.join(self.root, "train" if self.train else "val", "foregrounds", fg_file)
            ).convert("RGBA")

        if self.background_combination == "all":
            bg_idx = np.random.randint(len(self.backgrounds))
            bg_file = self.backgrounds[bg_idx]
        elif self.background_combination == "original":
            bg_file = fg_file.replace("foregrounds", "backgrounds").replace("WEBP", "JPEG")
        else:
            bg_idx = np.random.randint(len(self.cls_to_allowed_bg[trgt_cls]))
            bg_file = self.cls_to_allowed_bg[trgt_cls][bg_idx]

        if self._mode == "zip":
            with self._zf[worker_id]["bg"].open(bg_file) as f:
                bg_data = BytesIO(f.read())
                bg_img = Image.open(bg_data).convert("RGB")
        else:
            bg_img = Image.open(
                os.path.join(self.root, "train" if self.train else "val", "backgrounds", bg_file)
            ).convert("RGB")

        if not self.paste_pre_transform:
            bg_img = self.bg_transform(bg_img)

        # print(f"background: size={bg_size} area={bg_area}")
        # print(f"fg_file={fg_file}, fg_bg_ratio_keys={list(self.fg_bg_ratios.keys())[:3]}...")
        orig_fg_ratio = self.fg_bg_ratios[fg_file.replace("foregrounds", "backgrounds").replace("WEBP", "JPEG")]
        bg_fg_ratio = self.fg_bg_ratios[bg_file]

        comb_img, fg_mask = self.recombine(
            fg_img=fg_img, bg_img=bg_img, fg_bg_ratio_foreground=orig_fg_ratio, fg_bg_ratio_background=bg_fg_ratio
        )

        if self.return_fg_masks:
            comb_img = T.ToTensor()(comb_img)

            if self.join_transform:
                # img_mask_stack = torch.cat([bg_img, fg_mask.unsqueeze(0)], dim=0)
                # img_mask_stack = self.join_transform(img_mask_stack)
                # bg_img, fg_mask = img_mask_stack[:-1], img_mask_stack[-1]
                comb_img, fg_mask = apply_dense_transforms(comb_img, fg_mask, self.join_transform)
        elif self.join_transform:
            comb_img = self.join_transform(comb_img)

        if trgt_cls not in self.trgt_map:
            raise ValueError(f"trgt_cls={trgt_cls} not in trgt_map: {self.trgt_map}")
        trgt = self.trgt_map[trgt_cls]
        if self.target_transform:
            trgt = self.target_transform(trgt)

        if self.return_fg_masks:
            return comb_img, trgt, fg_mask

        return comb_img, trgt
