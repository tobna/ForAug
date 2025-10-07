import argparse
import json
import os
from datetime import datetime
from math import ceil

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import ToPILImage, ToTensor
from tqdm.auto import tqdm

from attentive_eraser import AttentiveEraser
from grounded_segmentation import grounded_segmentation
from infill_lama import LaMa
from utils import ImageFolderWithKey, already_segmented, save_img
from wordnet_tree import WNTree


def _collate_single(data):
    return data[0]


def _collate_multiple(datas):
    return {
        "images": [data["image"] for data in datas],
        "keys": [data["key"] for data in datas],
        "labels": [data["label"] for data in datas],
    }


def smallest_crop(image: torch.Tensor, mask: torch.Tensor):
    """Crops the image to just so fit the mask given.

    Mask and image have to be of the same size.

    Args:
        image (torch.Tensor): image to crop
        mask (torch.Tensor): cropping to mask

    Returns:
        torch.Tensor: cropped image

    """
    if len(mask.shape) == 3:
        assert mask.shape[0] == 1
        mask = mask[0]
    assert (
        len(image.shape) == 3 and len(mask.shape) == 2 and image.shape[1:] == mask.shape
    ), f"Invalid shapes: {image.shape}, {mask.shape}"
    dim0_indices = mask.sum(dim=0).nonzero()
    dim1_indices = mask.sum(dim=1).nonzero()

    return image[
        :,
        dim1_indices.min().item() : dim1_indices.max().item() + 1,
        dim0_indices.min().item() : dim0_indices.max().item() + 1,
    ]


def _synset_to_prompt(synset, tree, parent_in_prompt):
    prompt = f"an {synset.print_name}." if synset.print_name[0] in "aeiou" else f"a {synset.print_name}."
    if synset.parent_id is None or not parent_in_prompt:
        return prompt
    parent = tree[synset.parent_id]
    return prompt[:-1] + f", a type of {parent.print_name}."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounded Segmentation")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["imagenet", "imagenet-val", "tinyimagenet", "tinyimagenet-val"],
        default="imagenet",
        help="Dataset to use",
    )
    parser.add_argument("-df", "--dataset-folder", required=True, help="Directory of the dataset")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Batch size (currently only works with BS=1)")
    parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "-p", "--processes", type=int, default=1, help="Number of processes that are used to process the data"
    )
    parser.add_argument("-id", type=int, default=0, help="ID of this process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files, instead of skipping them")
    parser.add_argument(
        "--parent_labels", type=int, default=2, help="Number of parent labels to use; steps to go up the tree"
    )
    parser.add_argument("--output-ims", choices=["best", "all"], default="all", help="Output all or best masks")
    parser.add_argument("--mask-merge-threshold", type=float, default=0.9, help="Threshold on IoU for merging masks")
    parser.add_argument(
        "--parent-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include parent label in the prompt",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["LaMa", "AttErase"],
        default="LaMa",
        help="Model to use for erasing/infilling. Defaults to LaMa.",
    )

    args = parser.parse_args()
    assert args.batch_size == 1, "Batch size must be 1 for grounded segmentation (for now)"

    dataset = args.dataset.lower()
    part = "val" if dataset.endswith("-val") else "train"
    dataset = ImageFolderWithKey(args.dataset_folder)

    assert 0 <= args.id < args.processes, "ID must be in the range [0, processes)"
    if args.processes > 1:
        # take the id-th subset
        partlen = ceil(len(dataset) / args.processes)
        start_id = args.id * partlen
        end_id = min((args.id + 1) * partlen, len(dataset))
        dataset = torch.utils.data.Subset(dataset, list(range(start_id, end_id)))

    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.debug,
        drop_last=False,
        num_workers=10,
        collate_fn=_collate_single if args.batch_size == 1 else _collate_multiple,
    )

    infill_model = (
        LaMa(device="cuda" if torch.cuda.is_available() else "cpu") if args.model == "LaMa" else AttentiveEraser()
    )

    if args.dataset.startswith("imagenet"):
        with open("wordnet_data/imagenet1k_synsets.json", "r") as f:
            id_to_synset = json.load(f)
        id_to_synset = {int(k): v for k, v in id_to_synset.items()}
    elif args.dataset.startswith("tinyimagenet"):
        with open("wordnet_data/tinyimagenet_synset_names.txt", "r") as f:
            synsets = f.readlines()
        id_to_synset = [int(synset.split(":")[0].strip()[1:]) for synset in synsets]
        id_to_synset = sorted(id_to_synset)

    wordnet = WNTree.load("wordnet_data/imagenet21k+1k_masses_tree.json")

    # create the necessary folders
    os.makedirs(os.path.join(args.output, part, "foregrounds"), exist_ok=True)
    os.makedirs(os.path.join(args.output, part, "backgrounds"), exist_ok=True)
    os.makedirs(os.path.join(args.output, part, "no_detect"), exist_ok=True)
    os.makedirs(os.path.join(args.output, part, "error"), exist_ok=True)

    man_print_progress = int(os.environ.get("TQDM_DISABLE", "0")) == 1

    print(f"Starting segmentation @ {datetime.now()}")
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        if args.debug and i > 10:
            break

        if man_print_progress and i % 200 == 0:
            print(f"{datetime.now()} \t process {args.id}/{args.processes}: \t sample: {i}/{len(dataset)}", flush=True)

        image = data["image"]
        key = data["key"]

        if args.dataset.endswith("-val"):
            img_class = f"n{id_to_synset[data['label']]:08d}"  # overwrite the synset id on the val set
        else:
            img_class = key.split("/")[-1].split("_")[0]

        if (
            already_segmented(key, os.path.join(args.output, part, "foregrounds"), img_class=img_class)
            and not args.overwrite
        ):
            continue

        # get the next 3 labels up in the wordnet tree
        label_id = data["label"]
        offset = id_to_synset[label_id]  # int(data["key"].split("_")[0][1:])
        synset = wordnet[offset]
        labels = [_synset_to_prompt(synset, wordnet, args.parent_in_prompt)]
        while len(labels) < 1 + args.parent_labels and synset.parent_id is not None:
            synset = wordnet[synset.parent_id]
            _label = _synset_to_prompt(synset, wordnet, args.parent_in_prompt)
            _label = _label.replace("_", " ")
            labels.append(_label)

        assert len(labels) > 0, f"No labels for {key}"

        # detect and segment
        image_tensor, detections = grounded_segmentation(
            image, labels, threshold=args.threshold, polygon_refinement=True
        )

        # merge all detections for the same label
        masks = {}
        for detection in detections:
            if detection.label not in masks:
                masks[detection.label] = detection.mask
            else:
                masks[detection.label] |= detection.mask
        labels = list(masks.keys())
        masks = [masks[lbl] for lbl in labels]
        assert len(masks) == len(
            labels
        ), f"Number of masks ({len(masks)}) != number of labels ({len(labels)}); detections: {detections}"

        if args.debug:
            for detected_mask, label in zip(masks, labels):
                detected_mask = torch.from_numpy(detected_mask).unsqueeze(0) / 255
                mask_foreground = torch.cat((image_tensor * detected_mask, detected_mask), dim=0)
                ToPILImage()(mask_foreground).save(f"examples/{key}_{label}_masked.png")

        if len(masks) == 0:
            tqdm.write(f"No detections for {key}; skipping")
            save_img(image, key, os.path.join(args.output, part, "no_detect"), img_class=img_class, format="JPEG")
            continue
        elif len(masks) == 1:
            # max_overlap_mask = masks[0]
            masks = [masks[0]]
        elif args.output_ims == "best":
            # find the 2 masks with the largest overlap
            max_overlap = 0
            max_overlap_mask = None
            using_labels = None
            for i, mask1 in enumerate(masks):
                for j, mask2 in enumerate(masks):
                    if i >= j:
                        continue
                    iou = (mask1 & mask2).sum() / (mask1 | mask2).sum()
                    if iou > max_overlap:
                        max_overlap = iou
                        max_overlap_mask = mask1 | mask2
                        using_labels = f"{labels[i]} & {labels[j]}"
            if args.debug:
                tqdm.write(f"{key}:\tMax overlap: {max_overlap}, using {using_labels}")
            if max_overlap_mask is None:
                # assert len(masks) > 0 and len(masks) == len(labels), f"No detections for {key}"
                max_overlap_mask = masks[0]
            masks = [max_overlap_mask]
        else:
            # merge masks that are too similar
            has_changed = True
            while has_changed:
                has_changed = False
                for i, mask1 in enumerate(masks):
                    for j, mask2 in enumerate(masks):
                        if i >= j:
                            continue
                        iou = (mask1 & mask2).sum() / (mask1 | mask2).sum()
                        if iou > args.mask_merge_threshold:
                            masks[i] |= masks[j]
                            masks.pop(j)
                            labels.pop(j)
                            has_changed = True
                            break
                    if has_changed:
                        break

        for mask_idx, mask_array in enumerate(masks):
            mask = torch.from_numpy(mask_array).unsqueeze(0) / 255

            if args.debug:
                mask_image = ToPILImage()(mask)
                mask_image.save(f"examples/{key}_mask.png")

            # foreground = image_tensor
            foreground = torch.cat((image_tensor * mask, mask), dim=0)
            foreground = ToPILImage()(foreground)
            mask_img = foreground.split()[-1]
            foreground = smallest_crop(ToTensor()(foreground), ToTensor()(mask_img))  # TODO: fix smallest crop
            fg_img = ToPILImage()(foreground)

            background = (1 - mask) * image_tensor
            bg_image = ToPILImage()(background)

            # 2. infill background
            mask_image = Image.fromarray(mask_array)
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=7))
            try:
                infilled_bg = infill_model(np.array(bg_image), np.array(mask_image))
                infilled_bg = Image.fromarray(np.uint8(infilled_bg))
            except RuntimeError as e:
                tqdm.write(
                    f"Error infilling {key}: bg_image.shape={np.array(bg_image).shape},"
                    f" mask_image.shape={np.array(mask_image).shape}\n{e}"
                )
                save_img(
                    image,
                    key,
                    os.path.join(args.output, part, "error"),
                    img_class=img_class,
                    img_version=mask_idx if len(masks) > 1 else None,
                    format="JPEG",
                )
                infilled_bg = None

            if args.debug:
                data["image"].save(f"examples/{key}_orig.png")
                fg_img.save(f"examples/{key}_fg.png")
                infilled_bg.save(f"examples/{key}_bg.png")
            else:
                # save files
                class_name = key.split("_")[0]
                save_img(
                    fg_img,
                    key,
                    os.path.join(args.output, part, "foregrounds"),
                    img_class=img_class,
                    img_version=mask_idx if len(masks) > 1 else None,
                    format="WEBP",
                )
                if infilled_bg is not None:
                    save_img(
                        infilled_bg,
                        key,
                        os.path.join(args.output, part, "backgrounds"),
                        img_class=img_class,
                        img_version=mask_idx if len(masks) > 1 else None,
                        format="JPEG",
                    )
    if man_print_progress:
        print(
            f"{datetime.now()} \t process {args.id}/{args.processes}: \t done with all {len(dataset)} samples",
            flush=True,
        )
