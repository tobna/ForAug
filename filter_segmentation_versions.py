import argparse
import os
from math import log

import numpy as np
import torch
from PIL import Image
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    ViT_L_16_Weights,
    resnet50,
    resnet101,
    swin_s,
    swin_t,
    vit_b_16,
    vit_l_16,
)
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm.auto import tqdm


def score_f(  # noqa: D103
    idx,
    bg_probs,
    mean_probs,
    fg_ratio,
    max_idx,
    mean_probs_exp=1,
    bg_probs_exp=1,
    fg_ratio_exp=2,
    opt_fg_ratio=0.1,
    idx_exp=0.1,
):
    return (
        log(mean_probs) * mean_probs_exp
        + log(1 - bg_probs) * bg_probs_exp
        + log(1 - abs(fg_ratio - opt_fg_ratio)) * fg_ratio_exp
        + log(1 - idx / (max_idx + 1)) * idx_exp
    )


parser = argparse.ArgumentParser(description="Inspect image versions")
parser.add_argument("-f", "--base-folder", type=str, required=True, help="Base folder to inspect")
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for model inspection. Will be 1 background and batch_size - 1 foregrounds",
)
parser.add_argument("--imsize", type=int, default=224, help="Image size")

args = parser.parse_args()

bg_folder = os.path.join(args.base_folder, "backgrounds")
fg_folder = os.path.join(args.base_folder, "foregrounds")

classes = os.listdir(fg_folder)
classes = sorted(classes, key=lambda x: int(x[1:]))
assert len(classes) in [200, 1_000], f"Expected 200 (TinyImageNet) or 1_000 (ImageNet) classes, got {len(classes)}"

total_images = set()
for in_cls in classes:
    cls_images = {
        os.path.join(in_cls, "_".join(img.split(".")[0].split("_")[:-1]))
        for img in os.listdir(os.path.join(fg_folder, in_cls))
        if img.split(".")[0].split("_")[-1].startswith("v")
    }
    total_images.update(cls_images)
total_images = list(total_images)


# in_cls to print name/lemma
with open(os.path.join("wordnet_data", "tinyimagenet_synset_names.txt"), "r") as f:
    in_cls_to_name = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in f.readlines() if len(line) > 2}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Load in specific models for filtering
inspection_models = [
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
    resnet101(weights=ResNet101_Weights.IMAGENET1K_V2),
    vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1),
    vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1),
    swin_t(weights=Swin_T_Weights.IMAGENET1K_V1),
    swin_s(weights=Swin_S_Weights.IMAGENET1K_V1),
]  # load models for inspection


img_transform = Compose(
    [
        Resize((args.imsize, args.imsize)),
        CenterCrop(args.imsize),
        ToTensor(),
        Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    ]
)

total_versions = []

for img_name in tqdm(total_images, desc="Image version computation"):
    in_cls, img_name = img_name.split("/")
    versions = set()
    for img in os.listdir(os.path.join(fg_folder, in_cls)):
        if "_".join(img.split("_")[: len(img_name.split("_"))]) == img_name:
            versions.add(img)
    if len(versions) == 1:
        version = list(versions)[0]
        if version.split(".")[0].split("_")[-1].startswith("v"):
            tqdm.write(f"renaming single version image {version} to {img_name}.WEBP")
            os.rename(os.path.join(fg_folder, in_cls, version), os.path.join(fg_folder, in_cls, f"{img_name}.WEBP"))
            os.rename(
                os.path.join(bg_folder, in_cls, version.replace(".WEBP", ".JPEG")),
                os.path.join(bg_folder, in_cls, f"{img_name}.JPEG"),
            )
        continue
    elif len(versions) == 0:
        tqdm.write(f"Image {img_name} has no versions")
        continue
    versions = sorted(list(versions))
    assert all(
        [version.split(".")[0].split("_")[-1].startswith("v") for version in versions]
    ), f"Weird Versions: {versions} for image {img_name}"
    assert len(versions) <= 3, f"Too many versions for image {img_name}: {versions}"

    version_scores = []
    for v_idx, version in enumerate(versions):
        img = Image.open(os.path.join(fg_folder, in_cls, version))
        bg_img = Image.open(os.path.join(bg_folder, in_cls, f"{version.split('.')[0]}.JPEG"))
        img_mask = np.array(img.convert("RGBA").split()[-1])

        fg_ratio = np.sum(img_mask) / (255 * bg_img.size[0] * bg_img.size[1])

        fg_size = img.size
        monochrome_backgrounds = [
            Image.new(
                "RGB",
                (max(args.imsize, fg_size[0]), max(args.imsize, fg_size[1])),
                (255 * i // (args.batch_size - 2), 255 * i // (args.batch_size - 2), 255 * i // (args.batch_size - 2)),
            )
            for i in range(args.batch_size - 1)
        ]
        pasting_error = False
        for mc_bg in monochrome_backgrounds:
            try:
                mc_bg.paste(img, ((args.imsize - fg_size[0]) // 2, (args.imsize - fg_size[1]) // 2), img)
            except ValueError as e:
                tqdm.write(f"Image {img_name} could not be pasted into background: {e}")
                pasting_error = True
                break

        inp_batch = torch.stack(
            [img_transform(bg_img)] + [img_transform(mc_bg) for mc_bg in monochrome_backgrounds], dim=0
        ).to(device)

        cls_idx = classes.index(in_cls)
        bg_probs = []
        mean_probs = []
        for model in inspection_models:
            model.eval()
            with torch.no_grad():
                out_probs = model(inp_batch).softmax(dim=-1)[:, cls_idx].cpu().numpy()
            bg_probs.append(out_probs[0])
            mean_probs.append(np.mean(out_probs[1:]))

        # average the lists
        bg_probs = np.mean(bg_probs)
        mean_probs = np.mean(mean_probs)

        version_score = (
            score_f(
                idx=v_idx,
                bg_probs=float(bg_probs),
                mean_probs=float(mean_probs),
                fg_ratio=float(fg_ratio),
                max_idx=len(versions) - 1,
            )
            if not pasting_error
            else -100
        )
        version_scores.append(version_score)

    assert len(versions) == len(version_scores), f"Expected {len(versions)} scores, got {len(version_scores)}"

    if max(version_scores) > min(version_scores):
        # find best version
        best_version_idx = int(np.argmax(version_scores))
        best_version = versions[best_version_idx]

        # delete all other versions
        for version in versions:
            if version != best_version:
                os.remove(os.path.join(fg_folder, in_cls, version))
                os.remove(os.path.join(bg_folder, in_cls, f"{version.split('.')[0]}.JPEG"))
        # remove version tag in name
        new_version_name = "_".join(best_version.split("_")[:-1]) + "." + best_version.split(".")[-1]
        os.rename(os.path.join(fg_folder, in_cls, best_version), os.path.join(fg_folder, in_cls, new_version_name))
        os.rename(
            os.path.join(bg_folder, in_cls, f"{best_version.split('.')[0]}.JPEG"),
            os.path.join(bg_folder, in_cls, f"{new_version_name.split('.')[0]}.JPEG"),
        )
    else:
        tqdm.write(f"All versions have the same score for image {img_name}")
