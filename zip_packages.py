import argparse
import os
import zipfile
from multiprocessing import Manager, Process
from time import sleep

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Zip packages")
parser.add_argument("-f", "--folder", type=str, required=True, help="ForAug folder to work in.")

args = parser.parse_args()


def _file_gather(folder, part, images, classes, q, ret_dict):
    files = []
    ending = "WEBP" if images == "foregrounds" else "JPEG"
    for idx, c in enumerate(classes):
        files.append([f"{c}/{f}" for f in os.listdir(os.path.join(folder, part, images, c)) if f.endswith(ending)])
        q.put(([f"{part}/{images}"], idx + 1))

    ret_dict[f"{part}/{images}"] = [f for sublist in files for f in sublist]


def _zip_files(folder, part, images, filelist, update_dict):
    update_dict[f"{part}/{images}"] = 0
    with zipfile.ZipFile(os.path.join(args.folder, f"{args.images}_{args.part}.zip"), "w") as zf:
        for idx, file in enumerate(files):
            zf.write(os.path.join(args.folder, args.part, args.images, file), file)
            update_dict[f"{args.part}/{args.images}"] = idx + 1


# maybe all in one file and multiple progress bars: https://stackoverflow.com/questions/77359940/multiple-progress-bars-with-python-multiprocessing

# gather files
classes = [
    f
    for f in os.listdir(os.path.join(args.folder, "train", "foregrounds"))
    if os.path.isdir(os.path.join(args.folder, "train", "foregrounds", f))
]
assert len(classes) == 1000, f"Expected 1000 classes, got {len(classes)}"

print("Gathering files:")

processes = {}
manager = Manager()
files = manager.dict()
update_q = manager.Queue()
for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        p = Process(target=_file_gather, args=(args.folder, part, images, classes, update_q, files))
        p.start()
        processes[f"{part}/{images}"] = p

pbars = {}
pos = 0
last_update = {}
for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        pbars[f"{part}/{images}"] = tqdm(total=len(classes), desc=f"{part}/{images}", position=pos, disable=True)
        last_update[f"{part}/{images}"] = 0
        pos += 1

while len(processes) > 0:
    while not update_q.empty():
        folder, idx = update_q.get()
        print(folder, idx, last_update[folder])
        pbars[folder].update(idx - last_update[folder])
        last_update[folder] = idx
        if idx == len(classes):
            print(f"Closing {folder}")
            pbars[folder].close()
            del processes[folder]
    sleep(0.01)

for pbar in pbars.values():
    pbar.close()

for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        _EXPECTED_FILES = (1274557 if images == "foreground" else 1274556) if part == "train" else 49751
        assert (
            len(files[f"{part}/{images}"]) == _EXPECTED_FILES
        ), f"{part}/{images}: Expected {_EXPECTED_FILES} files, got {len(files[f'{part}/{images}'])}"

exit()

for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        p = Process(target=_zip_files, args=(args.folder, part, images, files[f"{part}/{images}"], update_q))
        p.start()
        processes[f"{part}/{images}"] = p
