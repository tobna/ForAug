[![arXiv](https://img.shields.io/badge/arXiv-2503.09399-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.09399)
[![Static Badge](https://img.shields.io/badge/Huggingface-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/TNauen/ForNet)

# ForAug
![ForAug](images/foraug.png)

This is the public code repository for the paper [_ForAug: Recombining Foregrounds and Backgrounds to Improve Vision Transformer Training with Bias Mitigation_](https://www.arxiv.org/abs/2503.09399).

### Updates

- [19.03.2025] We release the code to download and use ForNet in this repo :computer:
- [19.03.2025] We release the patch files of [ForNet on Huggingface](https://huggingface.co/datasets/TNauen/ForNet) :hugs:
- [12.03.2025] We release the preprint of [ForAug on arXiv](https://www.arxiv.org/abs/2503.09399) :spiral_notepad:

## Using ForAug/ForNet

### Preliminaries
To be able to download ForNet, you will need the ImageNet dataset in the usual format at `<in_path>`:
```
<in_path>
|--- train
|    |--- n01440764
|    |    |--- n01440764_10026.JPEG
|    |    |--- n01440764_10027.JPEG
|    |    |--- n01440764_10029.JPEG
|    |    `-  ...
|    |--- n01693334
|    `-  ...
`-- val
     |--- n01440764
     |    |--- ILSVRC2012_val_00000293.JPEG
     |    |--- ILSVRC2012_val_00002138.JPEG
     |    |--- ILSVRC2012_val_00003014.JPEG
     |    `-  ...
     |--- n01693334
     `-  ...
```

### Downloading ForNet
To download and prepare the already-segmented ForNet dataset at `<data_path>`, follow these steps:
#### 1. Clone this repository and install the requirements
```
git clone https://github.com/tobna/ForAug
cd ForAug
pip install -r prep-requirements.txt
```
#### 2. Download the diff files
```
./download_diff_files.sh <data_path>
```
This script will download all dataset files to `<data_path>`

#### 3. Apply the diffs to ImageNet
```
python apply_patch.py -p <data_path> -in <in_path> -o <data_path>
```
This will apply the diffs to ImageNet and store the results in the `<data_path>` folder. It will also delete the already-processes patch files (the ones downloaded in step 2). In order to keep the patch files, add the `--keep` flag.

#### Optional: Zip the files without compression
When dealing with a large cluster and dataset files that have to be sent over the network (i.e. the dataset is on another server than the one used for processing) it's sometimes useful to not deal with many small files and have fewer large ones instead.
If you want this, you can zip up the files (without compression) by using
```
./zip_up.sh <data_path>
```
### Creating ForNet from Scratch
Coming soon

### Using ForNet
To use ForAug/ForNet you need to have it available in folder or zip form (see [Downloading ForNet](#downloading-fornet)) at `data_path`.
Additionally, you need to install the (standard) requirements from 'requirements.txt':
```
pip install -r requirements.txt
```

Then, just do
```python
from fornet import ForNet

data_path = ...

dataset = ForNet(
            data_path, 
            train=True, 
            transform=None, 
            background_combination="all",
          )

```

For information on all possible parameters, run
```python
from fornet import ForNet

help(ForNet.__init__)
```

## Citation 
Coming Soon

## ToDos

- [x] release code to download and create ForNet
- [x] release code to use ForNet for training and evaluation
- [ ] integrate ForNet into Huggingface Datasets
