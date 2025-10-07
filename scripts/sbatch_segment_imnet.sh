#!/bin/bash

#SBATCH --array=0-299%40
#SBATCH --time=1-0
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=H200,H100,H100-PCI,A100-PCI,A100-40GB,A100-80GB,H200-PCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --export="TQDM_DISABLE=1"
#SBATCH --job-name="Segment ImageNet"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N-%a.out

srun -K \
  --container-image=CONTAINER_IMAGE \
  --container-workdir="$(pwd)" \
  --container-mounts=YOUR_MOUNTS,"$(pwd)":"$(pwd)" \
  python3 segment_imagenet.py -p 300 -id $SLURM_ARRAY_TASK_ID "$@"
