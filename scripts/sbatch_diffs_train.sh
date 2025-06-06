#!/bin/bash

#SBATCH --array=0-399
#SBATCH --time=1-0
#SBATCH --mem=16G
#SBATCH --partition=A100-PCI,RTX3090,RTXA6000,V100-16GB,H100-PCI,A100-40GB,A100-80GB,H100,H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --export="HF_HOME=/netscratch/nauen/HF_HOME/,NLTK_DATA=/netscratch/nauen/NLTK_DATA/,TQDM_DISABLE=1"
#SBATCH --job-name="Make ImageNet diffs (train)"
#SBATCH --output=/netscratch/nauen/slurm/%x-%j-%N-%a.out

srun -K \
  --container-image=/netscratch/nauen/images/create_in_diff.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/fscratch/$USER:/fscratch/$USER,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"`pwd`":"`pwd`" \
  python3 make_diffs.py -n 400 -id $SLURM_ARRAY_TASK_ID -w 16 -in /ds-sds/images/imagenet -fn /fscratch/nauen/datasets/INSegment_v1_f1 -s train -o /netscratch/nauen/datasets/INDiffs