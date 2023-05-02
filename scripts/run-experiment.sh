#!/bin/bash


#SBATCH --job-name=VQA
#SBATCH --output=logger-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=test  # test (1 GPU, 1 hour), quick (1 GPU, 1 day), normal (2 GPU, 2 days), big (4 GPU, 7 days)
#SBATCH --partition=student # rtx2080 (mini-servers), dgxteam (dgx1 for Team-Net), dgxmatinf (dgx2 for WMiI), dgx (dgx1 and dgx2)
#SBATCH --signal=SIGUSR1@90

cd $HOME/vqa || exit
export PYTHONPATH=.:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD


# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

singularity exec \
  --nv \
  -B .:/app \
  -B /shared/sets/datasets:/shared/sets/datasets \
  -B ~/.local/share:/.local/share \
  /shared/sets/singularity/miniconda_pytorch_py310.sif \
  bash scripts/experiment.sh


#done
exit 0
