#!/bin/bash
# Run ONCE on the Nano5 login node: bash /work/$USER/nnUNet/scripts/setup_env.sh
# Creates the conda env under /work (home is too small) with the same versions as the lab server
# (torch 2.7.1+cu118 supports H100/H200).
set -e
BASE=/work/$USER/nnUNet
export CONDA_PKGS_DIRS=$BASE/.conda_pkgs PIP_CACHE_DIR=$BASE/.pip_cache
module load miniconda3
conda create -p $BASE/envs/nnunet python=3.11 -y
ENV_BIN=$BASE/envs/nnunet/bin
$ENV_BIN/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
$ENV_BIN/pip install -e $BASE/nnUNet   # editable install of our nnU-Net copy, incl. nnUNetTrainer_clDice
$ENV_BIN/pip install scikit-image nibabel
$ENV_BIN/python -c "import nnunetv2, torch; from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainer_clDice import nnUNetTrainer_clDice; print('nnunetv2 OK, torch', torch.__version__)"
