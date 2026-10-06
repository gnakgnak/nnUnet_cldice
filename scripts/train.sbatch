#!/usr/bin/env bash
# Paired loss comparison on Dataset002, ResEnc L, one job with 4 GPUs:
#   GPU 0: nnUNetTrainer fold 0   GPU 1: nnUNetTrainer_clDice fold 0
#   GPU 2: nnUNetTrainer fold 1   GPU 3: nnUNetTrainer_clDice fold 1
# The first job also preprocesses the data (only if not done yet).
# gp1d is limited to 1 day, so submit this script several times: --dependency=singleton makes each copy wait
# for the previous one, and every run resumes from its checkpoint_latest.pth.
#   cd /work/$USER/nnUNet && for i in 1 2 3; do sbatch scripts/train.sbatch; done

#SBATCH --account=MST114546
#SBATCH --partition=normal
#SBATCH --job-name=d2-cldice-h100
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --mem=360G
#SBATCH --mail-user=kangenting93@gapp.nthu.edu.tw
#SBATCH --mail-type=END,FAIL

BASE=/work/$USER/nnUNet
export PATH=$BASE/envs/nnunet/bin:$PATH
export nnUNet_raw=$BASE/nnUNet_raw
export nnUNet_preprocessed=$BASE/nnUNet_preprocessed
export nnUNet_results=$BASE/nnUNet_results
export nnUNet_compile=f        # keep identical to the lab-server runs
export nnUNet_n_proc_DA=11     # data augmentation workers per run: 4 runs x 11 < 48 CPUs
PLANS=nnUNetResEncUNetLPlans
cd $BASE/logs

# Preprocess with the ResEnc L plans copied from the lab server (no re-planning, so plans stay identical)
PREP=$nnUNet_preprocessed/Dataset002_CardiacCT_no028/nnUNetPlans_3d_fullres
if [ "$(ls $PREP 2>/dev/null | wc -l)" -lt 609 ]; then
  nnUNetv2_preprocess -d 2 -plans_name $PLANS -c 3d_fullres -np 32 || exit 1
  [ "$(ls $PREP | wc -l)" -eq 609 ] || { echo "preprocessing incomplete: $(ls $PREP | wc -l) of 609 files"; exit 1; }
fi

train() {  # $1 = gpu, $2 = trainer, $3 = fold
  local out=$nnUNet_results/Dataset002_CardiacCT_no028/${2}__${PLANS}__3d_fullres/fold_$3
  [ -f $out/checkpoint_final.pth ] && return   # this run is already finished
  local c=""; [ -f $out/checkpoint_latest.pth ] && c="--c"
  CUDA_VISIBLE_DEVICES=$1 nnUNetv2_train 2 3d_fullres $3 -p $PLANS -tr $2 $c >> train_${2}_f$3.log 2>&1
}
train 0 nnUNetTrainer 0 &
train 1 nnUNetTrainer_clDice 0 &
train 2 nnUNetTrainer 1 &
train 3 nnUNetTrainer_clDice 1 &
wait
