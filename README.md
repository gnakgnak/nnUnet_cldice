# nnUnet_cldice

nnU-Net v2 with a soft clDice loss term (Shit et al., CVPR 2021) for thin tubular structures in cardiac CT,
compared against the default nnU-Net trainer.

## Contents

| Path | What |
|---|---|
| `nnunetv2/` | nnU-Net v2 with the clDice additions (`pip install -e .`); upstream readme in `documentation/nnUNet_readme.md` |
| `nnunetv2/training/loss/cldice.py` | soft skeleton, `SoftclDiceLoss`, `DC_CE_and_clDice_loss` |
| `nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainer_clDice.py` | trainer: Dice + CE + 0.5 · (1 − soft clDice) on classes 7, 8 |
| `scripts/setup_env.sh` | creates the conda env (torch 2.7.1 + cu118) |
| `scripts/train.sbatch` | Slurm job: both trainers × folds 0/1 on 4 GPUs, resumes from checkpoints |
| `scripts/status.sh` | training progress per run |
| `scripts/eval_cldice.py` | per-case Dice + clDice / Tprec / Tsens on validation predictions |
| `nnUNet_results/` | plans, training logs, `progress.png` and validation `summary.json` per fold |
| `logs/` | Slurm and training stdout/stderr |

Not included (too large for GitHub): raw / preprocessed data, model checkpoints (`*.pth`, ~780 MB each) and
validation predictions (`*.nii.gz`).

## Setup

- Dataset: `Dataset002_CardiacCT_no028`
- Plans: `nnUNetResEncUNetLPlans`, `3d_fullres`, 1000 epochs
- Labels: 1 RightAtrium, 2 RightVentricle, 3 LeftAtrium, 4 LeftVentricle, 5 MyocardiumLV, 6 Aorta, 7 Coronaries8, 8 Fat

```bash
bash scripts/setup_env.sh
for i in 1 2 3; do sbatch scripts/train.sbatch; done
python scripts/eval_cldice.py nnUNet_results/Dataset002_CardiacCT_no028/nnUNetTrainer_clDice__nnUNetResEncUNetLPlans__3d_fullres --gt <labelsTr>
```

## Results (validation Dice, nnU-Net `summary.json`)

| Trainer | Fold | FG mean | RA | RV | LA | LV | MyoLV | Aorta | Coronaries | Fat |
|---|---|---|---|---|---|---|---|---|---|---|
| nnUNetTrainer | 0 | **0.9058** | 0.9500 | 0.9487 | 0.9640 | 0.9448 | 0.9476 | 0.9680 | **0.7510** | 0.7722 |
| nnUNetTrainer_clDice | 0 | 0.9051 | 0.9493 | 0.9485 | 0.9648 | 0.9455 | 0.9478 | 0.9676 | 0.7441 | 0.7731 |
| nnUNetTrainer | 1 | **0.8947** | 0.9552 | 0.9282 | 0.9470 | 0.9356 | 0.9440 | 0.9503 | **0.7338** | 0.7637 |
| nnUNetTrainer_clDice | 1 | 0.8939 | 0.9553 | 0.9283 | 0.9470 | 0.9353 | 0.9436 | 0.9502 | 0.7281 | 0.7631 |

On folds 0 and 1, clDice makes no meaningful difference to volumetric Dice: the change is within ±0.001 for
most classes, and Coronaries drops by 0.006–0.007. Topology metrics (clDice / Tprec / Tsens) come from
`scripts/eval_cldice.py`.
