"""
Per-case Dice (all classes) + clDice / Tprec / Tsens (selected classes) for nnU-Net 5-fold validation predictions.

  Tprec = |S_P ∩ V_L| / |S_P|    Tsens = |S_L ∩ V_P| / |S_L|    clDice = 2 Tprec Tsens / (Tprec + Tsens)

Usage:
  python eval_cldice.py <trainer_dir> [--gt <labelsTr>] [--classes 7 8] [-np 8]
  <trainer_dir> e.g. nnUNet_results/Dataset002_CardiacCT_no028/nnUNetTrainer_clDice__nnUNetResEncUNetLPlans__3d_fullres
Writes <trainer_dir>/cldice_eval.csv and prints the per-class mean over all evaluated cases.
"""
import argparse
import csv
import glob
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize


def cl_dice(pred: np.ndarray, gt: np.ndarray):
    if not gt.any() and not pred.any():
        return np.nan, np.nan, np.nan
    s_p = skeletonize(pred).astype(bool)
    s_l = skeletonize(gt).astype(bool)
    tprec = (s_p & gt).sum() / s_p.sum() if s_p.any() else 0.
    tsens = (s_l & pred).sum() / s_l.sum() if s_l.any() else 0.
    cl = 2 * tprec * tsens / (tprec + tsens) if tprec + tsens > 0 else 0.
    return tprec, tsens, cl


def dice(pred: np.ndarray, gt: np.ndarray):
    denom = pred.sum() + gt.sum()
    return 2 * (pred & gt).sum() / denom if denom > 0 else np.nan


def evaluate_case(args):
    pred_file, gt_file, labels, cl_classes = args
    pred = np.asarray(nib.load(pred_file).dataobj).astype(np.uint8)
    gt = np.asarray(nib.load(gt_file).dataobj).astype(np.uint8)
    row = {'case': os.path.basename(pred_file)[:-7], 'fold': pred_file.split(os.sep)[-3]}
    for name, c in labels.items():
        row[f'Dice_{name}'] = dice(pred == c, gt == c)
        if c in cl_classes:
            row[f'Tprec_{name}'], row[f'Tsens_{name}'], row[f'clDice_{name}'] = cl_dice(pred == c, gt == c)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trainer_dir')
    parser.add_argument('--gt', default='/home/kang/nnUNet_raw/Dataset001_CardiacCT/labelsTr')
    parser.add_argument('--classes', type=int, nargs='+', default=[7, 8])
    parser.add_argument('-np', type=int, default=8)
    a = parser.parse_args()

    labels = {k: v for k, v in json.load(open(os.path.join(a.trainer_dir, 'dataset.json')))['labels'].items() if v != 0}
    preds = sorted(glob.glob(os.path.join(a.trainer_dir, 'fold_*', 'validation', '*.nii.gz')))
    assert preds, f'no fold_*/validation/*.nii.gz in {a.trainer_dir}'
    jobs = [(p, os.path.join(a.gt, os.path.basename(p)), labels, a.classes) for p in preds]
    with Pool(a.np) as pool:
        rows = pool.map(evaluate_case, jobs)

    out = os.path.join(a.trainer_dir, 'cldice_eval.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f'{len(rows)} cases from {sorted({r["fold"] for r in rows})} -> {out}')
    for k in rows[0]:
        if k in ('case', 'fold'):
            continue
        v = np.array([r[k] for r in rows], dtype=float)
        print(f'  {k:28s} {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}')
