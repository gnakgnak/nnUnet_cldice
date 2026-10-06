#!/bin/bash
R=/work/$USER/nnUNet/nnUNet_results/Dataset002_CardiacCT_no028
for d in $R/*/fold_*; do
  log=$(ls -t $d/training_log_*.txt 2>/dev/null | head -1); [ -z "$log" ] && continue
  ep=$(grep -oP ': Epoch \K[0-9]+' $log | tail -1)
  t=$(grep -oP 'Epoch time: \K[0-9.]+' $log | tail -5 | awk '{s+=$1} END {if (NR) printf "%.0f", s/NR}')
  dice=$(grep -oP 'Pseudo dice \K.*' $log | tail -1 | grep -oP '[0-9]\.[0-9]+' | awk '{printf "%.2f ", $1}')
  left=$([ -n "$t" ] && awk -v e=$ep -v t=$t 'BEGIN {printf "%.1f h", (1000-e)*t/3600}')
  [ -f $d/checkpoint_final.pth ] && left=done
  printf "%-22s %s  epoch %4s/1000  %4ss/epoch  left %-8s dice: %s\n" "$(basename $(dirname $d) | sed 's/__.*//')" "$(basename $d)" "$ep" "$t" "$left" "$dice"
done
