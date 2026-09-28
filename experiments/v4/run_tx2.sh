#!/bin/bash
cd "/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026/experiments/v4"
LOG=tx.log
PY="/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026/.venv/bin/python"
until grep -qE "V3FR (DONE|STOPPED)" v3fr.log 2>/dev/null; do sleep 30; done
for cfg in "us2in_qnorm US India qnorm 255 200 0.15" "us2in_noscale US India noscale 255 200 0.15" "us2in_qnorm_noscale US India qnorm_noscale 255 200 0.15"; do
  set -- $cfg
  for i in 1 2 3; do
    echo "=== $1 attempt $i $(date +%H:%M:%S)" >> $LOG
    if "$PY" -u -W ignore tx.py $1 $2 $3 $4 $5 $6 $7 >> $LOG 2>&1; then echo "=== $1 OK $(date +%H:%M:%S)" >> $LOG; break; fi
    echo "=== $1 FAILED $(date +%H:%M:%S)" >> $LOG; sleep 60
  done
done
echo "=== TX2 DONE $(date +%H:%M:%S)" >> $LOG
