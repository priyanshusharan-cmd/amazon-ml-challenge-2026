#!/bin/bash
cd "/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026/experiments/v4"
PY="/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026/.venv/bin/python"
true
for i in 1 2 3; do
  echo "=== pseudo attempt $i $(date +%H:%M:%S)" >> tx.log
  if "$PY" -u -W ignore tx_pseudo.py us2in_pseudo tx_tx_us2in_base 0.98 0.02 0.1 >> tx.log 2>&1; then echo "=== pseudo OK $(date +%H:%M:%S)" >> tx.log; break; fi
  echo "=== pseudo FAILED $(date +%H:%M:%S)" >> tx.log; sleep 60
done
echo "=== TX4 DONE $(date +%H:%M:%S)" >> tx.log
