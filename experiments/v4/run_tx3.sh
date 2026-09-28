#!/bin/bash
cd "/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026/experiments/v4"
PY="/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026/.venv/bin/python"
until grep -qE "V3FR (DONE|STOPPED)" v3fr.log && [ "$(grep -c 'V3FR' v3fr.log)" -ge 2 ]; do sleep 30; done
for i in 1 2 3; do
  echo "=== in2us_base attempt $i $(date +%H:%M:%S)" >> tx.log
  if "$PY" -u -W ignore tx.py in2us_base India US base 255 200 0.15 >> tx.log 2>&1; then echo "=== in2us_base OK $(date +%H:%M:%S)" >> tx.log; break; fi
  echo "=== in2us_base FAILED $(date +%H:%M:%S)" >> tx.log; sleep 60
done
echo "=== TX3 DONE $(date +%H:%M:%S)" >> tx.log
