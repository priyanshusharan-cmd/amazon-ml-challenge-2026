#!/bin/bash
cd "/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026"
LOG=experiments/v4/v3fr.log
PY="/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026/.venv/bin/python"
# wait for the transfer experiment to finish (strictly one heavy job at a time)
until grep -q "TX2 DONE" experiments/v4/tx.log 2>/dev/null; do sleep 30; done
for st in rows enrich predict export; do
  ok=0
  for i in 1 2 3; do
    echo "=== $st attempt $i $(date +%H:%M:%S)" >> $LOG
    if "$PY" -u -X utf8 -W ignore experiments/v4/v3fr.py $st >> $LOG 2>&1; then echo "=== $st OK $(date +%H:%M:%S)" >> $LOG; ok=1; break; fi
    echo "=== $st FAILED $(date +%H:%M:%S)" >> $LOG; sleep 60
  done
  [ $ok = 1 ] || { echo "=== V3FR STOPPED at $st" >> $LOG; exit 1; }
done
echo "=== V3FR DONE $(date +%H:%M:%S)" >> $LOG
