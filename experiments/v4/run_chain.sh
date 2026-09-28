#!/bin/bash
# sequential, resumable chain with bounded retries per stage (each stage resumes from its checkpoints)
cd "/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026"
LOG=experiments/v4/chain.log
PY="\"/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026/.venv/bin/python\" -u -W ignore"
stage() { # name, workdir, command...
  local name=$1 dir=$2; shift 2
  for i in 1 2 3; do
    echo "=== $name attempt $i $(date +%H:%M:%S)" >> $LOG
    if (cd "$dir" && eval "$@") >> $LOG 2>&1; then echo "=== $name OK $(date +%H:%M:%S)" >> $LOG; return 0; fi
    echo "=== $name FAILED attempt $i $(date +%H:%M:%S)" >> $LOG; sleep 60
  done
  echo "=== CHAIN STOPPED at $name" >> $LOG; exit 1
}
stage cands business_entity_resolution/src "$PY -c \"import candidates; candidates.generate('v4test')\""
stage feats business_entity_resolution/src "$PY -c \"import build_features; build_features.build('v4test')\""
stage featx experiments/v2 "$PY featx.py v4test"
stage c3score experiments/v2 "$PY train_v2.py C3_direct_x_more_data v4test"
stage c4score experiments/v2 "$PY stack_v2.py C3_direct_x_more_data v4test"
echo "=== CHAIN DONE $(date +%H:%M:%S)" >> $LOG
