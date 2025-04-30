#!/usr/bin/env bash
# ----------------------------------------------------------
# Launch MMDetection training with a single command.
# Usage:  bash tools/train.sh <config.py> [<num_gpus>]
#         If <num_gpus> is omitted, defaults to 1.
#         The script also runs fine on CPU for demo purposes.
# ----------------------------------------------------------
CONFIG=$1
GPUS=${2:-1}                          # Default: 1 GPU (or CPU)
WORK_DIR=${WORK_DIR:-"work_dirs/$(basename ${CONFIG%.*})"}

if [ -z "$CONFIG" ]; then
  echo "Usage: bash tools/train.sh <config.py> [<num_gpus>]"
  exit 1
fi

# Locate MMDetection’s built-in train.py
MMDET_TRAIN=$(python - <<'PY'
import mmdet, os
print(os.path.join(os.path.dirname(mmdet.__file__), 'tools', 'train.py'))
PY
)

# Start training; --seed and --deterministic keep the demo reproducible
python "$MMDET_TRAIN" \
       "$CONFIG" \
       --work-dir "$WORK_DIR" \
       --gpus "$GPUS" \
       --seed 0 --deterministic

