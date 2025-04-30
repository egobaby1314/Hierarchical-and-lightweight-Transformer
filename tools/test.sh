#!/usr/bin/env bash
# -------------------------------------------------
# Evaluate a trained detector on the test set and
# print mAP results.
# Usage: bash tools/test.sh <config.py> <checkpoint.pth>
# -------------------------------------------------
CONFIG=$1
CKPT=$2

if [ -z "$CKPT" ]; then
  echo "Usage: bash tools/test.sh <config.py> <checkpoint.pth>"
  exit 1
fi

# Locate MMDetection’s built-in test.py
MMDET_TEST=$(python - <<'PY'
import mmdet, os
print(os.path.join(os.path.dirname(mmdet.__file__), 'tools', 'test.py'))
PY
)

# Run evaluation (bbox mAP by default)
python "$MMDET_TEST" "$CONFIG" "$CKPT" --eval bbox
