#!/usr/bin/env bash
# Download a published pi0 checkpoint (and optionally its dataset) from the Hugging Face Hub.
#
#   ./download_pi0.sh                      # default: pi0-lora-marker100-h10 (baseline)
#   ./download_pi0.sh pi0-lora-marker100-h30
#   ./download_pi0.sh pi0-lora-marker100-h10 --with-dataset
#   DEST=/data/ckpts ./download_pi0.sh     # change download destination
#
# Prints the exact serve_policy.py command for the model you downloaded when it finishes.
set -euo pipefail

HF_USER="${HF_USER:-mimiminsoo}"
DEST="${DEST:-./checkpoints}"
MODEL="${1:-pi0-lora-marker100-h10}"
WITH_DATASET=""
[[ "${2:-}" == "--with-dataset" ]] && WITH_DATASET=1

# model repo -> (openpi config name, dataset repo, action_horizon)
case "${MODEL}" in
  pi0-lora-marker100-h10) CONFIG=pi0_lora_marker_100_h10; DATASET=marker_100;      HORIZON=10 ;;
  pi0-lora-marker)        CONFIG=pi0_lora_marker;         DATASET=marker_50;       HORIZON=10 ;;
  pi0-lora-marker100-h30) CONFIG=pi0_lora_marker_100;     DATASET=marker_100;      HORIZON=30 ;;
  pi0_150)                CONFIG=pi0_150;                 DATASET=marker_150;      HORIZON=30 ;;
  pi0_180_real)           CONFIG=pi0_180_real;            DATASET=marker_180_real; HORIZON=30 ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    echo "One of: pi0-lora-marker100-h10 pi0-lora-marker pi0-lora-marker100-h30 pi0_150 pi0_180_real" >&2
    exit 2 ;;
esac

# `hf` replaced `huggingface-cli` in recent huggingface_hub releases; accept either.
if command -v hf >/dev/null 2>&1; then
  HF="hf download"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF="huggingface-cli download"
else
  echo "Neither 'hf' nor 'huggingface-cli' found. Install with: pip install huggingface_hub" >&2
  exit 1
fi

MODEL_DIR="${DEST}/${MODEL}"
echo "[1/2] ${HF_USER}/${MODEL} -> ${MODEL_DIR}  (about 6.2GB, resumes if interrupted)"
mkdir -p "${MODEL_DIR}"
${HF} "${HF_USER}/${MODEL}" --local-dir "${MODEL_DIR}"

if [[ -n "${WITH_DATASET}" ]]; then
  DATASET_DIR="${DEST}/datasets/${DATASET}"
  echo "[2/2] ${HF_USER}/${DATASET} (dataset) -> ${DATASET_DIR}"
  mkdir -p "${DATASET_DIR}"
  ${HF} "${HF_USER}/${DATASET}" --repo-type dataset --local-dir "${DATASET_DIR}"
else
  echo "[2/2] dataset skipped (pass --with-dataset to fetch ${DATASET})"
fi

# The checkpoint ships assets/<user>/<dataset>/norm_stats.json, so serving needs no extra setup.
if [[ ! -f "${MODEL_DIR}/assets/${HF_USER}/${DATASET}/norm_stats.json" ]]; then
  echo "WARNING: norm_stats.json missing under ${MODEL_DIR}/assets/ -- serving will fail." >&2
fi

cat <<EOF

Done. Serve it with (run from your openpi checkout):

  XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \\
  uv run scripts/serve_policy.py --port 8000 \\
      --default-prompt "Pick up the marker and place it into the cup, then reset the arm to rest state." \\
      policy:checkpoint \\
      --policy.config=${CONFIG} \\
      --policy.dir=$(cd "${MODEL_DIR}" && pwd)

Then evaluate in Isaac Sim (from your leisaac checkout, --policy_action_horizon=${HORIZON}):

  python scripts/evaluation/policy_inference.py \\
      --task=LeIsaac-SO101-CupStack-v0 --eval_rounds=20 \\
      --policy_type=openpi --policy_host=localhost --policy_port=8000 \\
      --policy_action_horizon=${HORIZON} \\
      --policy_language_instruction="Pick up the marker and place it into the cup, then reset the arm to rest state." \\
      --device=cuda --enable_cameras
EOF
