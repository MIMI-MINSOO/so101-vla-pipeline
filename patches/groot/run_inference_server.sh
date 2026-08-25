#!/usr/bin/env bash
# GR00T N1.5 정책 서버 (websocket/zmq). leisaac의 policy_inference.py --policy_type=gr00tn1.5
# 클라이언트가 여기 접속함 (docs/PIPELINE_COMMANDS.md 3-B 참고).
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python scripts/inference_service.py \
    --server \
    --model-path ./cupstack-checkpoints/checkpoint-30000-merged \
    --embodiment-tag new_embodiment \
    --data-config custom_data_configs:So101MarkerH10DataConfig \
    --denoising-steps 4 \
    --port 5555
