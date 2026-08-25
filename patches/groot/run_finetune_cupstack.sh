#!/usr/bin/env bash
# GR00T N1.5 LoRA fine-tuning -- action head only (VLM backbone frozen).
# $GROOT_ROOT 밑에 custom_data_configs.py와 함께 놓고 실행하세요.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python scripts/gr00t_finetune.py \
    --dataset-path ~/.cache/huggingface/lerobot/<your-hf-username>/marker_100 \
    --output-dir ./cupstack-checkpoints \
    --data-config custom_data_configs:So101MarkerH10DataConfig \
    --video-backend torchvision_av \
    --batch-size 32 \
    --tune-diffusion-model \
    --lora-rank 32 \
    --lora-alpha 32 \
    --max-steps 30000 \
    --save-steps 5000 \
    --report-to wandb \
    --num-gpus 1
