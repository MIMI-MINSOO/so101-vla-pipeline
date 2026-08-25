#!/usr/bin/env bash
# GR00T N1.5 LoRA fine-tuning -- VLM backbone included (--lora-full-model).
# 위 run_finetune_cupstack.sh(action head만)와 비교하기 위한 아키텍처 비교 실험.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

python scripts/gr00t_finetune.py \
    --dataset-path ~/.cache/huggingface/lerobot/<your-hf-username>/marker_100 \
    --output-dir ./cupstack-checkpoints-fulllora \
    --data-config custom_data_configs:So101MarkerH10DataConfig \
    --video-backend torchvision_av \
    --batch-size 8 \
    --tune-diffusion-model \
    --lora-rank 32 \
    --lora-alpha 32 \
    --lora-full-model \
    --max-steps 30000 \
    --save-steps 5000 \
    --report-to wandb \
    --num-gpus 1 \
    --resume

if [ -d ./cupstack-checkpoints-fulllora/checkpoint-30000 ]; then
    echo "=== training finished, merging LoRA into base weights ==="
    python merge_lora_fulllora.py

    # fix PEFT's auto-generated README.md (base_model must be a valid HF id, not a local cache path)
    README=./cupstack-checkpoints-fulllora/checkpoint-30000/README.md
    if [ -f "$README" ]; then
        sed -i 's|base_model: .*|base_model: nvidia/GR00T-N1.5-3B|' "$README"
        sed -i 's|- base_model:adapter:.*|- base_model:adapter:nvidia/GR00T-N1.5-3B|' "$README"
    fi

    echo "=== uploading LoRA adapter to <your-hf-username>/groot_h10_fulllora ==="
    huggingface-cli upload <your-hf-username>/groot_h10_fulllora ./cupstack-checkpoints-fulllora/checkpoint-30000
    echo "=== upload done ==="
else
    echo "=== WARNING: training exited but checkpoint-30000 not found ==="
    find ./cupstack-checkpoints-fulllora -maxdepth 1 -type d
fi
