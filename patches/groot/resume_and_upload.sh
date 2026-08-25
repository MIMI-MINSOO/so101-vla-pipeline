#!/usr/bin/env bash
# 학습이 세션 끊김 등으로 중단됐을 때 이어서 실행 + 완료 시 HF 업로드.
# setsid+nohup+disown으로 세션과 분리해서 백그라운드 실행하는 걸 권장:
#   setsid nohup bash resume_and_upload.sh > resume_and_upload.log 2>&1 < /dev/null &
#   disown
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
    --num-gpus 1 \
    --resume

if [ -d ./cupstack-checkpoints/checkpoint-30000 ]; then
    huggingface-cli upload <your-hf-username>/groot_h10 ./cupstack-checkpoints/checkpoint-30000
fi
