# patches/openpi_config_pi0_lora_marker_100.py
#
# openpi(https://github.com/EverNorif/openpi @ 285c024)에 본인 fork가 없어
# `src/openpi/training/config.py`의 `_CONFIGS` 리스트에 추가한 TrainConfig 블록을
# 원문 그대로 보관합니다. 재현 시 이 블록을 config.py의 _CONFIGS 리스트 안에 넣으세요.
#
# 필요한 import (config.py 상단에 이미 있음):
#   import openpi.policies.so101_policy as so101_policy
#   from openpi.models import pi0_config
#   from openpi.training import weight_loaders

TrainConfig(
    # marker_100 (sim, 97 episodes / 31,291 frames, LeRobot v2.1, front+wrist)
    # action_horizon=30: autocorrelation 분석상 lag ~33에서 0.5 아래로 떨어짐 (1.0s @30fps
    # 로 계산됐으나, 실제 수집 주파수는 60Hz -- TROUBLESHOOTING.md #9 참고. 실제 물리
    # 시간은 약 0.5s).
    name="pi0_lora_marker_100",
    model=pi0_config.Pi0Config(
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotSO101DataConfig(
        repo_id="<your-hf-username>/marker_100",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=8,  # RTX 3090 24GB 기준
    num_workers=4,  # av1 비디오 디코딩이 병목이라 기본값 2보다 늘림
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
    freeze_filter=pi0_config.Pi0Config(
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
),
TrainConfig(
    # Same data as pi0_lora_marker_100, action_horizon 10 instead of 30, to compare how
    # chunk length affects real-robot success. Shorter chunks re-infer more often
    # (closer to closed-loop) at the cost of the arm idling during each server call.
    # Kept as its own config so both runs' checkpoints and norm stats coexist.
    name="pi0_lora_marker_100_h10",
    model=pi0_config.Pi0Config(
        action_horizon=10,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotSO101DataConfig(
        repo_id="<your-hf-username>/marker_100",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=8,
    num_workers=4,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
    freeze_filter=pi0_config.Pi0Config(
        action_horizon=10,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
),
