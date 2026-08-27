# References

이 문서 저장소가 참조하는 원본 코드베이스입니다. 각 코드베이스는 이제 본인 fork에 실제 커밋이 들어가 있습니다 (예전엔 openpi/GR00T/StarVLA에 fork가 없어서 `patches/`에 텍스트만 보관했지만, 지금은 전부 fork로 옮겨졌습니다 — `patches/`는 참고용으로 남겨둡니다).

## IsaacLab

- 공식 저장소: https://github.com/isaac-sim/IsaacLab
- 본인 fork: https://github.com/MIMI-MINSOO/IsaacLab (branch `so101-cupstack`)
- 사용 버전: `v2.3.2` 위에 수정 없음 (`source/leisaac`를 얹어 쓰는 용도)
- 설치 방식: [PIPELINE_COMMANDS.md](docs/PIPELINE_COMMANDS.md) 참고 — LeIsaac 공식 문서(submodule 설치)와 달리, IsaacLab을 먼저 설치하고 그 `source/`에 LeIsaac을 얹는 방식으로 검증됨

## LeIsaac

- 원본: https://github.com/LightwheelAI/leisaac
- 본인 fork: https://github.com/MIMI-MINSOO/leisaac-cupstack (branch `main`)
- fork base: 공식 `v0.4.0` 태그(`1651c321e9b0c1bb54233211fc7b3cd70d8373d5`)에서 분기
- 이 fork 위에 4개 커밋 존재: baseline VLA pipeline(teleop/SFT) / GRPO+skrl PPO RL extension / mimic·rerender 데이터 증강 실험 / GR00T·StarVLA 비교 평가 스크립트 — 전부 push 완료, 워킹트리 clean

## openpi

- 원본: https://github.com/Physical-Intelligence/openpi
- 사용한 fork: https://github.com/EverNorif/openpi (branch `so101`)
- **본인 fork**: https://github.com/MIMI-MINSOO/openpi (branch `so101`)
- `pi0_lora_marker_100`/`_h10` config + JAX→PyTorch 변환 도구(GRPO용) 커밋 완료

## Isaac-GR00T

- 원본: https://github.com/NVIDIA/Isaac-GR00T
- **본인 fork**: https://github.com/MIMI-MINSOO/Isaac-GR00T (branch `cupstack-marker100`)
- 참조 커밋: `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` (N1.5 계열, mainline — `n1-release` 브랜치 아님을 직접 확인) 위에 커밋
- 커스텀 data config/학습·병합·서빙 스크립트 커밋 완료. 체크포인트(7~8GB)는 gitignore 처리, HF Hub 링크로만 참조

## StarVLA

- 원본: https://github.com/starVLA/starVLA
- **본인 fork**: https://github.com/MIMI-MINSOO/starVLA (branch `so101-marker100`)
- Qwen3-VL 백본 완전동결 + action head(qwenpiv3) 스크래치 학습 설정 커밋 완료 — 다른 3개와 달리 독립적인 VLA 아키텍처 비교 트랙

## 데이터셋 / 체크포인트 (Hugging Face)

| 이름 | repo_id | 내용 |
|---|---|---|
| 수집 데이터셋 | `mimiminsoo/marker_100` | LeRobot Dataset, 97 episodes / 31,291 frames |
| π0 LoRA 체크포인트 | `mimiminsoo/pi0-lora-marker100-h30` | `pi0_lora_marker_100`, action_horizon=30 |
| GR00T N1.5 LoRA 체크포인트 (action head만) | `mimiminsoo/groot_h10` | action_horizon=10 |
| GR00T N1.5 LoRA 체크포인트 (백본 포함) | `mimiminsoo/groot_h10_fulllora` | action_horizon=10, `--lora-full-model` |
