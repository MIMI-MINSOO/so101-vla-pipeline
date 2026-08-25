# References

이 문서 저장소가 참조하는 원본 코드베이스입니다. 코드 자체는 이 저장소에 없고, 아래 커밋을 기준으로 문서를 작성했습니다.

## IsaacLab

- 공식 저장소: https://github.com/isaac-sim/IsaacLab
- 사용 버전: `v2.3.2` (`37ddf6268`)
- 설치 방식: [PIPELINE_COMMANDS.md](docs/PIPELINE_COMMANDS.md) 참고 — LeIsaac 공식 문서(submodule 설치)와 달리, IsaacLab을 먼저 설치하고 그 `source/`에 LeIsaac을 얹는 방식으로 검증됨

## LeIsaac (본인 fork)

- 원본: https://github.com/LightwheelAI/leisaac
- 본인 fork: https://github.com/MIMI-MINSOO/leisaac-cupstack
- 참조 커밋: `7e72f12` ("Add marker-in-cup task (CupStack) with converted marker asset")
- fork base: 공식 `v0.4.0` 태그(`1651c321e9b0c1bb54233211fc7b3cd70d8373d5`)에서 분기
- ⚠️ 참고: 이 fork의 워킹트리에는 아직 커밋되지 않은 변경사항(CupStack 태스크 확장, RL reward, 평가 스크립트 등)이 있습니다. 재현 시 이 문서의 [PIPELINE_COMMANDS.md](docs/PIPELINE_COMMANDS.md)에 있는 명령을 그대로 따라가면 되지만, 커밋 해시만으로 100% 재현되지는 않는 상태입니다.

## openpi

- 원본: https://github.com/Physical-Intelligence/openpi
- 사용한 fork: https://github.com/EverNorif/openpi (branch `so101`)
- 참조 커밋: `285c024` ("some fix.")
- 본인 fork는 없음 — `pi0_lora_marker_100` 등 이 프로젝트 전용 학습 config는 [patches/openpi_config_pi0_lora_marker_100.py](patches/openpi_config_pi0_lora_marker_100.py)에 원문 보관

## Isaac-GR00T

- 원본: https://github.com/NVIDIA/Isaac-GR00T
- 참조 커밋: `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` (N1.5 계열, mainline — `n1-release` 브랜치 아님을 직접 확인)
- 본인 fork는 없음 — 커스텀 data config/학습·병합 스크립트는 [patches/groot/](patches/groot/)에 원문 보관

## 데이터셋 / 체크포인트 (Hugging Face)

| 이름 | repo_id | 내용 |
|---|---|---|
| 수집 데이터셋 | `mimiminsoo/marker_100` | LeRobot Dataset, 97 episodes / 31,291 frames |
| π0 LoRA 체크포인트 | `mimiminsoo/pi0-lora-marker100-h30` | `pi0_lora_marker_100`, action_horizon=30 |
| GR00T N1.5 LoRA 체크포인트 (action head만) | `mimiminsoo/groot_h10` | action_horizon=10 |
| GR00T N1.5 LoRA 체크포인트 (백본 포함) | `mimiminsoo/groot_h10_fulllora` | action_horizon=10, `--lora-full-model` |
