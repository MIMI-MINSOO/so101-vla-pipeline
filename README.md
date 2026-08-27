# SO-101 CupStack VLA Pipeline

SO-101(SO-ARM101) 로봇팔로 **Isaac Sim 시뮬레이션 teleop 데이터 수집 → VLA(Vision-Language-Action) 정책 파인튜닝 → sim/real 추론**까지 이어지는 파이프라인 정리 문서입니다.

이 저장소는 **문서 전용**입니다. IsaacLab/leisaac, openpi, Isaac-GR00T, StarVLA는 각각 독립된 대형 오픈소스 코드베이스라 코드를 이 안으로 옮기지 않았고, 각자 fork에 실제 커밋을 넣어 아래처럼 참조합니다.

| 코드베이스 | 역할 | 본인 fork |
|---|---|---|
| [IsaacLab](https://github.com/MIMI-MINSOO/IsaacLab) + [LeIsaac](https://github.com/MIMI-MINSOO/leisaac-cupstack) | Isaac Sim 태스크, teleop, 데이터 변환, RL 확장 | [REFERENCES.md](REFERENCES.md) |
| [openpi](https://github.com/MIMI-MINSOO/openpi) | π0 (pi0) LoRA fine-tuning | 〃 |
| [Isaac-GR00T](https://github.com/MIMI-MINSOO/Isaac-GR00T) | GR00T N1.5 LoRA fine-tuning | 〃 |
| [StarVLA](https://github.com/MIMI-MINSOO/starVLA) | Qwen3-VL 백본 기반 VLA (아키텍처 비교) | 〃 |

## 파이프라인 개요

**Baseline (검증 완료)** 과 **Research Extension (진행 중/실험적)** 을 분리해서 표기합니다 — 전체가 하나의 완결된 파이프라인처럼 보이지 않도록 하는 게 중요합니다.

```text
Isaac Sim Teleop (SO-101 leader arm)
        │
        ▼
   HDF5 → LeRobot Dataset (mimiminsoo/marker_100, 97 episodes)
        │
        ├──▶ openpi π0 LoRA fine-tuning ──▶ Sim Eval ──▶ Real Robot Deployment   ← 검증된 baseline
        │
        ├──▶ GR00T N1.5 LoRA fine-tuning ──▶ Sim Eval                            ← Architecture Comparison
        │
        └──▶ StarVLA (Qwen3-VL, 백본 동결) fine-tuning ──▶ Sim Eval               ← Architecture Comparison

```

## 문서 목록

- [PIPELINE_COMMANDS.md](docs/PIPELINE_COMMANDS.md) — 처음부터 끝까지 복붙 가능한 명령어
- [ARCHITECTURE_COMPARISON.md](docs/ARCHITECTURE_COMPARISON.md) — π0 vs GR00T N1.5 같은 데이터로 비교
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — 실제로 겪은 문제와 해결
- [REFERENCES.md](REFERENCES.md) — 원본 코드베이스 fork/커밋 링크
- [patches/](patches/) — openpi/GR00T fork가 없던 시절 텍스트로만 보관했던 스크립트 (지금은 각 fork에 실제 커밋으로 들어가 있어 참고용으로만 남김)

## 환경

```text
OS            : Ubuntu 24.04 LTS
GPU           : NVIDIA GeForce RTX 3090 (24GB)
Isaac Sim     : 5.1.0
IsaacLab      : v2.3.2
Python        : 3.11 (uv venv, conda 미사용)
VLA backends  : openpi (JAX, π0 LoRA) / Isaac-GR00T (PyTorch, N1.5 LoRA)
```

## 알려진 한계 (문서화 시점 기준)

- 수집 데이터셋(HDF5)은 실제 60Hz로 기록되지만 LeRobot 변환 시 다운샘플 없이 `fps=30`으로만 라벨링되어 있습니다. `action_horizon=30`인 π0 청크의 실제 물리 시간은 1.0초가 아니라 약 0.5초입니다 (자세한 근거는 TROUBLESHOOTING.md).
- 현재 `--policy_action_horizon`으로 실행 개수를 줄여도(예: 30→10) 재추론 시 이전 청크 정보를 전혀 반영하지 않고 매번 완전히 새로 추론합니다(순수 truncate-and-discard). Physical Intelligence의 Real-Time Chunking 같은 청크 블렌딩 기법은 구현돼 있지 않습니다.
