# SO-101 CupStack VLA Pipeline

SO-101(SO-ARM101) 로봇팔로 **Isaac Sim에서 teleop 데이터를 수집 → VLA(Vision-Language-Action) 정책 파인튜닝 → 시뮬레이션/실기에서 추론**까지 이어지는 파이프라인입니다.

태스크는 CupStack (marker-in-cup): 책상 위 마커를 집어 컵에 꽂는 동작입니다.

## 전체 흐름

```text
①  환경 구축
        │
②  Teleop 데이터 수집          SO-101 리더암으로 시뮬레이션 조작
        │                      → datasets/*.hdf5
        ▼
③  LeRobot 포맷 변환
        │                      → ~/.cache/huggingface/lerobot/<id>/marker_100
        │
        ├──────────────┬──────────────────────┐
        ▼              ▼                      ▼
④ openpi π0      ④ Isaac-GR00T N1.5      ④ StarVLA (Qwen3-VL)
   LoRA 학습          LoRA 학습                action head 학습
        │              │                      │
        ▼              ▼                      ▼
⑤  정책 서버 기동 (websocket 또는 ZMQ)
        │
        ▼
⑥  Isaac Sim 평가
        │
        ▼
⑦  실기 배포 (SO-101 실물)
```

## 문서

| 문서 | 내용 |
|---|---|
| **[PIPELINE_COMMANDS.md](docs/PIPELINE_COMMANDS.md)** | ①~⑦ 전 과정 — 환경 구축, 데이터 수집/변환, **π0 · GR00T** 학습·서빙·평가, 실기 배포 |
| **[STARVLA_PIPELINE.md](docs/STARVLA_PIPELINE.md)** | **StarVLA** 경로 — 환경·데이터로더 구조가 달라 분리. ③번 데이터셋을 그대로 받아서 시작 |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 실행이 막히는 오류와 해결 |
| [REFERENCES.md](REFERENCES.md) | 원본 코드베이스 fork/커밋 링크, HF 데이터셋·체크포인트 |

처음 따라 하신다면 **PIPELINE_COMMANDS.md를 위에서부터 순서대로** 읽으시면 됩니다. StarVLA는 그 뒤에 별도로 보세요.

## 코드베이스

문서만 이 저장소에 있고, 코드는 각 fork에 있습니다.

| fork | 역할 |
|---|---|
| [IsaacLab](https://github.com/MIMI-MINSOO/IsaacLab) | 시뮬레이터 (v2.3.2, 수정 없음) |
| [leisaac-cupstack](https://github.com/MIMI-MINSOO/leisaac-cupstack) | CupStack 태스크, teleop, 데이터 변환, 평가 클라이언트 |
| [openpi](https://github.com/MIMI-MINSOO/openpi) | π0 LoRA 학습·서빙 |
| [Isaac-GR00T](https://github.com/MIMI-MINSOO/Isaac-GR00T) | GR00T N1.5 LoRA 학습·서빙 |
| [starVLA](https://github.com/MIMI-MINSOO/starVLA) | StarVLA (Qwen3-VL 백본) 학습·서빙 |

세 학습 프레임워크는 **각각 독립된 venv**를 씁니다 (conda 미사용).

## 검증 환경

```text
OS       : Ubuntu 24.04 LTS
GPU      : NVIDIA GeForce RTX 3090 (24GB) × 1
Isaac Sim: 5.1.0
IsaacLab : v2.3.2
Python   : 3.11 (IsaacLab/openpi) / 3.10 (GR00T/StarVLA)
```

VRAM 24GB 기준으로 batch size와 메모리 설정이 잡혀 있습니다. 더 큰 GPU를 쓰신다면 각 문서의 해당 주석을 참고해 올리세요.
