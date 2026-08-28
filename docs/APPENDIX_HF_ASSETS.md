# Appendix — 공개 데이터셋 / π0 체크포인트

학습을 직접 돌리지 않고 **이미 학습된 π0 체크포인트를 받아서 바로 추론만 돌려보고 싶을 때** 쓰는 문서입니다. 전부 Hugging Face Hub에 공개돼 있습니다.

전체 파이프라인(데이터 수집부터)은 [PIPELINE_COMMANDS.md](PIPELINE_COMMANDS.md)를 보세요.

> 이 문서는 **π0(openpi)만** 다룹니다. GR00T / StarVLA 체크포인트도 공개돼 있지만 아직 여기 정리되지 않았습니다.

---

## 가장 빠른 경로

```bash
git clone https://github.com/MIMI-MINSOO/so101-vla-pipeline.git
cd so101-vla-pipeline
bash scripts/download_pi0.sh          # baseline 모델(h10) 약 6.2GB 다운로드
```

받고 나면 스크립트가 **그 모델에 맞는 `serve_policy.py` / `policy_inference.py` 명령어를 그대로 출력**해줍니다. 복사해서 실행하면 됩니다.

다른 모델을 받거나 데이터셋까지 같이 받으려면:

```bash
bash scripts/download_pi0.sh pi0-lora-marker100-h30
bash scripts/download_pi0.sh pi0-lora-marker100-h10 --with-dataset
DEST=/data/ckpts bash scripts/download_pi0.sh        # 저장 위치 변경
```

추론을 돌리려면 openpi(서버)와 leisaac+Isaac Sim(평가)이 설치돼 있어야 합니다 → [PIPELINE_COMMANDS.md `[0] 환경 구축`](PIPELINE_COMMANDS.md#0-환경-구축)

---

## π0 체크포인트

전부 JAX/orbax 포맷이고 **약 6.16GB**입니다. `assets/<user>/<dataset>/norm_stats.json`이 같이 들어있어서 추가 준비 없이 바로 서빙됩니다.

**baseline은 `pi0-lora-marker100-h10`입니다.** 나머지는 학습 데이터 규모 순입니다.

| HF repo | openpi config | 학습 데이터 | episodes | action_horizon |
|---|---|---|---|---|
| **`mimiminsoo/pi0-lora-marker100-h10`** | `pi0_lora_marker_100_h10` | `marker_100` (sim) | 97 | **10** |
| `mimiminsoo/pi0-lora-marker` | `pi0_lora_marker` | `marker_50` (sim) | 49 | 10 |
| `mimiminsoo/pi0-lora-marker100-h30` | `pi0_lora_marker_100` | `marker_100` (sim) | 97 | 30 |
| `mimiminsoo/pi0_150` | `pi0_150` | `marker_150` (sim) | 146 | 30 |
| `mimiminsoo/pi0_180_real` | `pi0_180_real` | `marker_180_real` (sim+실기) | 176 | 30 |

전부 `--policy.config`에 위 config 이름을 그대로 넘기면 됩니다. config 이름이 다르면 norm_stats를 못 찾아 서빙이 실패합니다.

`pi0_180_real`은 이름과 달리 순수 실기 데이터가 아니라 **sim + 실기 혼합**입니다 — `marker_150`(sim 146ep) + `marker_real`(실기 30ep)이며, episode 수와 frame 수가 두 데이터셋의 합과 정확히 일치합니다(146+30=176, 44,215+18,433=62,648). 이름의 "150"/"180"은 반올림 표현입니다.

### ⚠️ 주의사항

**`train_state/`가 빠져 있습니다.** 업로드 시 옵티마이저 상태를 제외했기 때문에 **추론은 되지만 학습 재개(`--resume`)는 안 됩니다.** 이어서 학습하려면 직접 처음부터 돌려야 합니다.

**`pi0_base_lora_marker`는 다른 경로입니다.** 이름이 비슷하지만 openpi(JAX)가 아니라 LeRobot `lerobot-train`으로 만든 22MB짜리 PEFT adapter라, 이 문서의 `serve_policy.py` 경로와 호환되지 않습니다.

---

## 데이터셋

전부 LeRobot Dataset 포맷입니다. **학습을 직접 돌릴 때만 필요합니다** — 공개된 체크포인트로 추론만 할 거면 받을 필요 없습니다.

| HF repo | episodes | frames | 포맷 | 종류 | 비고 |
|---|---|---|---|---|---|
| **`mimiminsoo/marker_100`** | 97 | 31,291 | v2.1 | sim | **주력 데이터셋.** 위 h30/h10 모델이 이걸로 학습됨 |
| `mimiminsoo/marker_50` | 49 | 12,924 | v2.1 | sim | 초기 수집분 |
| `mimiminsoo/marker_150` | 146 | 44,215 | v2.1 | sim | 확장 수집분 |
| `mimiminsoo/marker_180_real` | 176 | 62,648 | v2.1 | 실기 | 실제 로봇 |
| `mimiminsoo/marker_real` | 30 | 18,433 | v3.0 | 실기 | |
| `mimiminsoo/marker_combined` | 79 | 31,357 | v3.0 | 혼합 | LeRobot 경로용 |
| `mimiminsoo/marker` | 4 | 1,216 | v2.1 | sim | 스모크 테스트용 소형 |

전부 `fps=30`, `robot_type=so101_follower`, 카메라 `front`/`wrist` 2대, action/state 6차원(`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`)입니다.

> **`fps=30`은 메타데이터 라벨일 뿐 실제 샘플링 레이트가 아닙니다.** 수집은 60Hz로 이뤄졌고 변환 시 다운샘플링을 하지 않습니다. `action_horizon=30`인 청크가 커버하는 실제 물리 시간은 1.0초가 아니라 **약 0.5초**입니다.

**포맷 버전 주의**: `v2.1`과 `v3.0`은 디렉토리 레이아웃이 다릅니다. openpi가 요구하는 lerobot 버전과 맞는 것을 받으세요 (이 프로젝트의 openpi 경로는 v2.1 기준으로 검증됨).

### 데이터셋만 따로 받기

```bash
hf download mimiminsoo/marker_100 --repo-type dataset \
    --local-dir ~/.cache/huggingface/lerobot/mimiminsoo/marker_100
```

openpi 학습에 쓰려면 경로가 `~/.cache/huggingface/lerobot/<repo_id>` 여야 합니다 (config의 `repo_id`와 일치). 이미 그 위치에 있으면 `HF_HUB_OFFLINE=1`로 오프라인 학습이 가능합니다.

---

## 수동 다운로드 (스크립트 없이)

```bash
# 모델 (baseline h10)
hf download mimiminsoo/pi0-lora-marker100-h10 --local-dir ./pi0-h10

# 서빙 (openpi 저장소에서)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
uv run scripts/serve_policy.py --port 8000 \
    --default-prompt "Pick up the marker and place it into the cup, then reset the arm to rest state." \
    policy:checkpoint \
    --policy.config=pi0_lora_marker_100_h10 \
    --policy.dir=/절대경로/pi0-h10

# 평가 (leisaac 저장소에서, 다른 터미널)
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-CupStack-v0 --eval_rounds=20 \
    --policy_type=openpi --policy_host=localhost --policy_port=8000 \
    --policy_action_horizon=10 \
    --policy_language_instruction="Pick up the marker and place it into the cup, then reset the arm to rest state." \
    --device=cuda --enable_cameras
```

`--policy_action_horizon`은 위 표의 값과 맞추고, `--policy_language_instruction`은 **위 문자열 그대로** 쓰세요 (학습 시 사용된 프롬프트와 글자 단위로 같아야 합니다).

구버전 `huggingface-cli download`도 동일하게 동작합니다.
