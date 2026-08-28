# Pipeline — Isaac Sim → 데이터 수집 → π0 / GR00T 학습 → 추론

SO-101 로봇팔로 시뮬레이션에서 데이터를 모아 VLA 정책을 학습하고, 다시 시뮬레이션과 실기에서 돌리기까지의 전 과정입니다. 위에서부터 순서대로 실행하면 됩니다.

StarVLA는 환경/데이터 로더 구조가 완전히 달라서 별도 문서로 분리했습니다 → [STARVLA_PIPELINE.md](STARVLA_PIPELINE.md)

## 전체 흐름

```text
[0] 환경 구축              venv 3개 (IsaacLab / openpi / Isaac-GR00T)
        │
[1] Teleop 데이터 수집      SO-101 리더암으로 조작       → datasets/*.hdf5
        │
[2] LeRobot 포맷 변환                                   → ~/.cache/huggingface/lerobot/<id>/marker_100
        │
        ├──[3-A] openpi π0 LoRA 학습                    → checkpoints/pi0_lora_marker_100_h10/marker100_h10/29999
        │              │
        │         [4-A] 정책 서버 기동 (포트 8000)
        │              │
        │         [5-A] Isaac Sim 평가 / [6] 실기 배포
        │
        └──[3-B] Isaac-GR00T N1.5 LoRA 학습             → cupstack-checkpoints/checkpoint-30000-merged
                       │
                  [4-B] 정책 서버 기동 (포트 5555)
                       │
                  [5-B] Isaac Sim 평가
```

디렉토리는 아래 레이아웃을 가정합니다. 다르게 두셨다면 경로만 바꿔서 쓰세요.

```text
~/IsaacLab/source/leisaac   LeIsaac (Isaac Sim 태스크 / teleop / 변환 / 평가)
~/openpi                    openpi  (π0 학습·서빙)
~/Isaac-GR00T               Isaac-GR00T (GR00T N1.5 학습·서빙)
```

세 코드베이스는 **각각 완전히 분리된 venv**를 씁니다 (conda 미사용, `uv` 기반).

---

## [0] 환경 구축

### 0-1. git-lfs (최초 1회)

`.usd`/`.hdf5` 에셋이 LFS로 관리되므로 clone 전에 먼저 설치해야 합니다.

```bash
sudo apt-get install -y git-lfs
git lfs install --skip-repo
```

### 0-2. IsaacLab + LeIsaac

```bash
# IsaacLab (v2.3.2)
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab

# LeIsaac fork를 IsaacLab/source/ 밑에
git clone https://github.com/MIMI-MINSOO/leisaac-cupstack.git ~/IsaacLab/source/leisaac
cd ~/IsaacLab/source/leisaac && git lfs pull

# 실행용 venv (python 3.11, uv)
uv venv --python 3.11 ~/IsaacLab/env_isaaclab
source ~/IsaacLab/env_isaaclab/bin/activate

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

sudo apt install cmake build-essential
cd ~/IsaacLab && ./isaaclab.sh --install

pip install -e "source/leisaac[lerobot]"
pip install numpy==1.26.0   # lerobot과 버전 정합
```

**확인**: `python -c "from leisaac.utils.constant import ASSETS_ROOT; print(ASSETS_ROOT)"` 가 `~/IsaacLab/source/leisaac/assets`를 출력하면 정상.

### 0-3. openpi

```bash
git clone --recurse-submodules https://github.com/MIMI-MINSOO/openpi.git ~/openpi
cd ~/openpi && git checkout so101
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 0-4. Isaac-GR00T

```bash
git clone https://github.com/MIMI-MINSOO/Isaac-GR00T.git ~/Isaac-GR00T
cd ~/Isaac-GR00T && git checkout cupstack-marker100

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -e .[base]

# nvcc가 없으면 flash-attn 소스 빌드가 실패하므로 prebuilt wheel 사용
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

---

## [1] Teleop 데이터 수집

SO-101 리더암으로 시뮬레이션 속 로봇을 직접 조작해 시연 데이터를 모읍니다.

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd ~/IsaacLab/source/leisaac
export LEISAAC_ASSETS_ROOT=~/IsaacLab/source/leisaac/assets   # 반드시 /assets까지

python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-CupStack-v0 \
    --teleop_device=so101leader \
    --port=/dev/ttyACM0 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/marker_100/dataset.hdf5
```

| 키 | 동작 |
|---|---|
| `b` | teleop 시작 |
| `n` | 성공으로 종료 (에피소드 저장됨) |
| `r` | 실패로 리셋 (에피소드 폐기) |

- 최초 실행 시 SO101Leader 캘리브레이션이 대화형으로 진행됩니다 (팔을 중앙에 두고 Enter → 전 관절을 끝까지 움직인 뒤 Enter). 다시 하려면 `--recalibrate`.
- 여러 세션에 나눠 모으려면 `--resume`, 목표 개수를 정하려면 `--num_demos=100`.
- `export LEISAAC_ASSETS_ROOT=...`은 **터미널 세션마다** 다시 실행해야 합니다. `cd`한다고 갱신되지 않습니다.

**결과물**: `./datasets/marker_100/dataset.hdf5` — 성공한 에피소드만 저장됨

---

## [2] LeRobot 포맷 변환

HDF5를 VLA 학습 프레임워크들이 공통으로 읽는 LeRobot Dataset 포맷으로 바꿉니다.

```bash
pip install lerobot==0.4.2 && pip install numpy==1.26.0

python scripts/convert/isaaclab2lerobotv3.py \
    --task_name=LeIsaac-SO101-CupStack-v0 \
    --repo_id=<your-hf-username>/marker_100 \
    --hdf5_root=./datasets/marker_100 \
    --hdf5_files=dataset.hdf5 \
    --task_description="Pick up the marker and place it into the cup, then reset the arm to rest state." \
    --enable_cameras
```
```bash
python scripts/convert/isaaclab2lerobot.py \
  --task_name=LeIsaac-SO101-CupStack-v0 \
  --repo_id=mimiminsoo/test \
  --fps=30 \
  --hdf5_root=./datasets/marker_100 \
  --hdf5_files=dataset.hdf5 \
  --device=cuda:0

```

- `--task_description`은 **학습 시 언어 지시문이 되고, 이후 평가할 때 주는 프롬프트와 글자 그대로 같아야** 합니다.
- 여러 HDF5를 합치려면 `--hdf5_files=a.hdf5,b.hdf5`처럼 콤마로 나열.
- HF Hub에 올리려면 `--push_to_hub` 또는 나중에 `python scripts/convert/push_to_hub.py --repo_id=<id>/marker_100`.

**결과물**: `~/.cache/huggingface/lerobot/<your-hf-username>/marker_100/` (97 episodes / 31,291 frames 규모)

---

## [3-A] openpi π0 LoRA 학습

```bash
cd ~/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 1) 정규화 통계 — 데이터셋이나 action_horizon이 바뀌면 반드시 다시 계산
uv run scripts/compute_norm_stats.py --config-name pi0_lora_marker_100_h10

# 2) 학습 (첫 실행 시 pi0_base 사전학습 가중치를 자동 다운로드)
uv run scripts/train.py pi0_lora_marker_100_h10 --exp-name=marker100_h10
```

- 이 config(`src/openpi/training/config.py`의 `pi0_lora_marker_100_h10`)는 action_horizon=10, batch_size=8(RTX 3090 24GB 기준), 30,000 steps, LoRA만 학습(backbone freeze)입니다.
- action_horizon=30 버전을 쓰려면 config 이름을 `pi0_lora_marker_100`, exp-name을 `marker100_h30`으로 바꾸면 됩니다. **둘을 함께 바꿔야** 체크포인트와 norm stats가 섞이지 않습니다.
- 학습은 GPU를 거의 독점하므로 정책 서버 등 다른 GPU 프로세스를 먼저 종료하세요.
- 이어서 학습하려면 `--resume`, 같은 이름으로 새로 시작하려면 `--overwrite`.
- 래퍼 스크립트 `./run_marker_100.sh`는 위 두 단계를 묶고 실수 방지 가드를 넣은 것입니다. 기본값이 h30이므로 h10으로 돌리려면:
  ```bash
  CONFIG_NAME=pi0_lora_marker_100_h10 EXP_NAME=marker100_h10 ./run_marker_100.sh
  ```

**결과물**: `checkpoints/pi0_lora_marker_100_h10/marker100_h10/29999/`

## [4-A] π0 정책 서버 기동

```bash
cd ~/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35   # Isaac Sim이 쓸 GPU 메모리를 남겨둠

uv run scripts/serve_policy.py \
    --port 8000 \
    --default-prompt "Pick up the marker and place it into the cup, then reset the arm to rest state." \
    policy:checkpoint \
    --policy.config=pi0_lora_marker_100_h10 \
    --policy.dir=checkpoints/pi0_lora_marker_100_h10/marker100_h10/29999
```

JAX는 기본적으로 GPU 메모리를 비율만큼 미리 예약합니다. Isaac Sim과 같은 GPU를 나눠 쓰려면 `0.35` 정도로 낮춰야 PhysX가 물리 씬을 만들 메모리가 남습니다.

## [5-A] Isaac Sim 평가

서버를 켜둔 채 **다른 터미널**에서:

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd ~/IsaacLab/source/leisaac

python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-CupStack-v0 \
    --eval_rounds=20 \
    --policy_type=openpi \
    --policy_host=localhost --policy_port=8000 \
    --policy_action_horizon=10 \
    --policy_language_instruction="Pick up the marker and place it into the cup, then reset the arm to rest state." \
    --device=cuda --enable_cameras
```

- `--policy_port`/`--policy_action_horizon`은 **반드시 명시**하세요. 기본값(5555 / 16)은 GR00T용입니다. `--policy_action_horizon`은 학습에 쓴 config의 action_horizon과 맞추세요(h10이면 10, h30이면 30).
- `--policy_language_instruction`은 [2]단계의 `--task_description`과 글자 그대로 같아야 합니다.
- 에피소드별 영상을 남기려면 `--video_dir=./artifacts/eval_videos`.

서버 기동 + 평가를 한 번에 묶은 래퍼: `./eval_marker_100.sh`

---

## [3-B] Isaac-GR00T N1.5 LoRA 학습

```bash
# modality.json 배치 (GR00T가 요구하는 스키마를 LeRobot 데이터셋에 추가)
cp ~/Isaac-GR00T/examples/SO-100/so100_dualcam__modality.json \
    ~/.cache/huggingface/lerobot/<your-hf-username>/marker_100/meta/modality.json

cd ~/Isaac-GR00T
source .venv/bin/activate
bash run_finetune_cupstack.sh
```

`run_finetune_cupstack.sh`는 `custom_data_configs:So101MarkerH10DataConfig`(action_horizon=10)로 LoRA rank=32 학습을 30,000 step 돌립니다. 백본까지 LoRA를 적용하려면 `train_fulllora_and_upload.sh`를 대신 쓰세요.

중단됐을 때 이어서 하려면 `resume_and_upload.sh` (`--resume` 포함, 세션과 분리해 백그라운드 실행 권장):

```bash
setsid nohup bash resume_and_upload.sh > resume.log 2>&1 < /dev/null &
disown
```

**LoRA 병합** — 서빙하려면 adapter를 base 모델에 병합한 독립 체크포인트가 필요합니다:

```bash
python merge_lora.py          # 또는 merge_lora_fulllora.py
```

**결과물**: `cupstack-checkpoints/checkpoint-30000-merged/`

## [4-B] GR00T 정책 서버 기동

```bash
cd ~/Isaac-GR00T
source .venv/bin/activate
bash run_inference_server.sh
```

내부적으로 실행되는 명령:

```bash
python scripts/inference_service.py \
    --server \
    --model-path ./cupstack-checkpoints/checkpoint-30000-merged \
    --embodiment-tag new_embodiment \
    --data-config custom_data_configs:So101MarkerH10DataConfig \
    --denoising-steps 4 --port 5555
```

## [5-B] Isaac Sim 평가

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd ~/IsaacLab/source/leisaac
pip install -e ".[gr00t]"   # pyzmq/pydantic/msgpack, 최초 1회

python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-CupStack-v0 \
    --eval_rounds=10 --seed=42 \
    --policy_type=gr00tn1.5 \
    --policy_host=localhost --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=10 \
    --policy_language_instruction="Pick up the marker and place it into the cup, then reset the arm to rest state." \
    --device=cuda --enable_cameras
```

`--policy_action_horizon`은 학습에 쓴 값(여기선 10)과 맞추면 됩니다.

---

## [6] 실기 배포 (SO-101 실물)

[4-A]의 π0 서버를 그대로 재사용합니다. 서버를 켜둔 채:

> `real_robot_inference.py`는 openpi의 **websocket** 프로토콜(`ws://host:port`)로만 통신합니다(`leisaac/policy/base.py`의 `WebsocketServicePolicy`). GR00T 서버는 **ZMQ**로 서빙하므로 이 스크립트로는 그대로 연결되지 않습니다. GR00T 정책을 실기에서 돌리려면 ZMQ 클라이언트(`Gr00tServicePolicyClient` 계열)를 쓰도록 스크립트를 별도로 손봐야 합니다.

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd ~/IsaacLab/source/leisaac

python scripts/evaluation/real_robot_inference.py \
    --robot_port=/dev/ttyACM0 \
    --front_camera_index=/dev/v4l/by-path/<front-camera-path> \
    --wrist_camera_index=/dev/v4l/by-path/<wrist-camera-path> \
    --policy_host=localhost --policy_port=8000 \
    --language_instruction="Pick up the marker and place it into the cup, then reset the arm to rest state." \
    --action_horizon=10 --num_episodes=5
```

- 카메라는 **숫자 index 대신 `/dev/v4l/by-path/...` 경로**를 쓰세요. 재연결 시 `/dev/videoN` 번호가 바뀌어 front/wrist가 조용히 뒤바뀔 수 있습니다. `ls -la /dev/v4l/by-path/`로 확인.
- `--max_relative_target`(기본 15.0)은 한 스텝당 관절 이동 폭을 제한하는 안전장치입니다.
- 에피소드 사이에 팔의 토크가 풀립니다. 팔을 손으로 잡아 시작 자세로 옮긴 뒤 Enter를 누르면 그 자리에서 토크가 다시 걸립니다.
