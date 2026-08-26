# Pipeline Commands

복붙 가능한 명령어 모음입니다. 이 문서 전체에서 `$LEISAAC_ROOT` / `$OPENPI_ROOT` / `$GROOT_ROOT`를 씁니다 — **본문 텍스트를 직접 고쳐 쓰지 말고, 아래처럼 실제 쉘 변수로 먼저 export하세요.** (`$LEISAAC_ROOT`를 export 없이 그대로 복붙하면 빈 문자열로 치환되어 `cd $LEISAAC_ROOT`가 `cd`(인자 없음) → 홈 디렉토리로 이동해버리고, 그 상태에서 만든 `LEISAAC_ASSETS_ROOT` 등 다른 변수까지 줄줄이 잘못된 값으로 export되니 주의하세요.)

```bash
export LEISAAC_ROOT=~/IsaacLab/source/leisaac
export OPENPI_ROOT=~/openpi
export GROOT_ROOT=~/Isaac-GR00T
```

각 코드베이스는 **완전히 분리된 venv**를 씁니다 (conda 미사용, `uv` 기반). git-lfs가 필요합니다 (`.usd`/`.hdf5` 에셋이 LFS로 관리됨).

---

## 0. 환경 준비

```bash
# git-lfs (최초 1회, sudo 필요)
sudo apt-get install -y git-lfs
git lfs install --skip-repo

# IsaacLab (v2.3.2)
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab

# LeIsaac fork (IsaacLab/source/ 밑에)
git clone https://github.com/MIMI-MINSOO/leisaac-cupstack.git ~/IsaacLab/source/leisaac
cd ~/IsaacLab/source/leisaac && git lfs pull

# IsaacLab 실행용 venv (python 3.11, uv)
uv venv --python 3.11 ~/IsaacLab/env_isaaclab
source ~/IsaacLab/env_isaaclab/bin/activate

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

sudo apt install cmake build-essential   # IsaacLab 빌드 의존성
cd ~/IsaacLab && ./isaaclab.sh --install

pip install -e "source/leisaac[lerobot]"
pip install numpy==1.26.0   # lerobot과 버전 정합
```

```bash
# openpi (별도 venv, uv)
git clone --recurse-submodules https://github.com/EverNorif/openpi.git ~/openpi
cd ~/openpi && git checkout so101
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

```bash
# Isaac-GR00T (별도 venv, python 3.10)
git clone https://github.com/NVIDIA/Isaac-GR00T ~/Isaac-GR00T
cd ~/Isaac-GR00T
git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -e .[base]
# nvcc가 없으면 flash-attn 사전빌드 wheel 사용
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

---

## 1. Teleop 데이터 수집 (SO-101 리더암, IsaacLab venv)

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd $LEISAAC_ROOT
export LEISAAC_ASSETS_ROOT=$(pwd)/assets   # 반드시 /assets까지! (repo root만 주면 경로가 틀어짐)

python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-CupStack-v0 \
    --teleop_device=so101leader \
    --port=/dev/ttyACM0 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/marker_100/dataset.hdf5
```
조작키: `b`=시작, `n`=성공 종료, `r`=실패 리셋. 최초 실행 시 SO101Leader 캘리브레이션이 대화형으로 진행됩니다.

## 2. HDF5 → LeRobot Dataset 변환

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

---

## 3-A. openpi π0 LoRA fine-tuning

```bash
cd $OPENPI_ROOT
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# config.py에 patches/openpi_config_pi0_lora_marker_100.py 내용을 TrainConfig 목록에 추가한 뒤
uv run scripts/compute_norm_stats.py --config-name pi0_lora_marker_100
uv run scripts/train.py pi0_lora_marker_100 --exp-name=marker100_h30
```

**서버 기동:**
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
uv run scripts/serve_policy.py \
    --port 8000 \
    --default-prompt "Pick up the marker and place it into the cup, then reset the arm to rest state." \
    policy:checkpoint \
    --policy.config=pi0_lora_marker_100 \
    --policy.dir=checkpoints/pi0_lora_marker_100/marker100_h30/29999
```

**Sim 평가 (다른 터미널, IsaacLab venv):**
```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd $LEISAAC_ROOT
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-CupStack-v0 \
    --eval_rounds=20 \
    --policy_type=openpi \
    --policy_host=localhost --policy_port=8000 \
    --policy_action_horizon=30 \
    --policy_language_instruction="Pick up the marker and place it into the cup, then reset the arm to rest state." \
    --device=cuda --enable_cameras
```

---

## 3-B. Isaac-GR00T N1.5 LoRA fine-tuning

```bash
# meta/modality.json 배치 (LeRobot 데이터셋에 GR00T가 요구하는 스키마 추가)
cp $GROOT_ROOT/examples/SO-100/so100_dualcam__modality.json \
    ~/.cache/huggingface/lerobot/<your-hf-username>/marker_100/meta/modality.json

# patches/groot/custom_data_configs.py 를 $GROOT_ROOT/ 에 복사한 뒤
cd $GROOT_ROOT
source .venv/bin/activate
bash patches/groot/run_finetune_cupstack.sh   # 또는 train_fulllora_and_upload.sh (백본까지 LoRA)
```

**서버 기동:**
```bash
python $GROOT_ROOT/scripts/inference_service.py \
    --server \
    --model-path $GROOT_ROOT/cupstack-checkpoints/checkpoint-30000-merged \
    --embodiment-tag new_embodiment \
    --data-config custom_data_configs:So101MarkerH10DataConfig \
    --denoising-steps 4 --port 5555
```

**Sim 평가:**
```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd $LEISAAC_ROOT
pip install -e ".[gr00t]"   # pyzmq/pydantic/msgpack (한 번만)

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

---

## 4. 실로봇 배포 (openpi 서버 재사용)

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd $LEISAAC_ROOT
python scripts/evaluation/real_robot_inference.py \
    --robot_port=/dev/ttyACM0 \
    --front_camera_index=/dev/v4l/by-path/<front-camera-path> \
    --wrist_camera_index=/dev/v4l/by-path/<wrist-camera-path> \
    --policy_host=localhost --policy_port=8000 \
    --language_instruction="Pick up the marker and place it into the cup, then reset the arm to rest state." \
    --action_horizon=10 --num_episodes=5
```
