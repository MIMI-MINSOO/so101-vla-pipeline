# Troubleshooting

**실행 자체가 막히는 오류**만 모았습니다. 각 항목은 `증상 → 원인 → 해결` 순입니다.

---

## 공통 / 설치

### clone 직후 `.usd`/`.hdf5` 파일이 몇 백 바이트짜리 텍스트

- **증상**: Isaac Sim이 USD를 못 읽거나, 데이터셋 파일이 이상하게 작음
- **원인**: `git-lfs` 없이 clone하면 LFS 포인터 파일만 받아짐 (`.gitattributes`에 `*.usd`, `*.hdf5`가 lfs로 지정돼 있음)
- **해결**:
  ```bash
  sudo apt-get install -y git-lfs
  git lfs install --skip-repo
  cd ~/IsaacLab/source/leisaac && git lfs pull
  ```
- **확인**: `file assets/robots/so101_follower.usd` → `USD crate`가 나와야 정상 (`ASCII text`면 아직 포인터)

---

## IsaacLab / LeIsaac

### `gymnasium.error.NameNotFound: LeIsaac-...`

- **원인**: IsaacLab의 범용 `scripts/reinforcement_learning/*/train.py`는 `leisaac`을 import하지 않아 `LeIsaac-*` gym id가 등록되지 않음
- **해결**: leisaac 저장소 안의 사본(`source/leisaac/scripts/reinforcement_learning/skrl/{train,play}.py`)을 쓰세요. upstream과의 차이는 `import leisaac` 한 줄뿐입니다.

### `FileNotFoundError: USD file not found at path at: '/home/<user>/assets/...'`

- **증상**: 에셋 경로가 홈 디렉토리 같은 엉뚱한 곳을 가리킴
- **원인**: `LEISAAC_ASSETS_ROOT`를 설정하면 코드가 `/assets`를 자동으로 붙여주지 않습니다 (`leisaac/utils/constant.py`의 `_resolve_assets_root`). 저장소 루트까지만 주면 하위 경로가 전부 어긋납니다.
- **해결**: `/assets`까지 명시
  ```bash
  export LEISAAC_ASSETS_ROOT=~/IsaacLab/source/leisaac/assets
  ```
- **주의**: 이 값은 **터미널 세션마다** 다시 설정해야 합니다. `cd`한다고 갱신되지 않고, 한 번 잘못 export하면 그 세션 내내 남아있습니다. 헷갈리면 새 터미널을 여는 게 가장 빠릅니다.
- **확인**: `echo $LEISAAC_ASSETS_ROOT`

---

## openpi (π0)

### `Not enough GPU memory available to create a PhysicsScene` (평가 시)

- **원인**: JAX는 GPU 메모리를 비율만큼 미리 예약합니다. 기본값(75%)이면 정책 서버가 대부분을 가져가서 Isaac Sim의 PhysX가 물리 씬을 만들 메모리가 남지 않습니다.
- **해결**: 서버 기동 전에 비율을 낮추세요.
  ```bash
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
  ```
  학습할 때는(Isaac Sim이 안 떠 있으므로) `0.9`까지 써도 됩니다.

### 학습 시작 시 `RESOURCE_EXHAUSTED`

- **원인**: 정책 서버 등 다른 GPU 프로세스가 이미 떠 있는 상태에서 학습을 시작함
- **해결**: `nvidia-smi`로 확인하고 해당 프로세스를 종료한 뒤 다시 실행

### `Missing norm stats` 또는 이상하게 정규화된 학습

- **원인**: `compute_norm_stats.py`를 실행하지 않았거나, 데이터셋/`action_horizon`이 바뀌었는데 예전 통계가 남아있음
- **해결**:
  ```bash
  uv run scripts/compute_norm_stats.py --config-name pi0_lora_marker_100
  ```
  기존 통계를 강제로 다시 만들려면 `assets/<config_name>/.../norm_stats.json`을 지우고 재실행

### wandb 설치 충돌

- **증상**: 의존성 해결 실패로 설치가 안 됨
- **해결**: `uv add "wandb>=0.16.0"` (openpi 기본 하한 `>=0.19.1`을 낮춤)

---

## Isaac-GR00T

### `NameError: name 'zmq' is not defined` (평가 클라이언트)

- **원인**: leisaac에 GR00T용 추가 의존성(pyzmq/pydantic/msgpack)이 설치되지 않음
- **해결**:
  ```bash
  cd ~/IsaacLab/source/leisaac && pip install -e ".[gr00t]"
  ```

### flash-attn 설치가 컴파일 단계에서 실패

- **원인**: 시스템에 nvcc가 없음
- **해결**: prebuilt wheel로 설치
  ```bash
  pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  ```

### 서버가 체크포인트를 못 읽음 (`config.json` 없음)

- **원인**: `inference_service.py`는 완전한 모델 디렉토리만 로드합니다. LoRA adapter만 있는 체크포인트(`adapter_config.json` + `adapter_model.safetensors`)는 그대로 못 씁니다.
- **해결**: base 모델에 adapter를 병합해 독립 체크포인트를 만드세요.
  ```bash
  python merge_lora.py          # 또는 merge_lora_fulllora.py
  ```

---

## StarVLA

### venv를 활성화했는데도 설치한 패키지를 `import` 못함

- **증상**: `which pip` → `/usr/bin/pip`, `pip --version` → python 3.12
- **원인**: `uv venv`는 venv 안에 pip를 만들지 않습니다(`--seed` 없이는). PATH에서 다음 순번인 시스템 pip가 대신 실행되어 **시스템 Python 3.12에 설치**됩니다.
- **해결**: 이 저장소에서는 **항상 `uv pip install ...`** 을 쓰세요. `uv pip`는 활성화된 venv를 정확히 잡습니다.

### flash-attn: `OSError: CUDA_HOME environment variable is not set`

- **원인**: 드라이버만 있고 CUDA 툴킷(nvcc)이 없는 상태에서 소스 빌드를 시도함
- **해결**: 환경에 맞는 prebuilt wheel을 받아 설치
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"
  gh release download v2.7.4.post1 --repo Dao-AILab/flash-attention \
    --pattern "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  uv pip install ./flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  ```
- **하지 말 것**: `nvidia-cuda-nvcc-cu12` pip 패키지에는 `ptxas`만 있고 실제 `nvcc` 실행파일이 없어 이 용도로 쓸 수 없습니다.

### 학습 시작 시 `MissingCUDAException: CUDA_HOME does not exist`

- **원인**: `train_starvla.py`가 무조건 `DeepSpeedPlugin()`을 생성하고, deepspeed는 **import 시점에** `$CUDA_HOME/bin/nvcc -V`를 호출해 버전을 확인합니다. 컴파일을 안 해도 nvcc가 실재해야 합니다 (flash-attn의 prebuilt wheel 우회는 여기엔 안 통함).
- **해결**:
  ```bash
  sudo apt install --no-install-recommends -y nvidia-cuda-toolkit
  export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
  ```
  Ubuntu 저장소 버전은 CUDA 12.0이지만 torch가 cu124여도 됩니다 (deepspeed는 major 버전만 비교). `--no-install-recommends`를 빼면 불필요한 패키지 60여 개가 같이 깔립니다.
- **참고**: `apt update`가 무관한 저장소(예: librealsense) 때문에 실패하면 `apt install`만 따로 실행하세요.

### `torch.OutOfMemoryError` (24GB GPU)

- **원인**: `freeze_modules`가 비어 있으면 Qwen3-VL-4B 백본 포함 **4.5B 파라미터 전부**가 학습 대상이 됩니다. AdamW는 파라미터당 fp32로 momentum+variance+master weight를 들고 있어야 해서 수십 GB가 필요합니다.
- **해결**: 백본 동결 + 옵티마이저 CPU offload + batch 1 — 셋 다 필요합니다.
  ```bash
  --trainer.freeze_modules "qwen_vl_interface"
  --config_file examples/SO101_Marker/train_files/deepspeed_zero2_offload.yaml
  --datasets.vla_data.per_device_batch_size 1
  ```
  제공된 학습 스크립트들에는 이미 반영돼 있습니다. 실측: bs=1 → 피크 20.3 GiB, bs=2 → 23.6 GiB(위험), bs=4 → 즉시 OOM.

### 정책 서버가 뜨지 않음 (`read_mode_config` 관련)

- **원인**: `config.yaml`과 `dataset_statistics.json`이 `.pt` 파일 기준 **두 단계 위 디렉토리**(런 디렉토리 최상단)에 있어야 합니다.
- **해결**: `push_model_to_hf.py`로 런 디렉토리 전체를 업로드했다면 구조가 맞습니다. `.pt` 파일만 따로 옮기지 마세요.
  ```text
  <run_dir>/
  ├── config.yaml                 ← 여기
  ├── dataset_statistics.json     ← 여기
  └── checkpoints/steps_N_pytorch_model.pt
  ```
