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

---

## RLinf (GRPO RL)

RLinf 실행 실패의 대부분은 **환경변수 누락**입니다. [RL_PIPELINE.md](RL_PIPELINE.md)의 env var 5종을 먼저 전부 export했는지 확인하세요 — 하나만 빠져도 아래처럼 전혀 다른 증상으로 나타나 원인 추적이 어렵습니다.

### `Unable to bootstrap inner kit kernel: EOF when reading a line` (프로세스가 멈춤)

- **원인**: `OMNI_KIT_ACCEPT_EULA` 미설정. Kit이 EULA 동의를 stdin으로 묻는데 백그라운드 실행이라 stdin이 닫혀 있어 즉시 EOF를 만납니다. 크래시가 아니라 hang이라 더 헷갈립니다.
- **해결**: `export OMNI_KIT_ACCEPT_EULA=Y` + 백그라운드 실행 시 `< /dev/null` 명시

### `TypeError: 'NoneType' object is not callable` (AppLauncher)

- **원인**: `run_embodiment.sh`의 `ISAAC_PATH` 기본값이 `/path/to/isaac-sim`이라는 **플레이스홀더 문자열**입니다. 이 값이 Ray 워커에 전파되면 `isaacsim/__init__.py::expose_api()`가 `SimulationApp`을 못 찾아 `None`으로 남깁니다.
- **해결**: `export ISAAC_PATH=$(pwd)/.venv/lib/python3.11/site-packages/isaacsim`
- **참고**: standalone 스크립트로 테스트할 때는 이 변수를 아예 설정하지 않아 문제가 없다가, `run_embodiment.sh`를 거치면서 처음 나타납니다.

### env 생성이 15분 넘게 걸리거나 사실상 멈춤 / Vulkan `ERROR_INCOMPATIBLE_DRIVER`

- **원인**: RLinf venv의 `activate`가 `VK_ICD_FILENAMES`를 존재하지 않는 경로(`/etc/vulkan/icd.d/`)로 설정합니다. 이 변수가 설정되면 Vulkan 로더는 **기본 검색 경로를 완전히 무시**하므로 GPU 디바이스 생성이 통째로 실패하고, PhysX가 CPU로 폴백합니다.
- **해결**: 실제 경로로 런타임 override (activate 스크립트 자체는 건드리지 않음)
  ```bash
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
  ```

### `RuntimeError: nvrtc: error: failed to open libnvrtc-builtins.so.13.0`

- **원인**: RLinf venv의 torch가 `nvidia-cuda-nvrtc-cu13`을 설치는 했지만 동적 링커 검색 경로에 없음
- **해결**:
  ```bash
  export LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages/nvidia/cu13/lib:.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}
  ```

### `ValueError` — precision 설정

- **원인**: `rlinf/config.py::torch_dtype_from_precision`이 `"bf16"` / `"bf16-mixed"`만 인식합니다. YAML에 `precision: bfloat16`이라고 쓰면 실패합니다.
- **해결**: `precision: "bf16"`

### advantage가 전부 NaN (reward는 정상값)

- **원인**: `algorithm.filter_rewards`는 sparse 0/1 성공 보상을 전제로 "그룹 전체가 성공/실패한 그룹"을 드롭하는 장치입니다. dense reward의 에피소드 합은 작은 음수(-0.04 근처)라 기본 bound `[0.1, 0.9]`에 절대 들어가지 않아 매 스텝 모든 그룹이 마스킹됩니다.
- **해결**: `filter_rewards: false`

### 실패한 실행 후 GPU/RAM이 계속 잡혀 있음

- **원인**: 좀비 ray 워커. 관측 사례: 고아 actor 하나가 21GB RSS 점유
- **해결**:
  ```bash
  pgrep -f "ray::" | xargs -r kill -9
  pkill -9 -f "train_embodied_agent"
  ```
  `recover_and_relaunch.sh`에 이 정리 단계가 포함돼 있습니다. 실행 전후로 `nvidia-smi` 확인을 습관화하세요.

### 체크포인트 저장(`save_interval`) 스텝에서 OOM

- **상태**: **미해결 — 현재 이 파이프라인의 최우선 블로커**
- **원인**: `save_checkpoint` → `torch.distributed.checkpoint.state_dict.get_state_dict()`의 DCP gather 오버헤드. 단일 GPU라 실제 샤딩이 없는데도 6GB+를 추가로 씁니다. 저장 시에는 optimizer state까지 함께 gather하므로 weight sync 때보다 무겁습니다.
- **우회**: `runner.save_interval`을 크게 잡아 저장을 미루면 학습 자체는 계속 돌지만, 결과물을 못 얻으므로 근본 해결이 필요합니다. weight sync 경로에 적용한 `world_size == 1` DCP 우회를 저장 경로에도 적용하는 것이 유력한 방향입니다.
