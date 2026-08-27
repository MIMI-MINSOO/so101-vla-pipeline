# Pipeline — StarVLA (Qwen3-VL 백본)

같은 `marker_100` 데이터셋을 StarVLA로 학습해 Isaac Sim에서 돌리는 과정입니다. [PIPELINE_COMMANDS.md](PIPELINE_COMMANDS.md)의 π0/GR00T 경로와 **환경·데이터로더 구조가 완전히 달라서** 문서를 분리했습니다.

데이터 수집([1])과 LeRobot 변환([2])은 π0/GR00T와 동일하므로, 이 문서는 **이미 `marker_100` LeRobot 데이터셋이 있다고 가정**하고 시작합니다.

## 전체 흐름

```text
(선행) marker_100 LeRobot 데이터셋   ← PIPELINE_COMMANDS.md [1]~[2]와 동일
        │
[0] 환경 구축        uv venv 3.10 + flash-attn wheel + nvcc
        │
[1] 데이터 연결      data_registry/data_config.py 등록 → dataloader 단독 검증
        │
[2] 학습            (a) 스모크테스트 20 step
        │           (b) RT-1 backbone warm start
        │           (c) 48h 실전 (115,000 step)
        │                                        → results/Checkpoints/<run_id>/
[3] HF 업로드 → 정책 서버 기동 (GR00T N1.6 ZMQ, 포트 5555)
        │
[4] Isaac Sim 평가   leisaac policy_inference.py --policy_type gr00tn1.6
```

**핵심 제약 (RTX 3090 24GB 기준)**: Qwen3-VL-4B 백본을 포함하면 4.5B 파라미터 전부가 학습 대상이 되어 OOM이 납니다. 이 파이프라인은 전 구간에서 **백본을 동결(`freeze_modules=qwen_vl_interface`)하고 action head만 학습**하며, 옵티마이저 상태는 **DeepSpeed ZeRO-2로 CPU offload**, `per_device_batch_size=1`로 고정합니다. VRAM이 40GB 이상이면 offload를 빼고 batch를 키울 수 있습니다.

---

## [0] 환경 구축

```bash
git clone https://github.com/MIMI-MINSOO/starVLA.git ~/starVLA
cd ~/starVLA && git checkout so101-marker100

uv venv --python 3.10 .venv
source .venv/bin/activate
```

### ⚠️ 반드시 `uv pip install`을 쓸 것

`uv venv`는 venv 안에 pip를 만들지 않습니다. venv를 활성화한 상태에서 그냥 `pip install`을 치면 **시스템 Python(3.12)에 설치**되고, venv 안에서는 `import`가 계속 실패하는 원인 모를 버그처럼 보입니다.

```bash
which pip   # /usr/bin/pip 가 나오면 위 상황임
```

이 저장소 작업에서는 **항상 `uv pip install ...`** 을 쓰세요.

### 0-1. 저장소 의존성

```bash
uv pip install -e .
```

검증된 조합: torch 2.6.0+cu124 / transformers 4.57.0 / deepspeed 0.16.9 / flash-attn 2.7.4.post1

### 0-2. flash-attn — 소스 빌드하지 말고 prebuilt wheel

nvcc 없이 소스 빌드를 시도하면 `OSError: CUDA_HOME environment variable is not set`으로 실패합니다. 환경에 정확히 맞는 released wheel을 받아 설치하세요.

```bash
# 내 환경에 맞는 wheel 태그부터 확인
python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"

gh release download v2.7.4.post1 --repo Dao-AILab/flash-attention \
  --pattern "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
uv pip install ./flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

> `nvidia-cuda-nvcc-cu12` pip 패키지로 우회하려 하지 마세요. 그 패키지에는 `ptxas`만 있고 실제 `nvcc` 실행파일이 없습니다.

### 0-3. CUDA 툴킷 — deepspeed는 import만으로도 nvcc가 필요

`train_starvla.py`는 무조건 `DeepSpeedPlugin()`을 생성하고, deepspeed는 `import` 시점에 `$CUDA_HOME/bin/nvcc -V`를 서브프로세스로 호출합니다. 컴파일을 안 해도 nvcc가 **실재해야** 합니다 (0-2의 prebuilt wheel 트릭은 여기엔 안 통함).

```bash
sudo apt install --no-install-recommends -y nvidia-cuda-toolkit
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
```

- `--no-install-recommends`를 빼면 openjdk 등 불필요한 패키지 60여 개가 같이 깔립니다.
- Ubuntu 저장소 버전은 CUDA 12.0이지만 torch가 cu124여도 문제없습니다 (deepspeed는 major 버전만 비교).
- `CUDA_HOME`은 **터미널 세션마다** 필요합니다. 학습 스크립트들은 내부에서 기본값을 설정해두지만, 수동 실행 시엔 직접 export하세요.

---

## [1] 데이터 연결

StarVLA는 `examples/*/train_files/data_registry/data_config.py`를 자동 스캔해 데이터셋을 등록합니다. `so101-marker100` 브랜치에 이미 들어 있는 파일입니다:

**`examples/SO101_Marker/train_files/data_registry/data_config.py`**

```python
class SO101MarkerDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys  = ["video.front", "video.wrist"]
    state_keys  = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    action_indices = list(range(8))   # trainer.action_horizon과 반드시 같아야 함

DATASET_NAMED_MIXTURES = {
    "so101_marker_task": [("marker_100", 1.0, "so101_marker")],
}
```

- 키 이름들은 `marker_100/meta/modality.json`의 실제 스키마에 맞춘 것입니다. 다른 데이터셋을 쓰려면 그쪽 `modality.json`을 먼저 열어보고 맞추세요.
- `annotation.` 뒤는 **점이 포함된 flat 문자열 키**입니다 (중첩 dict 아님).
- `data_root_dir`는 데이터셋의 **부모 디렉토리**를 가리킵니다 (`.../lerobot/mimiminsoo`). HF 캐시 경로를 그대로 절대경로로 써도 동작하므로 심볼릭 링크를 걸 필요는 없습니다.

### 학습 전에 dataloader만 단독 실행해서 검증

스키마가 안 맞으면 여기서 먼저 걸러집니다. 무거운 학습을 띄우기 전에 반드시 통과시키세요.

```bash
source ~/starVLA/.venv/bin/activate
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
cd ~/starVLA

python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/SO101_Marker/train_files/starvla_qwenpiv3_so101_marker.yaml
```

`Used action keys (reordered): ['single_arm', 'gripper']` 처럼 키가 잡히면 정상입니다.

---

## [2] 학습

세 스크립트를 순서대로 쓰면 됩니다. 전부 `examples/SO101_Marker/train_files/` 안에 있고, 실행 전 `data_root_dir`를 본인 경로로 바꾸세요.

### (a) 스모크테스트 — 20 step

파이프라인이 끝까지 도는지 먼저 확인합니다.

```bash
cd ~/starVLA
bash examples/SO101_Marker/train_files/run_so101_marker_qwenpiv3_train.sh
```

주요 설정:

| 항목 | 값 | 이유 |
|---|---|---|
| `Framework_name` | `QwenPI_v3` | Qwen 백본 + π0 스타일 flow-matching action head |
| `freeze_module_list` | `qwen_vl_interface` | 백본 동결. 빈 문자열로 두면 4.5B 전체 학습 → 24GB에서 OOM |
| `deepspeed_accelerate_config` | `deepspeed_zero2_offload.yaml` | 옵티마이저 상태를 CPU로 offload. 이게 없으면 백본 동결 + bs=1이어도 20GB에서 OOM |
| `per_device_batch_size` | 1 | bs=2는 23.6/24.5 GiB로 여유가 거의 없음, bs=4는 즉시 OOM |
| `max_train_steps` | 20 | 스모크테스트 |

### (b) RT-1 backbone warm start

공개 체크포인트(`StarVLA/Qwen3VL-PI_v3-Bridge-RT_1`, Bridge+Fractal 학습)의 **백본만** 가져와 시작합니다. action_dim이 다르므로(7 vs SO-101의 6) action head는 항상 새로 초기화됩니다.

```bash
bash examples/SO101_Marker/train_files/run_so101_marker_qwenpiv3_from_rt1_train.sh
```

- 스크립트가 백본 체크포인트를 자동으로 `huggingface-cli download` 합니다 (이미 있으면 건너뜀).
- `reload_modules='qwen_vl_interface'` — 백본만 로드. `load_pretrained_backbones`가 이 서브모듈에만 `strict=True`로 적용됩니다.
- 이것도 `max_train_steps=20` 스모크 프로파일입니다. 정상 동작을 확인한 뒤 (c)로 넘어가세요.

### (c) 48h 실전 학습 — 115,000 step

```bash
bash examples/SO101_Marker/train_files/run_so101_marker_qwenpiv3_from_rt1_48h.sh
```

RTX 3090에서 실측한 값 기준으로 잡힌 설정입니다:

| 항목 | 값 | 근거 |
|---|---|---|
| `max_train_steps` | 115,000 | 1.50 s/it × 115k ≈ 47.9시간. 31,291 frame 기준 약 3.7 epoch |
| `per_device_batch_size` | 1 | 피크 20.3 GiB (여유 ~4.3 GiB) |
| `save_interval` | 10,000 | 체크포인트 11~12개 × 약 11GB = 약 130GB 디스크 |
| `eval_interval` | 1,000 | |
| wandb | 활성화 | `wandb_project`/`wandb_entity`를 본인 것으로 수정 |

장시간 실행이므로 세션과 분리해 띄우는 것을 권합니다:

```bash
setsid nohup bash examples/SO101_Marker/train_files/run_so101_marker_qwenpiv3_from_rt1_48h.sh \
  > ~/starVLA/train_48h.log 2>&1 < /dev/null &
disown
```

**결과물**: `results/Checkpoints/<run_id>/checkpoints/steps_N_pytorch_model.pt` + `config.yaml` + `dataset_statistics.json`

---

## [3] HF 업로드 → 정책 서버 기동

StarVLA의 배포 서버는 **GR00T N1.6 ZMQ 프로토콜**로 말합니다. leisaac의 `Gr00t16ServicePolicyClient`와 바이트 호환이라 클라이언트를 따로 만들 필요가 없습니다.

```bash
cd ~/starVLA
bash examples/SO101_Marker/eval_files/run_policy_server.sh
```

이 스크립트는 HF에서 체크포인트를 받아(`hf_repo_id` 변수 수정) `final_model/pytorch_model.pt` 또는 가장 높은 step의 `checkpoints/steps_N_pytorch_model.pt`를 자동으로 찾아 서빙합니다.

> **중요**: `config.yaml`과 `dataset_statistics.json`이 `.pt` 파일보다 **두 단계 위 디렉토리(런 디렉토리 최상단)** 에 있어야 합니다. `push_model_to_hf.py`가 런 디렉토리 전체를 올리므로 정상 업로드했다면 그대로 맞습니다. 수동으로 파일만 옮기면 서버가 뜨지 않습니다.

로컬 체크포인트로 바로 띄우려면:

```bash
source ~/starVLA/.venv/bin/activate
python deployment/model_server/server_policy_gr00t_zmq.py \
  --ckpt_path results/Checkpoints/<run_id>/checkpoints/steps_115000_pytorch_model.pt \
  --port 5555 \
  --use_bf16
```

---

## [4] Isaac Sim 평가

서버를 켜둔 채 **다른 터미널**에서 leisaac 쪽 평가를 실행합니다.

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
cd ~/IsaacLab/source/leisaac

OMNI_KIT_ACCEPT_EULA=Y python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-CupStack-v0 \
    --eval_rounds=10 --seed=42 \
    --episode_length_s=60 \
    --policy_type=gr00tn1.6 \
    --policy_host=localhost --policy_port=5555 \
    --policy_timeout_ms=30000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the marker and place it into the cup, then reset the arm to rest state." \
    --video_dir=./artifacts/starvla_eval/videos \
    --device=cuda --enable_cameras
```

- `--policy_type=gr00tn1.6` — StarVLA 서버가 이 프로토콜을 쓰므로 `gr00tn1.5`가 아닙니다.
- `--policy_action_horizon`은 학습 config의 `action_horizon` 값과 맞추세요. 체크포인트의 `config.yaml`에서 확인할 수 있습니다:
  ```bash
  awk '$1 == "action_horizon:" {print $2; exit}' <ckpt_dir>/config.yaml
  ```
- `--policy_timeout_ms`를 30000으로 넉넉히 잡습니다. 첫 추론 시 모델 컴파일로 시간이 걸립니다.

### 자동화 스크립트

서버 기동(포트 열릴 때까지 대기) → 평가 → 서버 정리까지 묶은 래퍼가 leisaac 쪽에 있습니다:

```bash
cd ~/IsaacLab/source/leisaac
MODEL=rt1 EVAL_ROUNDS=10 bash eval_starvla_marker_models.sh
```

`MODEL`은 `groot` / `pi` / `bs8h10` / `rt1` / `both` 중 선택. 체크포인트는 `artifacts/starvla_models/<model_name>/checkpoints/` 밑에 있어야 하고, `action_horizon`은 각 모델의 `config.yaml`에서 자동으로 읽습니다. 결과(MP4, results.json, 로그)는 `artifacts/starvla_eval/<타임스탬프>/`에 저장됩니다.
