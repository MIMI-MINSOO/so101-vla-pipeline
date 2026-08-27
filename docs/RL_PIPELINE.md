# Pipeline — RLinf GRPO RL fine-tuning (진행 중)

> **현재 상태: 인프라 완성, 실제 실험은 미착수.**
> GRPO 학습이 39 스텝 연속 정상 동작하는 것까지 확인했습니다. 다만 스텝 40의 **체크포인트 저장에서 OOM**이 나서 중단되었고, 이게 해결되기 전까지는 장시간 학습을 돌려도 결과물을 얻을 수 없습니다. 아래 [현재 블로커](#현재-블로커) 참고.
>
> [PIPELINE_COMMANDS.md](PIPELINE_COMMANDS.md)(π0/GR00T)와 [STARVLA_PIPELINE.md](STARVLA_PIPELINE.md)는 실제로 정책을 만들어낸 검증된 경로지만, 이 문서는 **아직 결과물이 없는 진행 중인 연구 트랙**입니다.

## 왜 RL인가

SFT(π0 LoRA)는 100개 데모가 커버한 **좁은 초기 조건**에서만 잘 동작합니다. 시뮬레이터는 초기 조건을 무한히 샘플링할 수 있으므로, 커리큘럼으로 랜덤화 범위를 단계적으로 넓히면서 RL을 돌리면 정책이 잘 동작하는 분포(in-distribution) 자체를 데모 범위 너머로 확장할 수 있습니다.

```text
π0 LoRA SFT 체크포인트 (좁은 초기 분포에서만 성공)
        │
        │  RLinf GRPO + dense reward + curriculum
        ▼
넓어진 랜덤화 범위 전체에서 성공하는 정책
```

## 전체 흐름

```text
(선행) π0 LoRA SFT 체크포인트          ← PIPELINE_COMMANDS.md [3-A]에서 생성
        │
[0] RLinf 환경 구축                    venv + env var 5종 (여기서 제일 많이 막힘)
        │
[1] 체크포인트 변환                    JAX LoRA → merge → PyTorch
        │                              → checkpoints/pi0_lora_marker_100_old_pytorch/
[2] CupStack 태스크 통합               RLinf에 env 등록 + 그룹 초기상태 동기화
        │
[3] Reward / Curriculum 설계           leisaac 쪽 reward v4 + Stage 0/1
        │
[4] GRPO 학습 실행                     run_embodiment.sh
        │
        ▼
   [현재 여기서 막힘] 체크포인트 저장 OOM
```

## 코드 위치

| 저장소 | 브랜치 | 역할 |
|---|---|---|
| [RLinf](https://github.com/MIMI-MINSOO/RLinf) | `cupstack-grpo` | GRPO 학습 엔진, CupStack env 통합, FSDP 수정 |
| [leisaac-cupstack](https://github.com/MIMI-MINSOO/leisaac-cupstack) | `main` | reward v4, curriculum (`tasks/cup_stack/mdp/`) |
| [openpi](https://github.com/MIMI-MINSOO/openpi) | `so101` | 체크포인트 변환 스크립트 (`scratch_*.py`) |

---

## [0] RLinf 환경 구축

```bash
git clone https://github.com/MIMI-MINSOO/RLinf.git ~/RLinf
cd ~/RLinf && git checkout cupstack-grpo
# 의존성 설치는 RLinf 저장소의 자체 문서를 따르세요 (Isaac Sim 포함, venv는 .venv)
```

### ⚠️ 환경변수 5종 — 하나라도 빠지면 전혀 다른 증상으로 실패합니다

이게 이 파이프라인에서 가장 많이 막히는 지점입니다. 아래는 **참고용이 아니라 전부 필수**입니다.

```bash
cd ~/RLinf && source .venv/bin/activate

export LEISAAC_ASSETS_ROOT=/home/minsoo/IsaacLab/source/leisaac/assets
export OMNI_KIT_ACCEPT_EULA=Y
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages/nvidia/cu13/lib:.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}
export ISAAC_PATH=$(pwd)/.venv/lib/python3.11/site-packages/isaacsim
```

| 변수 | 빠뜨리면 나타나는 증상 |
|---|---|
| `OMNI_KIT_ACCEPT_EULA` | `Unable to bootstrap inner kit kernel: EOF when reading a line` → 프로세스가 그냥 멈춤(hang). Kit이 EULA 동의를 stdin으로 묻는데 백그라운드라 stdin이 닫혀 있어 즉시 EOF |
| `VK_ICD_FILENAMES` / `VK_DRIVER_FILES` | RLinf venv의 `activate`가 존재하지 않는 경로(`/etc/vulkan/icd.d/`)를 가리킴. 이 변수가 설정되면 Vulkan 로더는 기본 검색 경로를 완전히 무시하므로 GPU 디바이스 생성이 실패 → PhysX가 CPU로 폴백 → env 생성이 15분+ 걸리거나 사실상 멈춤 |
| `LD_LIBRARY_PATH` | `RuntimeError: nvrtc: error: failed to open libnvrtc-builtins.so.13.0` |
| `ISAAC_PATH` | `run_embodiment.sh`의 기본값이 `/path/to/isaac-sim`이라는 **플레이스홀더 문자열**입니다. Ray 워커에 그대로 전파되어 `TypeError: 'NoneType' object is not callable`로 죽습니다 |
| `LEISAAC_ASSETS_ROOT` | CupStack USD 에셋을 못 찾음 |

백그라운드로 띄울 때는 `< /dev/null`로 stdin을 명시적으로 리다이렉트하세요.

---

## [1] 체크포인트 변환 (JAX → PyTorch)

π0 SFT 체크포인트는 JAX/orbax 포맷이고 LoRA가 활성 상태인데, RLinf는 병합된 PyTorch 체크포인트를 요구합니다.

```bash
cd ~/openpi && source .venv/bin/activate

# 1) LoRA를 base weight에 병합 (JAX 상태 유지)
python scratch_merge_lora.py

# 2) 병합본을 PyTorch로 변환
python scratch_convert_to_old_pytorch.py

# 3) norm_stats를 수동 복사 — 병합/변환 단계가 assets를 안 들고 옵니다
cp -r checkpoints/pi0_lora_marker_100/marker100_h30/29999/assets/mimiminsoo \
      checkpoints/pi0_lora_marker_100_old_pytorch/
```

```text
checkpoints/pi0_lora_marker_100/marker100_h30/29999/params      (JAX, LoRA-active)
    ↓ scratch_merge_lora.py
checkpoints/pi0_lora_marker_100_merged/.../params               (JAX, LoRA-free)
    ↓ scratch_convert_to_old_pytorch.py → openpi의 convert_pi0_checkpoint()
checkpoints/pi0_lora_marker_100_old_pytorch/                    (PyTorch, bf16, 6.6GB)
```

> **⚠️ RLinf에는 pi0 계열 통합체가 2개 있고 체크포인트 포맷이 서로 호환되지 않습니다.**
> - `rlinf/models/embodiment/openpi/` ← **이걸 씁니다.** upstream openpi의 `PI0Pytorch`를 직접 상속하며 `paligemma_with_expert.*` 키 포맷을 소비합니다.
> - `rlinf/models/embodiment/openpi_pytorch/` ← from-scratch 재구현체. 키가 `img.*`/`llm.*`이고 `get_model()`이 `pi05=True`를 하드코딩합니다. RLinf의 `ckpt_convertor/openpi/jax2new.py`는 **이쪽 전용**이라 우리 체크포인트에 쓰면 안 됩니다.
>
> 이름만 보고 "RLinf의 openpi 변환기"를 골랐다가 `Missing key(s): paligemma_with_expert.gemma_expert...`로 실패한 적이 있습니다.

**변환 정확도 검증** (원본 JAX vs 최종 PyTorch, 동일 노이즈 3회):

| 비교 | 평균 상대오차 |
|---|---|
| 병합 단계만 | 1.17% |
| 변환 단계만 | 1.96% |
| **end-to-end** | **1.49%** |

bf16 반올림 + 프레임워크 간 커널 구현 차이가 10-step denoising에 누적된 수준으로, SFT 스킬은 보존됩니다.

---

## [2] CupStack 태스크 통합

`rlinf/envs/isaaclab/tasks/cup_stack.py`(RLinf fork에 커밋됨)가 leisaac의 `LeIsaac-SO101-CupStack-Rewarded-v0`를 RLinf 태스크로 등록합니다.

### 그룹 초기상태 동기화가 왜 필요한가

GRPO의 advantage는 **그룹 내 z-score**입니다:

```python
grouped_rewards = rewards.view(-1, group_size)
advantages = (grouped_rewards - group_mean) / (group_std + 1e-6)
```

그룹 멤버들이 서로 다른 초기 상태에서 시작하면 "이 궤적이 더 좋았다"가 아니라 "이 초기 상태가 더 쉬웠다"를 비교하게 됩니다. 그래서 `_GroupBroadcastResetWrapper`가 reset 직후 그룹 리더(최소 env_id)의 marker pose를 나머지에게 복사합니다.

LIBERO는 녹화된 reset-state 풀에서 인덱스를 뽑는 방식이라 CupStack의 절차적 랜덤화에는 이식이 안 됐고, 대신 위 방식으로 구현했습니다.

**실측 검증** (num_envs=4, group_size=2):

| env | group | marker pos |
|---|---|---|
| 0 | 0 | `[4.1555, -4.0883, 0.0090]` |
| 1 | 0 | `[4.1555, -4.0883, 0.0090]` ← env0과 동일 |
| 2 | 1 | `[-3.8559, -4.0767, 0.0090]` |
| 3 | 1 | `[-3.8559, -4.0767, 0.0090]` ← env2와 동일 |

다음 reset에서는 값 자체가 새로 샘플링되지만 그룹 내부는 여전히 동일합니다.

---

## [3] Reward / Curriculum

### Dense reward v4 (leisaac `tasks/cup_stack/mdp/rewards.py`)

Reach → Secure Grasp → Lift → Safe Lift → Above-Cup → Insertion → Success 순서의 단계별 dense reward입니다. v4에서 `ReachProgress`/`LiftProgress`가 **"직전 스텝 대비 개선"에서 "역대 최고 기록 대비 개선"으로** 바뀌었습니다:

```python
# v3: 밀어냈다가 다시 접근하면 또 보상을 받음 (exploit)
raw = (phi_now - prev_phi) / step_dt

# v4: 새 최고 기록일 때만 보상
improvement = clamp(phi_now - best_phi, min=0.0)
best_phi = max(best_phi, phi_now)
reward = gate * improvement / step_dt
```

`ReachProgress`는 추가로 secure_grasp가 한 번 발동하면 **영구히 gate=0**이 됩니다 (잡은 뒤에 다시 접근 보상을 받을 이유가 없음).

> **로그 해석 주의**: v4에서 `reach_progress`/`lift_progress`가 평평해지는 것은 "정체"가 아니라 "그 이상의 진전이 없음"이라는 뜻입니다. v3와 해석이 다릅니다.

### Curriculum (`tasks/cup_stack/mdp/curriculum.py`)

| Stage | marker 범위 (cup 기준 offset) | 상태 |
|---|---|---|
| 0 | `dx∈[0.13,0.19], dy∈[-0.06,0], yaw±60°` (데모 범위와 동일) | ✅ 구현 + 실측 검증 |
| 1 | `dx∈[0.08,0.24], dy∈[-0.11,0.05], yaw±180°` | ⚠️ 구현됨, 단 **실제 도달가능성 미검증** |
| 2~6 | — | ❌ 미구현 (전체 IK 도달영역 실측 없이는 범위를 정당화할 수 없어 보류) |

전환은 `env.common_step_counter` 기반이며 `CURRICULUM_STAGE1_STEPS`로 제어합니다.

> **⚠️ 이 값을 반드시 재조정하세요.** 기본값 `50_000`은 처리량 측정 전에 넣은 플레이스홀더입니다. 실측 27.3초/스텝 기준으로 50,000스텝은 **약 380시간**이라 현실적이지 않습니다.

---

## [4] GRPO 학습 실행

```bash
cd ~/RLinf && source .venv/bin/activate
# [0]의 환경변수 5종을 먼저 전부 export

bash examples/embodiment/run_embodiment.sh cup_stack_grpo_openpi
```

백그라운드로 띄우려면 (stdin 리다이렉트 필수):

```bash
nohup bash examples/embodiment/run_embodiment.sh cup_stack_grpo_openpi \
    < /dev/null > train.log 2>&1 &
```

크래시 후 최신 체크포인트에서 재시작하려면 `bash recover_and_relaunch.sh` (좀비 ray 워커 정리 + warm start 포함, 단 optimizer state는 복원되지 않음).

### 설정 (RTX 3090 24GB 스모크 스케일)

```yaml
group_size: 2                # GRPO advantage가 ±0.707 두 값만 가짐 — 신호가 거칠음
total_num_envs: 2            # 그룹 1개뿐
precision: "bf16"            # "bfloat16"은 파싱 안 됨 (bf16 / bf16-mixed만 인식)
num_steps: 4                 # denoising steps (SFT는 10을 썼을 가능성)
action_chunk: 10
train_expert_only: true      # VLM freeze, action expert만 학습
add_value_head: false        # GRPO는 critic-free
noise_method: flow_sde
filter_rewards: false        # ← 반드시 false. 아래 설명 참고
fsdp_config:
  use_orig_params: true      # false면 FlatParameter로 뭉개져 frozen 필터링이 작동 안 함
  cpu_offload: false         # actor.enable_offload와 충돌
  mixed_precision: null      # 모델이 이미 bf16 네이티브
  gradient_checkpointing: false   # openpi 필수 제약
weight_syncer:
  init_sync: {enabled: false}
```

> **`filter_rewards`를 켜면 advantage가 전부 NaN이 됩니다.** 이 필터는 sparse 0/1 성공 보상을 전제로 "그룹 전체가 성공/실패한 그룹"을 드롭하는 장치인데, dense reward의 에피소드 합은 작은 음수(-0.04 근처)라 기본 bound `[0.1, 0.9]`에 절대 들어가지 않아 **매 스텝 모든 그룹이 마스킹**됩니다.

### 실측 결과 (39 스텝)

| 지표 | 값 | 판정 |
|---|---|---|
| step time | 27.3초/스텝 | 1,000스텝 ≈ 7시간 16분 |
| `actor/total_loss` | -0.021 ~ -0.075 | 정상 (발산 아님) |
| `actor/ratio` | 0.974 ~ 1.364 | 정상 범위 |
| `actor/grad_norm` | 77.8 ~ 152.7 | `clip_grad=1.0`으로 클리핑됨. ⚠️ 절대값이 큰 편 — 장기 학습 시 관찰 필요 |
| `advantages` | ±0.707, mean≈0 | 정상 (group_size=2의 z-score 이론값과 일치) |
| `reward` (dense) | -0.0007 ~ -0.0010 | dense reward가 실제로 흐름 |
| `success_once` | 0.5 | 그룹 2개 궤적 중 1개 성공 |
| OOM / NaN / 크래시 | 0건 | — |

### 확인할 지표

| 지표 | 의미 |
|---|---|
| `Episode_Termination/success` | `marker_in_cup` 절대 성공률. TensorBoard에 자동 로깅되며 reward 평균보다 신뢰할 수 있음 |
| `advantages` 분포 | 그룹 내 std가 0에 가까우면 학습 신호가 없다는 뜻 |
| peak GPU memory | `torch.cuda.max_memory_allocated()` |

---

## 현재 블로커

**스텝 40의 체크포인트 저장에서 OOM** — `save_checkpoint` → `checkpoint.py::state_dict` → `torch.distributed.checkpoint.state_dict.get_state_dict()`.

학습 루프와는 무관한 별도 경로입니다. weight sync 쪽에서 겪었던 것과 같은 성격의 DCP 오버헤드 문제(단일 GPU인데도 DCP gather가 6GB+를 추가로 씀)인데, 저장 시에는 model state_dict뿐 아니라 **optimizer state까지** 함께 gather하므로 더 무겁습니다.

weight sync 때는 `world_size == 1` 전용 경로를 추가해 DCP를 우회하는 방식으로 해결했으므로, 저장 경로에도 같은 접근이 통할 가능성이 높습니다.

## 다음 할 일

| 우선순위 | 항목 | 왜 |
|---|---|---|
| **A** | 체크포인트 저장 OOM 수정 | 저장이 안 되면 아무리 오래 학습해도 결과물이 없음 |
| **B** | SFT 정책의 현재 ID 경계 측정 (RL 없이) | "얼마나 넓혔는지" 말하려면 출발점이 필요. Stage 0 / Stage 1 / 더 넓은 범위에서 각각 SFT 성공률 측정 |
| **C** | Stage 0에서 장기 학습 (수백~수천 스텝) | RL이 실제로 개선을 만드는지 확인 |
| **D** | Stage 1 전환 후 학습 | ID 확장의 첫 단계 |
| **E** | Stage 2+ 설계 | 전체 5-joint IK 도달영역 실측이 선행돼야 함 |
| **F** | 최종 평가: 여러 랜덤화 범위에서 SFT vs RL 성공률 비교 | 목표 달성 여부 정량화 |

## 알려진 한계

| 항목 | 상태 |
|---|---|
| `mixed_precision` 비활성화 | dtype mismatch 우회책. 근본 수정 아님. 39스텝에선 정상이었으나 장기 영향 미검증 |
| `grad_norm` 77~152 | 클리핑되어 당장은 문제없으나 장기 학습에서 관찰 필요. 필요 시 lr(5e-6) 하향 검토 |
| Curriculum Stage 1 도달가능성 | 단일 높이 슬라이스 추정만 했고 전체 IK 스윕은 안 함 |
| denoising steps 4 vs 10 | SFT는 10을 썼을 가능성. 품질/속도 트레이드오프 미검증 |
| 좀비 ray 워커 | 실패한 실행 후 GPU/RAM을 잡고 남는 현상 반복 관찰. 실행 전 `nvidia-smi` 확인, 필요 시 `ray stop --force` |
| 디버그 print 잔존 | `[DEBUG_KEY_MISMATCH]` 등, 파이프라인 안정화 후 정리 예정 |
