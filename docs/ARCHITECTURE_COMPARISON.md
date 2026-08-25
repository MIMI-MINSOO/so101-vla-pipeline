# π0 (openpi) vs GR00T N1.5 — Architecture Comparison

같은 데이터셋(`marker_100`, 97 episodes)으로 두 VLA 백본을 파인튜닝해 sim에서 비교했습니다. 이건 baseline(π0) 파이프라인과 별개의 **아키텍처 비교 실험**입니다.

## 결과 요약

| | π0 (`pi0_lora_marker_100`) | GR00T N1.5 (`groot_h10`) | GR00T N1.5 fulllora (`groot_h10_fulllora`) |
|---|---|---|---|
| 데이터셋 | `marker_100` (97 episodes) | 동일 | 동일 |
| action_horizon | **30** | **10** | **10** |
| LoRA 대상 | 백본(2B, rank16) + action expert(300M, rank32) | action head만(rank32), 백본 완전 동결 | 백본 포함(`--lora-full-model`) |
| sim 성공률 | **80%+** (20회) | **35%** (7/20) | **~40%** (정확한 시행 횟수 미기록) |

> fulllora 결과: `patches/groot/train_fulllora_and_upload.sh`로 학습/병합/업로드까지 실행되어 `mimiminsoo/groot_h10_fulllora`로 존재하며, sim 평가 성공률은 약 40%로 확인됨(h10 대비 +5%p). 다만 정확한 시행 횟수/신뢰구간은 기록이 남아있지 않아 §3의 표본 크기 주의사항이 h10보다도 더 크게 적용됩니다.

## 성공률 격차 분석 (π0 80%+ vs GR00T h10 35% / fulllora ~40%)

같은 데이터셋인데도 큰 차이가 났습니다. 확신도 순으로 정리합니다.

### 1. action_horizon 불일치 (가장 유력, 근거 있음)

π0가 80%+를 찍은 `pi0_lora_marker_100`은 `action_horizon=30`을 씁니다 — 임의값이 아니라 `marker_100` 데이터의 autocorrelation을 분석해서 **lag ~33에서 상관계수가 0.5 밑으로 떨어지는 지점**을 근거로 정한 값입니다(openpi config 주석에 명시).

GR00T는 `action_horizon=10`으로 학습했는데, 이건 `marker_100`이 아니라 이전 버전 데이터(`marker_50`, 49 episode)용 π0 config였던 `pi0_lora_marker`의 값을 그대로 가져온 것이었습니다. 정작 같은 데이터(`marker_100`)로 80%를 찍은 config와는 다른 horizon을 쓴 셈입니다. horizon이 너무 짧으면 각 청크가 태스크의 실제 동작 패턴을 다 못 담고, 더 자주 재추론하면서 노이즈가 누적될 수 있습니다.

> **추가 참고**: 이 프로젝트의 수집 데이터(HDF5)는 실제로 60Hz로 기록되지만 LeRobot 변환 시 다운샘플 없이 `fps=30`으로만 라벨링돼 있습니다. π0의 `action_horizon=30`이 커버하는 실제 물리 시간은 (fps=30이 암시하는) 1.0초가 아니라 **약 0.5초**입니다. GR00T의 `action_horizon=10`도 동일한 기준으로 재해석하면 약 0.17초에 해당합니다. 자세한 근거는 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 참고.

### 2. VLM 백본이 완전히 동결됨 (h10 config의 설계)

π0의 LoRA는 백본(2B) + action expert(300M) 전체에 걸립니다 — 씬의 "marker"와 "cup"을 어떻게 시각적으로 인식할지까지 같이 적응합니다.

GR00T h10 config는 `--lora-full-model`을 안 줬기 때문에 LoRA가 action head(약 655만 파라미터)에만 붙고, Eagle-2 VLM 백본은 30,000 step 내내 한 번도 업데이트되지 않습니다. 정책은 사전학습 때 본 일반적인 시각 특징을 그대로 쓰고, action head만 이 로봇 동작에 맞춰 학습됩니다.

**검증 결과**: `--lora-full-model`로 백본까지 LoRA를 붙여 재학습한 fulllora는 35%→~40%로 소폭만 개선됐습니다. 백본 동결이 어느 정도는 병목이었을 수 있지만, 이 정도 개선폭으로는 π0와의 격차(80%+ vs ~40%, 40%p 이상)를 설명하기엔 부족합니다 — **#1의 action_horizon 불일치가 더 지배적인 원인일 가능성이 높아짐**.

### 3. 표본 크기 (통계적 주의사항)

7/20 = 35%의 95% 신뢰구간은 대략 17%~58%로 넓습니다. fulllora의 ~40%는 시행 횟수 자체가 기록에 없어 신뢰구간을 계산할 수도 없습니다. π0와의 격차 자체는 크게 봐서 실재하지만, 35%/40%라는 숫자를 너무 정밀하게 믿지는 않는 게 좋습니다.

### 4. denoising steps (미검증 가설)

`--denoising-steps 4`는 Isaac-GR00T가 GR1 등 사전학습된 임베디먼트 기준으로 벤치마크한 기본값입니다. SO-101처럼 LoRA로 새로 적응시킨 `new_embodiment` action head에 4 step이 충분한지는 검증되지 않았습니다.

## 다음에 시도해볼 것

1. `action_horizon=30`(π0와 동일 기준)으로 GR00T 재학습 — fulllora가 격차를 크게 못 줄인 것으로 봐서 가장 유력한 다음 시도. 이 문서 시점까지 미시도 (`custom_data_configs.py`엔 여전히 `So101MarkerH10DataConfig`만 있음)
2. ~~fulllora(`--lora-full-model`) 결과를 실제 sim 평가로 정량화~~ — 완료, ~40% (위 표)
3. denoising-steps 8~16으로 늘려서 추론만 재테스트 (재학습 불필요, 가장 저렴)
4. 정확한 시행 횟수를 기록하며 eval_rounds 30~50으로 재평가해 신뢰구간 좁히기 (h10/fulllora 둘 다)
