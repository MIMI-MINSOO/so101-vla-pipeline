# π0 (openpi) vs GR00T N1.5 — Architecture Comparison

같은 데이터셋(`marker_100`, 97 episodes)으로 두 VLA 백본을 파인튜닝해 sim에서 비교했습니다. 이건 baseline(π0) 파이프라인과 별개의 **아키텍처 비교 실험**입니다.

## 결과 요약

| | π0 (`pi0_lora_marker_100`) | GR00T N1.5 (`groot_h10`) | GR00T N1.5 fulllora (`groot_h10_fulllora`) |
|---|---|---|---|
| 데이터셋 | `marker_100` (97 episodes) | 동일 | 동일 |
| action_horizon | **30** | **10** | **10** |
| LoRA 대상 | 백본(2B, rank16) + action expert(300M, rank32) | action head만(rank32), 백본 완전 동결 | 백본 포함(`--lora-full-model`) |
| sim 성공률 (20회) | **80%+** | **35%** (7/20) | 재학습 실행됨, 이 문서 기준 정량 결과 미확인 |

> ⚠️ fulllora 결과: 학습/병합/업로드까지는 `patches/groot/train_fulllora_and_upload.sh`로 실행됐고 체크포인트가 `mimiminsoo/groot_h10_fulllora`로 존재하지만, 이 문서 저장소가 참조하는 로컬 기록(`PIPELINE_NOTES.md`, 2026-08-07~10 작성)엔 fulllora의 sim 평가 성공률이 남아있지 않습니다. 재평가 후 이 표를 갱신하세요.

## 성공률 격차 분석 (π0 80%+ vs GR00T h10 35%)

같은 데이터셋인데도 큰 차이가 났습니다. 확신도 순으로 정리합니다.

### 1. action_horizon 불일치 (가장 유력, 근거 있음)

π0가 80%+를 찍은 `pi0_lora_marker_100`은 `action_horizon=30`을 씁니다 — 임의값이 아니라 `marker_100` 데이터의 autocorrelation을 분석해서 **lag ~33에서 상관계수가 0.5 밑으로 떨어지는 지점**을 근거로 정한 값입니다(openpi config 주석에 명시).

GR00T는 `action_horizon=10`으로 학습했는데, 이건 `marker_100`이 아니라 이전 버전 데이터(`marker_50`, 49 episode)용 π0 config였던 `pi0_lora_marker`의 값을 그대로 가져온 것이었습니다. 정작 같은 데이터(`marker_100`)로 80%를 찍은 config와는 다른 horizon을 쓴 셈입니다. horizon이 너무 짧으면 각 청크가 태스크의 실제 동작 패턴을 다 못 담고, 더 자주 재추론하면서 노이즈가 누적될 수 있습니다.

> **추가 참고**: 이 프로젝트의 수집 데이터(HDF5)는 실제로 60Hz로 기록되지만 LeRobot 변환 시 다운샘플 없이 `fps=30`으로만 라벨링돼 있습니다. π0의 `action_horizon=30`이 커버하는 실제 물리 시간은 (fps=30이 암시하는) 1.0초가 아니라 **약 0.5초**입니다. GR00T의 `action_horizon=10`도 동일한 기준으로 재해석하면 약 0.17초에 해당합니다. 자세한 근거는 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 참고.

### 2. VLM 백본이 완전히 동결됨 (h10 config의 설계)

π0의 LoRA는 백본(2B) + action expert(300M) 전체에 걸립니다 — 씬의 "marker"와 "cup"을 어떻게 시각적으로 인식할지까지 같이 적응합니다.

GR00T h10 config는 `--lora-full-model`을 안 줬기 때문에 LoRA가 action head(약 655만 파라미터)에만 붙고, Eagle-2 VLM 백본은 30,000 step 내내 한 번도 업데이트되지 않습니다. 정책은 사전학습 때 본 일반적인 시각 특징을 그대로 쓰고, action head만 이 로봇 동작에 맞춰 학습됩니다. 마커처럼 작고 위치 정밀도가 중요한 물체를 다루는 태스크에서는 백본이 씬에 적응하지 못한 게 병목이었을 가능성이 있습니다. (→ fulllora 재학습으로 검증 시도, 위 표 참고)

### 3. 표본 크기 (통계적 주의사항)

7/20 = 35%의 95% 신뢰구간은 대략 17%~58%로 넓습니다. π0와의 격차 자체는 크게 봐서 실재하지만, 35%라는 숫자를 너무 정밀하게 믿지는 않는 게 좋습니다.

### 4. denoising steps (미검증 가설)

`--denoising-steps 4`는 Isaac-GR00T가 GR1 등 사전학습된 임베디먼트 기준으로 벤치마크한 기본값입니다. SO-101처럼 LoRA로 새로 적응시킨 `new_embodiment` action head에 4 step이 충분한지는 검증되지 않았습니다.

## 다음에 시도해볼 것

1. `action_horizon=30`(π0와 동일 기준)으로 GR00T 재학습 — 근거가 가장 뚜렷하지만, 이 문서 시점까지 시도되지 않음 (`custom_data_configs.py`엔 여전히 `So101MarkerH10DataConfig`만 있음)
2. fulllora(`--lora-full-model`) 결과를 실제 sim 평가로 정량화
3. denoising-steps 8~16으로 늘려서 추론만 재테스트 (재학습 불필요, 가장 저렴)
4. 개선되면 eval_rounds를 30~50으로 늘려 신뢰구간 좁히기
