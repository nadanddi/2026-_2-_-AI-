---
name: training-cleaning-noncausal-ok
description: User ruling 2026-10-10 — cleaning TRAINING inputs (e.g. repairing injected-noise days) may use past and future values; evaluation-row features stay current/past only
metadata:
  node_type: memory
  type: project
  originSessionId: 96fe1d15-2691-4cbd-a117-4a3e2d96bdf9
  modified: 2026-10-10T07:24:07.292Z
---

User ruling 2026-10-10 ("해봐"). I had asked whether training-data cleaning may use future values. The case was centred smoothing of the injected-noise inputs (in_co2, in_hum, in_temp) on CO2-rough training days (catalog 6.437).

The answer: cleaning or preprocessing TRAINING rows may use both earlier and later values of the same greenhouse. Features of evaluation rows (test_X) must still use only the same greenhouse's current and earlier inputs. Training-row features are still built in the MASK world. In validation, the held-out rows keep their ORIGINAL inputs, so the validation stays honest.

Official rule text the user pasted (2026-10-10). Allowed: "공개된 train_X·train_y 전체로 학습한 모델", "공개 학습 자료로 만든 모델·전처리를 평가 입력에 적용". Banned: "해당 행보다 뒤의 입력 — 전처리·탐색·연결·보정·후처리 등 그 행의 예측값에 이르는 어떤 단계에서도"; "평가 입력이나 그 예측값으로 모델·전처리·판정 기준 학습·갱신"; "평가 자료 전체의 통계·분포·구성에 맞춘 예측값·판정 기준 조정". My reading is that "해당 행" means the predicted (evaluation) row. If the user reads it to include training rows, stop and discard any experiment that relies on this ruling. Never use test-composition-weighted scores in adoption rules.

**Why:** TD17 found that on those days the labels (sub_temp) are as smooth as on normal days, while the indoor sensors are corrupted. That pattern points to input-only noise injection in training, with clean test inputs. Repairing the inputs could let the true labels be used fully. An earlier causal-smoothing repair (6.21) hurt, probably because the causal smoother added lag.

**How to apply:** Apply the cleaning to training rows only. Never clean or alter test_X inputs. Record the ruling in the catalog entry of any experiment that uses it. Related: [[rule5-interpretation]], [[skill-not-luck]].
