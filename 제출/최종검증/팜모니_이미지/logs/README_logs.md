# 로그 안내

- `logs/`: 연구실 GPU 서버 `~/storage/nadan/farmai_img/logs/` 전체 사본(2026-10-17)입니다. 최종 모델과 직접 관련된 로그는 다음과 같습니다.

| 단계 | 로그 |
|---|---|
| 가짜 제작(SD·SDXL, 5판·6판·7판, Qwen) | `s07`~`s20b.log`, `gen_v5*.log`, `gen_v6.log`, `qsmoke*.log` |
| 기본 판별기·전문가 C3v2 빌드 | `build_img03.log`, `build_img05.log`, `sealed_c3v2.log` |
| 정밀 지문 문턱 | `finefp_v2.log` |
| 전문가 H 제작·학습·봉인 | `h_gen_*.log`, `h_feats_*.log`, `h2.log`, `h2_r3.log`, `build8.log` |
| E_gr | `ip_feats.log`, `ip_train.log`, `ip_r4.log` |
| 나단9 점검·오탐 분해 | `n9_audit.log`, `n9_fp_split.log` |
| 나단10 점검·기각 | `im68*.log`~`im73.log` |
| E_gpt | 코덱스 `code/codex_E_gpt/run_v3.log`, `result.json`, `verification_*.json` |

## 재현·시간 기록
- 나단11 공식 `self_check.py`(서버 RTX 6000 Ada, 2026-10-16 14:2x)
  - RESULT: PASS, 공개 21장 accuracy 1.000, 모델 로드 7.261초 + 예측 1.93초, 순서 독립성 OK
  - 경고 1건: requirements 추가 패키지 2개(opencv-python, safetensors, 허용 목록)
- 나단11과 나단9의 실제 패키지 예측 비교(서버, 같은 날, 0.5 기준 판정)

  | 자료 | 나단9 → 나단11 |
  |---|---|
  | 공개 21 | 1.000 → 1.000 |
  | R5 700 | .9443 → .9443 |
  | newreal 795 | .9937 → .9937 |
  | gpt2 | .880 → 1.000 |
  | gpt3 | .965 → 1.000 |

  gpt2·gpt3은 최종 E_gpt의 학습 자료라 표본 안 수치입니다.
- Colab T4 시간(사용자 측정, `colab_시간측정.py`, 524장 환산 합계): 265.1 / 273.7 / 272.5 / 272.5초
