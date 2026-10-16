# submission_29 (2026-10-16, 집 클로드, 가설 검정 — 27번 + 온도 시간 감쇠 예측 사슬 + EC 앵커 확률 비례)
- 전 과정: `python checks_v29.py produce <원본 자료 폴더>` → submission_29.csv (중간 산출물 output_29/) / 검사: `python checks_v29.py check <같은 폴더>` → checks_v29.log
- 연결표: 27번과 같음(온도/linker base·t1·t3 → linker_v7/ap_fl.py train(입력 비운 사본) / eval(실제) → fl_eval_links.csv, fl_eval_anchor.csv)
- 온도: temp_candidate_nx27.csv → code/make_temp_v22_anchor.py → code/make_temp_v27_tx3.py → **code/make_temp_v29_chain.py**(정답 없는 같은 농장 앞 번호 평가 기록의 최종 23시 예측을 오늘 새벽 기준으로, .9·exp(−h/8))
- EC: **EC/ec29_post.py**(27번 NH + 어제 EC 앵커 × .8 × p0) → output_29/ec_nh_fl.csv
- 결합: merge_temp_ec_v29.py.  의존성 경로: 환경변수 EC_CORE_DEPENDENCY_PATH(numpy·pandas·sklearn·lightgbm)·EC_DEPENDENCY_PATH(torch·tabpfn), 온도 스크립트는 PYTHONPATH 에 같은 핵심 경로.
- 자세한 내용: ../설명자료_29_27온도사슬_EC앵커확률비례_가설검정.md

# submission_27 (2026-10-15, 집 클로드, 가설 검정 — 5점 모델, 현장형 링커 FL)
- 연결표: 온도/linker(base·t1·t3, ap_v2 없음) 를 실제 자료(LINKER_WORK=lw)와 test_X 입력을 모두 비운 사본(lwm)에서 → 온도/linker_v7/ap_fl.py train <out>(비운 사본: FL 가중치 w_fl.npy·학습 짝) / ap_fl.py eval <out> <w_fl.npy>(실제 자료: 평가 기록별 0시 독립 판단 → fl_eval_anchor.csv)
- 온도: 온도/temp_candidate_nx27.csv(NX 기본 = W40G-S 에서 day·second 제거; 고정 입력, 재생성 온도/code/make_submission_v26_temp.py) → code/make_temp_v22_anchor.py(FL 앵커) → code/make_temp_v27_tx3.py(고EC 동 추운 행 +.25℃) → 온도/temp_candidate_v27.csv
- EC: EC/ec27_post.py <fl_eval_anchor.csv>(NH = SG2 전 R3+TabPFN + HR5, 이어서 FL 어제 EC 앵커) → EC/output_27/ec_nh_fl.csv
- 결합: merge_temp_ec_v27.py → submission_27.csv / 규정 검사·전 과정 재현: checks_v27.py <원본 CSV 폴더> → checks_v27.log (ALL PASS)
- 자세한 내용: 설명자료_27_5점모델_현장형링커_가설검정.md

# submission_25 (2026-10-15, 집 클로드, 가설 검정 — '5점(현장 정보)' 구성)
- 연결표: 온도/linker(base·t1·t3·ap_v2) → linker_v15r/v15r_eval_pkg.py(평가 60기록 전날 짝) / 같은 링커를 test_X 입력을 모두 비운 사본에서 → linker_v15r/v15r_mirror_pkg.py(학습 400기록 전날 짝, MASK 세계)
- 온도: 15회차 온도에서 V3 → V3L(같은 동 전날 창, 온도/code/make_temp_v25_v3lbase.py) → 이후 사슬·v7 앵커·TNA8 그대로 → 온도/temp_candidate_v25.csv
- EC: SG2·W2·B5 를 뺀 5점 구성 — R3+TabPFN(SG2 전) → HR5(학습 기준도 SG2 전 OOF: EC/hr5_train_base_nosg_oof.csv) → EV3L(같은 동 전날 창 EC GRU) 10% (EC/f5e_post.py) → EC/output_25/ec_f5e.csv
- 결합: merge_temp_ec_v25.py → submission_25.csv / 규정 검사·전 과정 재현: checks_v25.py <원본 CSV 폴더> → checks_v25.log
- 자세한 내용: 설명자료_25_5점구성_V3L온도_EV3L_EC_가설검정.md

# submission_24 (2026-10-14, 집 클로드, 가설 검정)
- 온도: 14회차 온도 구성의 기본 단계에 2층 GRU 구성원 V3 를 10% 섞음(온도/code/make_temp_v24_v3base.py, P9 = temp_candidate_v13.csv) -> 이후 14회차와 같은 사슬(make_temp_v18_chain.py) -> 링커 -> make_temp_v22_anchor.py -> linker_v15r -> make_temp_v23_tna8.py -> 온도/temp_candidate_v24.csv
- EC: 14회차 EC(EC/output_23/ec_b5.csv) + HR5 TabPFN 하루 수준 보정(EC/hr5_post.py, 학습 기본 예측 EC/hr5_train_base_oof.csv) -> EC/output_24/ec_hr5.csv
- 결합: merge_temp_ec_v24.py -> submission_24.csv / 규정 검사·전 과정 재현: checks_v24.py <원본 CSV 폴더> -> checks_v24.log (단계마다 별도 프로세스)
- 자세한 내용: 설명자료_24_V3GRU온도_TabPFN하루수준EC_가설검정.md

# submission_23 (2026-10-13, 집 클로드, 가설 검정)
- 온도: 13회차 온도 + TNA8 앵커(v15R 짝 = HP2 짝 또는 v15R 확신 >= .8; 온도/linker_v15r/v15r_eval_pkg.py -> 온도/code/make_temp_v23_tna8.py) -> 온도/temp_candidate_v23.csv
- EC: 12회차 EC(EC/w2only_post.py) + B5 같은 날 고EC 배타 내림 .290(EC/b5_post.py) -> EC/output_23/ec_b5.csv
- 결합: merge_temp_ec_v23.py -> submission_23.csv / 규정 검사·전 과정 재현: checks_v23.py <원본 CSV 폴더> -> checks_v23.log
- 자세한 내용: 설명자료_23_일치앵커TNA8_고EC배타B5_가설검정.md

# submission_22 (2026-10-12, 집 클로드, 가설 검정)
- 온도: 11회차 온도 + 짝찾기기 v7 앵커, 측정 감쇠(온도/code/make_temp_v22_anchor.py) -> 온도/temp_candidate_v22.csv
- EC: 9회차 EC + EA2 짝찾기기 v7 앵커(EC/ea2_post.py) -> EC/output_21/ec_ea2.csv
- 결합: merge_temp_ec_v22.py -> submission_22.csv / 규정 검사: checks_v22.py <원본 CSV 폴더> (ALL PASS)
- 자세한 내용: 설명자료_22_측정감쇠앵커_v7EC앵커_가설검정.md

# submission_21 (2026-10-12, 집 클로드, 규정 정리판 + 짝찾기기 v7)
- 온도: 11회차 온도 + 짝찾기기 v7 앵커(온도/linker_v7) → 온도/temp_candidate_v21.csv
- EC: 9회차 EC + EA2 짝찾기기 v7 앵커(EC/ea2_post.py) → EC/output_21/ec_ea2.csv
- 결합: merge_temp_ec_v21.py → submission_21.csv / 규정 검사: checks_v21.py <원본 CSV 폴더> (ALL PASS)
- 자세한 내용: 설명자료_21_짝찾기v7_다른온실학습기록.md

# submission_20 (2026-10-11, 집 클로드, 규정 정리판)
- 온도: 11회차 온도 + 짝찾기기 v5 앵커(온도/linker_v5) → 온도/temp_candidate_v20.csv
- EC: 9회차 EC + EA1 짝찾기기 v5 앵커(EC/ea1_post.py) → EC/output_20/ec_ea1.csv
- 결합: merge_temp_ec_v20.py → submission_20.csv / 규정 검사: checks_v20.py <원본 CSV 폴더> (ALL PASS)
- 자세한 내용: 설명자료_20_짝찾기v5_규정정리판.md

# submission_19 (2026-10-11, 집 클로드, 가설 검정)
- 온도: 11회차 온도(temp_candidate_v18.csv) + 짝찾기 앵커(온도/linker → 온도/code/make_temp_v19_anchor.py) → 온도/temp_candidate_v19.csv
- EC: 9회차 EC + W2 구조 규칙 단독 (EC/w2only_post.py) → EC/output_19/ec_w2only.csv
- 결합: merge_temp_ec_v19.py → submission_19.csv / 규정 검사: checks_v19.py <원본 CSV 폴더> → checks_v19.log (ALL PASS)
- 자세한 내용: 설명자료_19_짝찾기앵커_EC구조규칙단독_가설검정.md

# submission_18 (2026-10-11, 집 클로드, 가설 검정)
- 온도: 9회차 W40G-S(온도/temp_candidate_v13.csv) + 사슬 인식 새벽 보정 → 온도/temp_candidate_v18.csv (온도/code/make_temp_v18_chain.py)
- EC: 9회차 EC 구성원(EC/output_14) + B4 + HG3 + W2 구조 규칙 → EC/output_18/ec_c18.csv (EC/c18post.py)
- 결합: merge_temp_ec_v18.py → submission_18.csv / 규정 검사: checks_v18.py → checks_v18.log (ALL PASS)
- 자세한 내용: 설명자료_18_사슬새벽보정_EC구조규칙_가설검정.md

# 팜모니 — 정형데이터 미션 재현 패키지 (submission_17)

**submission_17 = 온도 W40G-S + 날 단위 문턱 GRU 멤버 + EC 15.** 이 패키지는 submission_14 패키지(아래 원문)에 15·16·17 파일을 더한 것입니다.

| 열 | 내용 | 생성 | 산출 |
|---|---|---|---|
| sub_temp | 9회차 W40G-S + 0.1·g·h10·(GRU − W40G-S): 따뜻한 날(그날 실내 최저 ≥ 10℃ 근처)의 따뜻한 시각에만 GRU 10% | `온도/code/make_submission_v17_temp.py ../output` | `온도/temp_candidate_v17.csv` |
| sub_ec | EC 15: 9회차 EC 구성에서 일평균 EC ≥ 1.2 학습일 30일 제외 | `cd EC` → `python -B -u reproduce_15.py --output 새출력폴더` | `EC/output_15/submission_15.csv` |

결합: `python merge_temp_ec_v17.py 온도/temp_candidate_v17.csv EC/output_15/submission_15.csv 결합.csv` → `submission_17.csv`와 같아야 합니다.
규정 검사: `cd 온도/code` → `python temp_v17_checks.py`.
상세 설명: `설명자료_17_GRU날문턱_고EC제외_가설검정.md`.

| 파일 | SHA-256 |
|---|---|
| submission_17.csv | `7f02bcd7eb08b4c7d98e5f2130c0788dc6be7c929f65b00f08d63b2900f814de` |
| 온도/temp_candidate_v17.csv | `7a29721190e8049a6bc13a02fee11cdef1359c75ff8f708c9df4e8bb24f46e3d` |

(submission_16 안내는 `설명자료_16_NOH0_고EC제외_가설검정.md` 참조)

---

## (원문) submission_14 패키지 안내

제1회 전국 농과계 대학 농업 AI 경진대회, 정형데이터 미션.
최종 답안 `submission_14.csv` (1,440행, row_id·sub_temp·sub_ec)는 두 부분으로 만들어집니다.

| 열 | 담당 | 폴더 | 산출 파일 |
|---|---|---|---|
| sub_temp | 온도 모델 W40G-S (8회차 W40G(실제 0.5160)에서 Codex·TabPFN의 `day` 를 계절 지표로 교체, 리더보드 판정용) | `온도/` | `temp_candidate_v13.csv` 의 sub_temp |
| sub_ec | EC 모델 14 (계절 v2 + DP1 운영 순서 특징 + SG2 이웃 학습일 보정) | `EC/` | `submission_14.csv` 의 sub_ec (EC 전달본, sub_temp 빈칸) |

두 열은 `merge_temp_ec_v14.py` 가 row_id로 결합합니다(ID 유일성·같은 집합·1,440행·유한값 검사,
온도 파일 행 순서 = sample_submission 순서, UTF-8, 소수 6자리).

## 재현 절차

1. **온도**: `온도/README.md` 2절. 대회 원본 CSV 3개를 `온도/data/` 에 넣고
   ```bash
   cd 온도/code
   python make_submission_v13_temp.py ../output   # → 온도/output/temp_candidate_v13.csv (CPU 약 20분)
   python temp_v13_checks.py                       # 규정 검사 7개 ALL PASS
   ```
2. **EC**: `EC/설명자료_14_EC_SG2.md`. EC 패키지는 필요한 입력·체크포인트를 동봉합니다.
   ```bash
   cd EC
   python -B -u reproduce.py --output 새출력폴더    # → EC 전달본 (EC/submission_14.csv 와 같음, EC/manifest.json)
   ```
3. **결합**
   ```bash
   python merge_temp_ec_v14.py 온도/output/temp_candidate_v13.csv EC/submission_14.csv 결합_새이름.csv
   ```
   결과가 동봉한 `submission_14.csv` 와 같아야 합니다.

## 해시 (SHA-256)

| 파일 | SHA-256 |
|---|---|
| submission_14.csv | `f2fe5cf7e5299c2c0f2ac39f0e071066744b0c75ea19e1d7dff1e3a6282ba18a` |
| 온도/temp_candidate_v13.csv | `c899e563f4395802b4657d3a9a30ff0a6a52a4b500a59534e45c8886dbf2050f` |
| EC/submission_14.csv (EC 전달본) | `c3798821c938238350b1e5d5f552633815911da622a1113dc1bfe5dbee1b72a8` |

## 환경 차이 주의

두 패키지의 검증 환경이 다릅니다. 온도는 `온도/requirements.txt`, EC는 `EC/requirements.txt`(NumPy 2.5.3 고정 등)를 각각의
가상환경에 설치해 실행하세요. 공개 사전학습 모델 TabPFN v2 회귀(Prior Labs)를 두 모델 모두 사용하며, 외부 데이터·원격 추론은
사용하지 않습니다. 생성형 AI 사용 내역은 각 폴더 문서에 있습니다(Anthropic Claude: 온도, OpenAI Codex: EC).
