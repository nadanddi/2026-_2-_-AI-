# 팜모니 — 정형데이터 미션 재현 패키지 (submission_12, 온도 후보 W40G-S)

제1회 전국 농과계 대학 농업 AI 경진대회, 정형데이터 미션.
문제설명서 8.2절이 요구하는 코드, 설정, 실행 안내, 자료 사용 내역을 담았습니다.
`temp_candidate_v13.csv` 는 아래 절차로 다시 만들어집니다. 최종 답안의 EC 열은 상위 폴더 EC/ 패키지에서 옵니다(상위 README 참조).

- **sub_temp**: W40G-S = 8회차 W40G(실제 0.5160)에서 Codex·TabPFN 구성원의 `day` 입력을 계절 지표로 교체.
  `make_submission_v13_temp.py` 가 생성(구성원 학습은 `make_submission_v8_temp.py` 와 같은 코드).
- **sub_ec**: 3회차 제출(submission_04, 실제 0.2055)과 같은 값. 동봉한
  `make_submission_v7.py` 가 같은 값을 재생성하고 자가 검사합니다.

---

## 1. 폴더 구성

```
├─ README.md
├─ requirements.txt
├─ temp_candidate_v13.csv       온도 답안 (sub_temp 사용, sub_ec 열은 3회차 값으로 최종 답안에는 쓰지 않음)
├─ config/manifest_12.json      입력 해시, 답안 해시, 피처 목록, 구성원·가중치·시드
├─ reference/                   submission_04.csv, submission_06.csv, submission_08.csv (자가 검사·비교용)
├─ data/                        원본 CSV 3개를 넣는 곳 (배포본은 비어 있음)
├─ output/                      실행 결과가 써지는 곳
└─ code/
   ├─ make_submission_v13_temp.py  ★ 온도 답안 생성 (W40G-S)
   ├─ temp_v13_checks.py           ★ 온도 파이프라인 전체 규정 검사
   ├─ season_ec/                   계절 지표 표 (EC 팀원 season.py 산출, 학습 외기만 사용)
   ├─ make_submission_v12_temp.py  8회차 W40G 생성기
   ├─ make_submission_v8_temp.py   6회차 G_C2 생성기 (구성원 정의 공유)
   ├─ make_submission_v7.py        EC 열 재생성(3회차와 동일) + 공용 함수
   ├─ temp_mask_v1.py              MASK 세계(학습 행 피처를 test 입력 없이 계산), Codex 구성원
   ├─ resid_reset_features.py      Codex 구성원·TabPFN 입력 피처 (자정 재시작 잔차 피처)
   ├─ train_flags_v6.py            학습 행 가중치 (학습 입력만 사용)
   ├─ features_v2.py, features_v4.py, fp_features.py, feat_new.py, feat_temp74.py, feat_lib.py
   ├─ harness.py, common.py, env.py, env_extra.py
   ├─ make_submission_v3.py, make_submission_v4.py, screen_v6.py, cold_v5.py, ...  (공용 정의)
   ├─ submissions/                 생성기가 읽는 기준 답안 (reference/ 와 같은 파일)
   └─ causality_*.py               기존 피처 단위 규정 검사
```

## 2. 실행 안내

### 2.1 환경
Python 3.12.4 (Windows 10, x64). **GPU 미사용**, 전 과정 CPU.
numpy 2.5.3 / pandas 3.0.1 / scikit-learn 1.9.1 / lightgbm 4.7.0 /
torch 2.14.0 (CPU) / tabpfn 9.0.0

### 2.2 설치
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2.3 사전학습 가중치 (TabPFN v2 회귀)
처음 실행할 때 tabpfn 이 Hugging Face `Prior-Labs/TabPFN-v2-reg` 에서
`tabpfn-v2-regressor.ckpt` 를 자동으로 내려받습니다(학습 단계의 모델 준비이며,
평가 자료를 외부로 보내지 않습니다). 사용한 파일의 SHA-256:
`2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`

### 2.4 원본 자료
`data/` 에 train_X.csv, train_y.csv, test_X.csv 를 넣습니다(또는 환경변수 AGRI_DATA).

### 2.5 실행
```bash
cd code
python make_submission_v7.py                 # EC 열 재생성 (output/submission_07.csv, 자가 검사 포함)
python make_submission_v13_temp.py ../output  # 온도 W40G-S (output/temp_candidate_v13.csv), CPU 약 20분
```
`output/temp_candidate_v13.csv` 가 동봉한 `temp_candidate_v13.csv` 와 같아야 합니다
(기대 해시: `config/manifest_12.json` 의 `output_sha256`, Windows CRLF 기준.
다른 OS 에서는 줄바꿈 차이로 해시가 다를 수 있으니 수치를 비교하세요).
`sub_ec` 열은 `output/submission_07.csv` 의 `sub_ec` 와 같습니다.

### 2.6 규정 검사
```bash
cd code
python temp_v13_checks.py     # 온도 파이프라인 전체 (특징 → 가중치 → 세 구성원 → 게이트)
python causality_test.py
python causality_new.py
python causality_v4.py
python causality_v7.py
```
모두 `ALL PASS` 여야 합니다. `temp_v13_checks.py` 는
[D] 두 번 실행 결과가 비트 단위로 같음, [M] 학습 행 피처가 test 입력과 무관(MASK),
[F] 온실별 두 절단 시점 이후의 test 입력을 10배 확대·잡음·33% 결측으로 바꿔도
절단 이전 예측이 비트 단위로 같음, [X] 다른 온실 입력을 바꿔도 해당 온실 예측이
같음을 확인합니다(TabPFN 은 표본 1개로 검사; 검사 대상 성질은 표본 수와 무관).

## 3. 모델과 설정

모델 이진 파일은 포함하지 않았습니다. 파이프라인이 결정론적이라 재학습하면 같은 값이
나옵니다(LightGBM deterministic, 고정 시드, TabPFN CPU·고정 시드).

### 3.1 배지 온도 (W40G-S)

```
g    = clip((현재 시각 실내온도 − 8) / 2, 0, 1)
예측 = 0.4 × 기준 + (0.2 + 0.4(1−g)) × Codex + 0.4 g × TabPFN
```
Codex·TabPFN 구성원은 `day` 대신 계절 지표(`code/season_ec/`)를 씁니다. 학습일은 training_day_to_season.csv,
평가일은 evaluation_day_to_season.csv(학습 변환표를 온실·일차로 보간, 평가 입력 미사용)에서 읽습니다.
따뜻한 행(실내 10℃ 이상)은 기준 0.4·Codex 0.2·TabPFN 0.4, 추운 행(8℃ 이하)은
기준 0.4·Codex 0.6·TabPFN 0 입니다. TabPFN 은 학습 범위 밖 추위에서 크게
틀리므로(추운 날 외삽 검증 편향 +1.8℃) 추운 행에서는 빼고, 추운 날 검증에서 가장
강한 Codex 구성원의 몫을 늘렸습니다.

| 구성원 | 내용 |
|---|---|
| 기준 | 5회차 온도 구성 그대로: 0.65×(선형 물리 기준선 + LightGBM(huber) 잔차) + 0.25×Ridge + 0.10×Nystroem, 피처 135개, 시드 7·101·2024 평균 |
| Codex | Ridge 물리 기준선 + LightGBM 잔차, 자정 재시작 잔차 피처 89개, 시드 726·727 평균 |
| TabPFN | TabPFN v2 회귀, CPU, n_estimators 4, 학습 행 2,000개 문맥(5회차 가중치 비례 추출), 표본 1~8 평균 (문맥 시드 = 모델 시드 = 표본 번호) |

학습 행 가중치: 복원 행과 잡음 낀 날을 0.2 (5회차와 같음, 9,600행 중 2,341행).

### 3.2 규정 준수 설계
- **학습 행 피처는 MASK 세계에서 계산**: test_X 입력을 모두 비운 상태로 만들어,
  학습된 모델이 평가 입력을 전혀 보지 않습니다.
- **평가 행 피처**는 같은 온실의 해당 시각과 이전 입력만 사용합니다.
- 게이트 g 는 해당 행의 현재 시각 실내온도만 사용합니다.

## 4. 자료 사용 내역

| 구분 | 사용 여부 | 내용 |
|---|---|---|
| 외부 데이터 | 사용하지 않음 | 대회 제공 CSV 3개만 사용 |
| 공개 사전학습 모델 | **사용함** | TabPFN v2 회귀 (Prior Labs). 이용조건: Apache-2.0 기반 라이선스, 출처 표기 조건. 활용: 학습 행 2,000개를 문맥으로 주고 평가 행을 예측하는 구성원(비중 최대 0.4) |
| 자체 생성 자료 | 사용하지 않음 | 합성 데이터 없음 |
| 생성형 AI | **사용함** | 아래 |

평가 예측 과정에서 외부 API·원격 추론을 쓰지 않고 test_X 를 외부로 보내지 않습니다.
TabPFN 은 로컬에서 실행합니다.

- **Anthropic Claude (Claude Code)**: 코드 작성, 탐색적 자료 분석, 실험 설계, 검증
  방법론 검토, 결과 해석. 유료 구독, 산출물 사용 권한은 이용자에게 있음.
- **OpenAI Codex**: 별도 분석 계열(Codex 구성원의 자정 재시작 잔차 피처와 모델 설계,
  교차검증). 유료 구독, 산출물 사용 권한은 이용자에게 있음.
- 예측값을 생성형 AI 로 만들지 않았고, 모든 수치는 동봉한 코드를 로컬에서 실행해 산출.

인용: Hollmann et al. (2025), "Accurate predictions on small data with a tabular
foundation model", Nature 637, 319–326 (TabPFN).

## 5. 검증 요약 (온도)

8회차 W40G(실제 0.5160)에서 Codex·TabPFN 구성원의 `day` 입력만 계절 지표로 바꿨습니다(카탈로그 6.92).
기록은 달력을 두 번 돌고(179일 이후 재시작) 평가일은 모두 2차 구간이라, `day` 를 그대로 쓰면
평가일을 학습 기간 끝보다 더 늦은 시기로 읽게 됩니다. EC 담당 팀원의 같은 변경은 EC 0.2055 → 0.1455였습니다.
탐색 결과(W40G 대비, 2차 구간 행): DIAG10 −1.35%, EXT10 −0.50%, EXT12 −1.06%.
전체 행: −0.03%, +0.14%, −0.40%. 시드 1조합 탐색이라 **리더보드 판정용 후보**입니다.
