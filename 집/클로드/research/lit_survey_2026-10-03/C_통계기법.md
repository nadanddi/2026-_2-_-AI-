# C. 통계·데이터 분석 기법 조사 (2026-10-03, 집/클로드 조사 하위 에이전트)

> 범위: 농업 지식이 아니라 통계·시계열·ML 방법론. 우리 문제(F13·F47 시간별 EC·배지온도, 하루 수준 오차 지배, 고EC 이봉 상태, 후반 평가 구간의 공변량 이동)에 맞춰 정리.
>
> **확인 수준 표기**
> - [본문 확인] = 웹 페이지 본문을 가져와 읽음 (요약 도구를 거쳤으므로 세부 수치는 원문 재확인 권장)
> - [초록·요약만] = 검색 결과의 초록/요약 문장만 봄. 본문 주장·실험은 확인 안 함
> - [목록만] = 서지·제목 수준, 내용은 2차 인용
> - (일반 지식) = 출처를 이번에 읽지 않고 교과서 수준 지식으로 쓴 내용
>
> **이미 실패한 것(반복 제안 안 함):** 칼만/혼합효과(GPBoost), 직전 정답 수준 특징, 앞뒤 정답 보간, 다일 이력·날씨 특징, 평균 수축, 선형 재보정, 분위수 회귀, 고EC 분류+전문가 혼합, 표본 가중, TabPFN 비중, NNLS 스태킹, 출처 군집 특징, 0시 지문 출처 식별, 리더보드 기반 비중.

---

## 0. 출처 목록

| # | 서지 | URL | 확인 수준 |
|---|---|---|---|
| S1 | Gneiting, T. (2011). Making and evaluating point forecasts. *JASA* 106, 746–762. | https://www.bundesbank.de/resource/blob/635562/7d3de0f3fc003e5b4864828143f268cf/mL/2012-06-01-eltville-11-gneiting-paper-data.pdf | [초록·요약만] |
| S2 | Bishop, C. M. (1994). Mixture density networks. Aston Univ. tech. report. | https://research.aston.ac.uk/en/publications/mixture-density-networks/ | [초록·요약만] |
| S3 | Jacobs, Jordan, Nowlan, Hinton (1991). Adaptive mixtures of local experts. *Neural Computation* 3, 79–87. / Jordan & Jacobs (1994). Hierarchical mixtures of experts and the EM algorithm. *Neural Computation* 6, 181–214. | https://www.semanticscholar.org/paper/c8d90974c3f3b40fa05e322df2905fc16204aa56 , https://apps.dtic.mil/sti/tr/pdf/ADA276516.pdf | [초록·요약만] |
| S4 | Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57, 357–384. | https://www.econometricsociety.org/publications/econometrica/1989/03/01/new-approach-economic-analysis-nonstationary-time-series-and | [초록·요약만] |
| S5 | Hughes & Guttorp (1994). A class of stochastic models for relating synoptic atmospheric patterns to regional hydrologic phenomena. *WRR* 30, 1535–1546. (비균질 HMM, NHMM) | https://academic.oup.com/jrsssc/article/48/1/15/6990642 (후속 논문 Hughes, Guttorp, Charles 1999) | [목록만] (2차 인용으로 개념 확인) |
| S6 | Athanasopoulos, Hyndman, Kourentzes, Petropoulos (2017). Forecasting with temporal hierarchies. *EJOR* 262, 60–74. | https://ideas.repec.org/a/eee/ejores/v262y2017i1p60-74.html | [초록·요약만] |
| S7 | Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica* 46, 69–85. + Stata Blog (2015) "Fixed effects or random effects: The Mundlak approach" | https://blog.stata.com/2015/10/29/fixed-effects-or-random-effects-the-mundlak-approach/ | 블로그 [본문 확인], 논문 [초록·요약만] |
| S8 | Duan et al. (2020). NGBoost: Natural gradient boosting for probabilistic prediction. *ICML*. | https://proceedings.mlr.press/v119/duan20a.html | [초록·요약만] |
| S9 | Zadrozny & Elkan (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD*. | https://dl.acm.org/doi/10.1145/775047.775151 | [초록·요약만] |
| S10 | Hurdle(two-part) 모형 정리: Wikipedia "Hurdle model", Heiss (2022) 블로그 | https://en.wikipedia.org/wiki/Hurdle_model , https://www.andrewheiss.com/blog/2022/05/09/hurdle-lognormal-gaussian-brms/ | [초록·요약만] |
| S11 | Rothenhäusler, Meinshausen, Bühlmann, Peters (2021). Anchor regression: heterogeneous data meet causality. *JRSS-B* 83, 215–246. | https://people.math.ethz.ch/~buhlmann/publications/anchor-JRSSB.pdf | [초록·요약만] |
| S12 | Arjovsky et al. (2019) Invariant Risk Minimization / Rosenfeld et al. (2021) The risks of IRM. *ICLR*. | https://openreview.net/pdf?id=BbNIbVPJ-42 | [초록·요약만] |
| S13 | Shi, Li, Li (2019). Gradient boosting with piece-wise linear regression trees. *IJCAI*. (LightGBM `linear_tree`의 근거) | https://www.ijcai.org/proceedings/2019/0476.pdf | [초록·요약만] |
| S14 | Grinsztajn, Oyallon, Varoquaux (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS*. | https://www.researchgate.net/publication/362123616 | [초록·요약만] |
| S15 | Meyer & Pebesma (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods Ecol. Evol.* 12, 1620–1633. | https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13650 | [초록·요약만] (+S17에서 [본문 확인]) |
| S16 | Meyer, Reudenbach, Hengl, Katurji, Nauss (2018). Improving performance of spatio-temporal ML models using forward feature selection and target-oriented validation. *Env. Modelling & Software* 101, 1–9. | https://research.wur.nl/en/publications/improving-performance-of-spatio-temporal-machine-learning-models-/ | [초록·요약만] |
| S17 | Meyer et al. (2024). The CAST package for training and assessment of spatial prediction models in R. arXiv:2404.06978. | https://arxiv.org/html/2404.06978v1 | [본문 확인] |
| S18 | Ding et al. "The temporal overfitting problem with applications in wind power curve modeling" (arXiv:2012.01349, Technometrics 게재로 알려짐) | https://arxiv.org/html/2012.01349 | [본문 확인] |
| S19 | Kaufman, Rosset, Perlich (, Stitelman) (2011/2012). Leakage in data mining: formulation, detection, and avoidance. *KDD'11 / ACM TKDD*. | https://dblp.org/rec/conf/kdd/KaufmanRP11.html | [초록·요약만] |
| S20 | Strobl, Boulesteix, Zeileis, Hothorn (2007). Bias in random forest variable importance measures. *BMC Bioinformatics* 8:25. | https://www.mendeley.com/catalogue/ebcc93c9-a50e-364b-959c-e812f8c0018e/ | [초록·요약만] |
| S21 | Altmann et al. (2010). Permutation importance: a corrected feature importance measure. *Bioinformatics* 26, 1340. + Kaggle "Feature Selection with Null Importances"(ogrellier) + 한국어 정리 블로그(databreak, 2019) | https://academic.oup.com/bioinformatics/article/26/10/1340/193348 , https://databreak.netlify.app/2019-04-21-null_importance/ | 한국어 블로그 [본문 확인], 논문·Kaggle [초록·요약만] |
| S22 | Roberts et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40, 913–929. | https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881 | [초록·요약만] |
| S23 | Bergmeir, Hyndman, Koo (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *CSDA* 120, 70–83. | https://research.monash.edu/en/publications/a-note-on-the-validity-of-cross-validation-for-evaluating-autoreg/ | [초록·요약만] |
| S24 | Bates, Hastie, Tibshirani (2023/arXiv 2021). Cross-validation: what does it estimate and how well does it do it? *JASA*. | https://arxiv.org/abs/2104.00673 | [초록·요약만] |
| S25 | Cawley & Talbot (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR* 11, 2079–2107. | https://www.jmlr.org/papers/v11/cawley10a.html | [초록·요약만] |
| S26 | Varma & Simon (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 7:91. | https://pmc.ncbi.nlm.nih.gov/articles/PMC3994246/ (관련 정리) | [초록·요약만] |
| S27 | Tibshirani & Tibshirani (2009). A bias correction for the minimum error rate in cross-validation. *Ann. Appl. Stat.* | https://arxiv.org/abs/0908.2904 | [초록·요약만] (초록 페이지 열람) |
| S28 | Hansen, Lunde, Nason (2011). The model confidence set. *Econometrica* 79, 453–497. | https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA5771 | [초록·요약만] |

---

## 1. 그룹(하루) 단위 잠재 상태를 공변량으로 예측하는 기법

### 1.1 핵심 관점 (일반 지식 + S6, S7)
- 시간별 목표 y(d,h) = L(d) + s(d,h) 로 쪼개면, 오차의 87%(EC)·57%(온도)가 L(d) 쪽입니다. L(d)의 **유효 표본 수는 시간 수가 아니라 날 수(≈400~460)**이고, 출처가 둘이면 출처별로는 그 절반 안팎입니다. 시간 단위 학습은 같은 날 24행이 거의 같은 L(d)를 공유하므로, 트리가 "그 날을 알아보는 특징"을 찾기만 하면 학습 손실이 크게 줄어듭니다. 이것이 이력 특징 +18% 악화와 같은 현상의 일반적 설명입니다(S18의 temporal overfitting과 같은 구조).
- 따라서 **일 수준 모형은 일 단위 행(날 수 = 표본 수)으로 학습**하는 편이 과적합 감시가 정직해집니다.

### 1.2 기법 표

| 기법 | 우리 문제 적용법 | 기대 효과 | 실패 위험 | 규정 주의 |
|---|---|---|---|---|
| **2단계(일 수준 + 시간 모양), temporal hierarchy** (S6) | ① 일 평균 목표를 날 단위 행(약 400행)으로 학습. ② 시간 모양 s(d,h)=y−L(d)는 별도 모형(시간·실내 입력). 예측 = L̂(d)+ŝ(d,h). S6는 여러 집계 수준 예측을 '조정(reconciliation)'해 결합 | 일 수준 오차를 직접 겨냥. 트리가 시간 행 24개 복제로 날을 외우는 길이 줄어듦 | 날 단위 400행은 작음 → 특징 수를 적게(10개 안팎) 유지해야 함. 시간 모양 모형이 수준 정보를 다시 흡수하면 이중 계산 | **하루 평균 입력은 그 날 미래 시각 입력을 포함** → 평가 시각 h에서는 0~h시까지 누적(expanding) 집계 또는 전날 집계만 사용해야 함. 날 단위 모형도 "h시까지의 입력"으로 학습·예측(시간마다 L̂ 갱신) |
| **Mundlak / within–between 분해** (S7) | 시간 단위 모형에 "그 날 지금까지의 입력 평균"(between)과 "현재값−그 평균"(within)을 함께 넣어, 날 간 차이와 날 안 변화를 분리 | 하루 수준 신호를 시간 행 모형에 명시적으로 주어, 트리가 다른 우회 특징으로 날을 식별할 유인 감소 | 누적 평균이 아침 시간엔 불안정(표본 1~5시간). 사실상 '다일 이력'과 겹치지 않게 **당일 범위만** 써야 함 | 당일 0~h시 누적만. 다음 날·미래 시각 금지 |
| **계층 베이지안(부분 풀링)** | 출처×상태별 일 수준 평균에 사전분포, 공변량 회귀 계수는 공유 | 고EC 날처럼 표본이 적은 칸의 μ 추정 안정 | 이미 실패한 혼합효과/날짜 랜덤효과와 본질이 같음. **평가 날의 랜덤효과는 0으로 수축**되므로 공변량이 설명 못 하는 날 수준은 못 맞힘 | 문제 없음 |
| **다중 과제 학습(EC·온도 공동)** | 일 수준 잠재 요인을 EC·온도가 공유한다고 보고 공동 모형 | 배지 '분리' 날과 고EC 날이 같은 잠재 원인이면 상호 정보 이득 | 두 목표의 날 수준 상관이 낮으면 이득 없음. 먼저 **학습 정답에서 일 평균 잔차 상관**을 재는 것이 싼 진단 | 상대 목표의 **평가 구간 정답은 없음** → 예측값끼리만 공유 가능 |
| **혼합분포 회귀 / mixture of experts / MDN** (S2, S3) | 게이트 p(x)=고EC 확률, 전문가 μ_k(x). MoE는 EM으로 게이트·전문가를 함께 학습(분류 따로·회귀 따로의 2단계와 다름) | 상태가 공변량으로 식별되면 μ_1(고EC 크기)을 별도로 학습 | **이미 '고EC 분류+전문가 혼합' 실패.** EM 공동 학습은 표본이 작으면 국소해·한 성분 붕괴. 상태 식별 정보가 입력에 없으면 어떤 구조도 조건부 평균 이상을 못 냄(2절) | 문제 없음 |
| **Markov switching / 은닉 마르코프(HMM), 비균질 HMM(NHMM)** (S4, S5) | 하루를 숨은 상태(출처 A/B × 정상/고EC)로 보고, 전이확률을 외기·구동기 공변량에 의존(NHMM, S5). 정답일에서 상태를 고정하고 평가 블록 안을 전이로 채움 | '번갈아 붙음'(출처 교대)과 '며칠 지속'(고EC 지속)을 구조로 표현할 수 있는 유일한 계열 | 칼만(연속 상태 공간)이 실패한 것과 같은 이유로 실패 가능: 평가 블록 5~10일 동안 정답 없이 전이만으로 상태를 끌고 가면 불확실성이 빠르게 정상분포로 수렴 → 결국 조건부 평균. 상태 수가 늘면 400일로 추정 불가 | 블록 **뒤쪽** 정답일까지 써서 평활(smoothing)하면 '미래 정보'로 볼 여지 → 앞쪽 정답일(필터링)만 쓰는 버전을 기본으로. 앞뒤 정답 보간은 이미 실패 |

### 1.3 언제 실패하나 (요약, 일반 지식)
- **공변량이 상태를 결정하지 않을 때**: 어떤 잠재 상태 모형도 E[y|x]보다 나아질 수 없습니다. 고EC가 출처·관리 결정 같은 입력 밖 요인으로 정해진다면 구조 개선의 상한은 낮습니다.
- **그룹 수가 작고 그룹 내 행이 많을 때**: 행 단위 손실로 학습하면 날 식별 특징에 과적합(S18, S17의 "Clever Hans").
- **상태 지속성을 쓰려면 이전 정답이 필요**: 평가 블록 안에는 정답이 없어서 지속성 이득이 블록 앞쪽 며칠에 몰립니다.

---

## 2. RMSE에서 "있는 줄은 알지만 크기를 모르는" 이봉 목표

### 2.1 이론 (S1 + 일반 지식)
- 제곱오차는 **조건부 평균**에 대해 일관(consistent)된 점수입니다(S1). 즉 RMSE 최적 예측은 언제나 E[y|x]이며, 이봉이면 두 봉우리 사이 값이 **정답**입니다. "중간값으로 얼버무림"은 정보가 모자랄 때 RMSE 관점의 올바른 행동입니다.
- 이진 상태 Z(고EC=1)에 대해 (일반 지식, 직접 유도):
  - E[y|x] = p(x)·μ₁(x) + (1−p(x))·μ₀(x)
  - MSE = E[Var(y|x,Z)] + E[ p(1−p)·(μ₁−μ₀)² ] + (모형 추정 오차)
  - 가운데 항은 **상태를 모르는 데서 오는 비가약 오차**입니다. μ₁≈3~4·μ₀이면 p가 0.5 근처인 날 하나가 RMSE를 지배합니다. 이 항을 줄이는 길은 **p를 0/1 쪽으로 더 날카롭게 만드는 새 정보**뿐이고, 손실 함수 설계로는 줄지 않습니다.
- **AUC .875는 순위 지표**라서 p의 보정(calibration)을 말해 주지 않습니다. 보정은 신뢰도 도표·Brier 점수로 따로 확인해야 합니다(S9의 isotonic 보정은 순위를 보존하며 확률만 맞춤).
- **딱딱한 선택(argmax 전문가)** 은 p가 0/1에 가깝지 않으면 RMSE를 키웁니다. 소프트 가중(확률 가중 평균)만 RMSE와 일치합니다.

### 2.2 기법 표

| 기법 | 우리 문제 적용법 | 기대 효과 | 실패 위험 | 규정 주의 |
|---|---|---|---|---|
| **오차 분해 진단(오라클 상태)** | 검증에서 (a) 실제 상태 Z를 주고 μ̂_Z만 쓴 예측, (b) p̂ 대신 보정된 p̃, (c) 현행 — 세 RMSE 비교. (a)−현행 = 상태 정보의 가치, (b)−현행 = 보정 이득 | **새 모형 전에 어디가 막혔는지 숫자로**: 상태를 알면 0.18→? 인지, μ₁ 크기 자체가 틀렸는지 구분 | 진단일 뿐 개선은 아님 | 오라클은 검증 분석에만. 제출 경로에 섞지 않음 |
| **확률 가중 예측 + isotonic 보정** (S9, S10) | p̂를 블록 CV 바깥 접기에서 isotonic 보정 → E[y]=p̃μ̂₁+(1−p̃)μ̂₀ | 이미 실패한 혼합과 다른 점은 **보정 단계 하나**. p̂가 과신/과소신이면 이득 | 400일·고EC 날 수십 일이면 isotonic이 계단 과적합 → Platt(로지스틱) 쪽이 안전. 이미 실패한 안의 변형이라 기대 낮음 | 보정용 자료는 학습 정답만 |
| **분포 회귀(NGBoost, MDN)** (S8, S2) | y|x를 2성분 혼합으로 직접 적합, 점예측은 평균 | 분산·혼합 비율을 함께 추정 | **점예측이 평균이면 결국 조건부 평균 회귀와 같은 목표** → RMSE 이득은 '추정 효율'에서만. 표본 작으면 불안정 | 없음 |
| **로그 변환 목표 + 편향 보정** (일반 지식) | log(EC) 학습 후 exp(m+σ²/2) | 배수형(3~4배) 차이를 덧셈형으로 바꿔 트리 분할이 쉬워짐 | 보정 없이 exp(m)이면 **중앙값**을 예측해 RMSE 손해. σ²가 상태별로 다르면 보정 자체가 틀림 | 없음 |

---

## 3. 공변량 이동·외삽 (test 입력 통계를 쓰지 않는 방법)

| 기법 | 우리 문제 적용법 | 기대 효과 | 실패 위험 | 규정 주의 |
|---|---|---|---|---|
| **Anchor regression** (S11) | '환경' 변수(예: 학습 전반/후반, 계절 위치 구간, 출처)를 anchor A로 두고, 잔차가 A로 설명되는 부분을 벌점(γ). γ=1 → OLS, γ→∞ → A 부분 제거. 선형이므로 **트리 앞단 기준선 또는 잔차 선형층**으로 사용 | A 방향으로 분포가 이동할 때 최악 위험을 줄이는 이론적 보장(선형 가정 하). 평가 구간이 "더 추운 후반"인 이동이 A 방향과 비슷하면 이득 | 이동이 A가 포착하지 못한 방향이면 효과 없음. 선형 가정. γ 선택이 또 하나의 다중비교 | A는 학습 자료로만 정의(test 통계 미사용) — 규정상 문제 없음 |
| **IRM** (S12) | 기간별 환경을 두고 불변 표현 학습 | 개념상 맞음 | S12: 벌점이 작아도 불변이 아닌 반례, 소표본·비선형에서 ERM보다 낫다는 근거 약함 → **우선순위 낮음** | — |
| **트리 대 선형 외삽** (S13, S14) | 트리는 학습 범위 밖에서 **상수**(가장자리 잎 값)로 예측. LightGBM `linear_tree=True`(S13 근거)나 "선형 기준선 + 트리 잔차"로 외기온·실내온의 1차 추세만 외삽 | 더 추운 평가 구간에서 온도 수준 편향 감소 가능 | 선형 외삽은 **틀린 방향이면 무한히 틀림**. linear_tree는 잎마다 선형이라 소표본 잎에서 기울기가 튐 → 잎 최소 표본·L2 강하게 | 없음 |
| **단조 제약** (LightGBM/XGBoost `monotone_constraints`, 일반 지식) | 물리적으로 방향이 분명한 입력(예: 배지온도 ↔ 실내기온 +)에만 + 제약 | 외삽 구간에서 엉뚱한 반전 방지, 분산 감소 | 실제 관계가 상태에 따라 반전(배지 '분리' 날)하면 제약이 오히려 해 | 없음 |
| **적용 영역(AOA)·비유사도 지수 DI** (S15, S17) | 학습 행 사이 거리(중요도 가중·정규화 특징 공간)로 DI 임계값을 **학습 CV에서** 정하고, 평가 행마다 그 행 입력으로 DI 계산 → 임계 초과면 미리 정한 보수 예측(예: 선형 기준선/수축)으로 대체 | 범위 밖 행만 골라 보호. 전체 성능을 해치지 않고 꼬리 위험만 줄임 | 대체 규칙이 실제로 더 나은지는 검증에서 '범위 밖 비슷한' 접기가 있어야 확인 가능. 임계·대체 규칙을 **실행 전 고정**해야 함 | 평가 행 DI는 **그 행의 현재 입력만** 사용(다른 평가 행 통계 미사용)이면 규정 내. test 전체 분포로 임계를 정하면 안 됨 |
| **구간별 정규화** | 입력을 "같은 온실의 과거 n일 입력 기준"으로 정규화(롤링 z-점수) | 수준 이동을 상대값으로 바꿔 불변성 확보 | 정규화 창이 곧 '다일 이력' → **날 식별 특징이 될 위험**(이미 +18% 악화 사례). 창이 길고 매끄러울수록 위험 | 과거 입력만 사용(규정 내). test 기간 전체 평균으로 정규화 금지 |

---

## 4. 식별자형 특징(날을 구별하는 매끄러운 특징) 진단·방지

### 4.1 문헌이 말하는 것
- S18(본문 확인): 시간 자기상관이 있는 자료에서 무작위 CV는 "같은 시기" 성능만 재서 낙관적이고, 다른 시기에 대해 크게 악화(풍력 사례: 무작위 CV에서 30% 우세 → 다음 해 5% 열세). 처방은 데이터를 시간 bin으로 **솎아(thinning)** 서로 떨어진 bin끼리로 학습하는 것.
- S16, S17(본문 확인): 위치·시간을 나타내는 예측변수가 "Clever Hans" 효과를 일으킴. **목표 지향 CV(Leave-Time-Out 등)로 전진 특징 선택(ffs)** 하면 그런 변수가 자동으로 빠짐.
- S19: 누수를 특징 누수/표본 누수로 구분. 우리 경우는 '학습-검증 간 같은 날 공유'에 의한 **표본 누수형**에 가까움.
- S20, S21: 트리 분할 기반 중요도는 값 종류가 많은(연속·매끄러운) 특징을 과대평가. **목표 순열 null 중요도**로 보정(S21 한국어 블로그 본문 확인: 목표를 섞어 80회 학습해 null 분포를 만들고 실제 중요도와 비교).

### 4.2 기법 표

| 기법 | 우리 문제 적용법 | 기대 효과 | 실패 위험 | 규정 주의 |
|---|---|---|---|---|
| **블록 단위 null 중요도** (S21 변형, 본 조사자 제안) | 시간 행이 아니라 **날 단위로 목표를 섞음**(날의 24시간 벡터를 통째로 다른 날에 붙임). 섞은 목표로도 중요도가 높은 특징 = 날을 구별하는 능력만으로 중요해지는 특징 | 행 단위 순열은 날 구조를 깨서 null이 너무 약함 → 날 단위가 우리 문제에 맞는 null | 계산량(×수십 회). 계절 위치처럼 **실제로 유용하면서 매끄러운** 특징도 null 중요도가 높게 나올 수 있음 → 제거 기준은 "실제/null 비율"로 | 없음 |
| **날 식별 가능성 검사** (본 조사자 제안, 일반 지식 조합) | 특징만으로 "어느 날인지(날 번호)"를 맞히는 회귀/분류기를 학습해 정확도를 재고, 특징별 기여를 봄. 높을수록 식별자형 | 새 특징을 넣기 **전에** 싸게 위험 점수화 | 정답과 무관한 진단이라 '유용하면서 식별 가능'한 특징을 구분 못 함 → null 중요도와 같이 봐야 | 없음 |
| **LTO-CV 전진 선택(ffs)** (S16, S17) | 블록 CV(평가 배치 모사)로 특징을 하나씩 추가, 개선 없으면 중단 | 식별자형 특징이 자동 탈락 | 선택 자체가 다중비교 → 선택 결과는 **바깥 접기**로 따로 검증(5절 nested) | 없음 |
| **특징 규제** (일반 지식) | 식별 위험 특징에 `feature_fraction`·`min_data_in_leaf`를 날 수 기준으로(예: 잎 최소 = 2~3일분 행) 설정, 또는 해당 특징을 거친 단계(분위수 bin 4~8개)로 양자화 | 매끄러운 특징의 세밀한 분할(=날 구별)을 막음 | 너무 거칠면 계절 위치 같은 유용한 정보도 잃음 | 없음 |
| **시간 솎아내기 학습** (S18) | 학습 날을 bin으로 나눠 인접 날이 같은 접기/잎에 몰리지 않게(예: bagging 단위를 '날 블록'으로) | 날 간 자기상관에 의한 낙관 감소 | 표본 손실 | 없음 |

---

## 5. 하루 수준 오차가 지배할 때의 검증 설계와 소표본 모델 선택

| 기법 | 우리 문제 적용법 | 기대 효과 | 실패 위험 | 규정 주의 |
|---|---|---|---|---|
| **블록 CV(평가 배치 모사)** (S22, S23) | 접기 단위 = 5~10일 블록, 앞뒤 2일 간격(gap)을 두고 그 바깥에 정답일 존재 — 실제 평가 배치와 같은 기하. S22는 잔차에 상관이 안 보여도 블록 CV를 권함 | 검증-평가 간 낙관 차이 축소 | 후반(2차 구간) 블록이 적으면 접기 수가 적어 분산 큼. S23: 무작위 K-fold가 타당한 건 순수 AR 모형+무상관 오차 조건뿐 → 우리 경우 해당 안 됨 | 없음 |
| **유효 표본 = 블록 수** (일반 지식, S24) | 신뢰구간·p값을 시간 행이 아니라 블록/날 단위 차이로 계산(이미 DIAG10이 이 방향이면 유지) | 과신 방지 | S24: CV 분산의 통상 추정은 너무 작아 구간이 좁음 → **nested CV로 분산 추정** 제안 | 없음 |
| **nested CV / 선택 편향 보정** (S25, S26, S27) | 안쪽 루프 = 하이퍼파라미터·특징 선택, 바깥 루프 = 성능. 또는 Tibshirani 보정(S27)으로 "최솟값 CV 오차"의 낙관 편향 추정 | 여러 후보 중 최소 검증오차를 고르는 **승자의 저주** 보정 | 계산량. 소표본에선 보정이 위쪽 편향을 만들기도(S27 요약) | 없음 |
| **Model Confidence Set** (S28) | 블록별 손실 차이로 "최선을 포함하는 후보 집합"을 구함. 집합에 여러 안이 남으면 **차이를 주장하지 않음** | 본페로니보다 덜 보수적이면서 다중비교를 다룸. "구분 불가" 결론을 공식화 | 블록 수 작으면 집합이 커서 결론이 안 남(그 자체가 정보) | 없음 |
| **사전 고정 규칙 유지** (S25 요지) | 현행 "모든 시드×검증기 같은 방향 + DIAG10 p<0.025(본페로니)"는 S25가 경고하는 '모델 선택 과적합'에 대한 합리적 방어 | — | 검증기 자체가 평가 분포를 못 닮으면 무력 → 3절 AOA로 '평가와 닮은 블록' 가중 검토 | 없음 |

---

## 6. 우리 문제에 바로 시험할 우선순위 (3~5개)

> 기대 효과는 정성적 추정이며 실측 아님. 이미 실패한 목록과 겹치지 않도록 골랐고, 겹칠 소지가 있는 건 차이를 명시함.

| 순위 | 제안 | 목표 | 왜 지금 | 기대 효과 | 주요 위험 | 규정 주의 |
|---|---|---|---|---|---|---|
| **1** | **오라클 상태 오차 분해 + p 보정 진단** (2.2) | EC | 고EC가 오차 46%인데, 막힌 곳이 "상태를 모름"인지 "크기 μ₁을 모름"인지 "p 보정"인지 아직 숫자로 분리 안 됨. 이 결과에 따라 아래 2~4의 우선순위가 바뀜 | 개선은 0, 대신 상한·방향을 줌. 상태를 알 때도 크게 틀리면 μ₁ 모형 문제, 상태만 알면 0.06대라면 **정보 탐색(입력 밖 단서)이 유일한 길** | 없음(진단) | 오라클은 분석 전용 |
| **2** | **2단계 일 수준 모형(temporal hierarchy) — 당일 누적 집계 특징, 날 단위 행 학습** (1.2) | EC·온도 | 오차의 87%/57%가 일 수준. 시간 행 학습이 날 식별 과적합을 부르는 구조를 직접 끊음 | 중간. 일 수준 RMSE 감소 가능, 특히 온도 '분리' 날의 수준 | 400행 소표본 → 특징 소수 고정. 시간 모양 모형의 이중 계산 | 당일 0~h시 누적만(미래 시각 금지) |
| **3** | **식별자형 특징 감사: 날 단위 null 중요도 + 날 식별 가능성 + LTO-CV 전진 선택** (4.2) | EC·온도 | 이력 특징 +18% 악화의 원인을 일반화해 현 특징 집합에서 잠재 식별자를 걸러냄. '계절 위치' 성공처럼 **특징을 덜어내거나 바꾸는 쪽**이 지금까지 유일한 성공 방향 | 중간~작음. 분포 이동 구간에서 강건성 | 유용한 매끄러운 특징까지 제거. 선택 다중비교 → 바깥 접기 확인 | 없음 |
| **4** | **외삽 보호: 선형 기준선(anchor 또는 OLS) + 트리 잔차, 단조 제약, DI 기반 대체** (3) | 온도 우선 | 평가 구간이 더 추움 → 트리 상수 외삽의 수준 편향 가능성. test 통계 없이 행별로 적용 가능 | 작음~중간, 범위 밖 날에서만 | 선형 외삽 방향 오류. DI 임계·대체 규칙은 실행 전 고정 필요 | DI 임계는 학습 CV로만 |
| **5** | **NHMM(공변량 의존 전이) 상태 필터링** (1.2) | EC | 출처 교대·고EC 지속을 구조로 담는 유일한 계열. 단 칼만·직전정답 특징 실패를 고려해 **1번 진단에서 "상태를 알면 크게 좋아진다"가 나올 때만** | 불확실(조건부) | 블록 안 정답 부재로 정상분포로 수렴 → 조건부 평균과 같아짐. 상태 수 증가 시 추정 불가 | 앞쪽 정답일 필터링만 기본, 뒤쪽 평활은 규정 검토 후 |

### 채택 판단 관련 메모
- 2~4는 각자 여러 설정이 생기므로, 실행 전에 설정 수를 고정하고 본페로니 또는 MCS(S28)로 다중비교를 처리. 최솟값 선택 낙관은 S27/nested CV로 보고.
- 손실 함수를 바꾸는 류(분위수, 비대칭 손실)는 RMSE 평가에서 이론상 이득이 없음(S1) — 이미 분위수 회귀 실패와 일치.

---

## 7. 이 조사의 한계
- 대부분의 출처는 [초록·요약만] 수준이며, 본문을 읽은 것은 S7(블로그), S17, S18, S21(블로그) 네 개뿐입니다. 본문 확인도 요약 도구를 거쳤으므로 수치 인용 전 원문 재확인이 필요합니다.
- "블록 단위 null 중요도", "날 식별 가능성 검사"는 문헌 기법을 우리 문제에 맞게 조합한 **조사자 제안**이며, 그 형태 그대로 검증된 문헌은 찾지 못했습니다.
- 2.1의 오차 분해식은 교과서 수준 유도(일반 지식)입니다.
