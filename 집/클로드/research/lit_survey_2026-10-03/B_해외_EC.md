# B. 해외 문헌 조사 — 배지(근권) EC 예측·하루 수준 오차·센서 함의

- 작성: 2026-10-03, 집 클로드(문헌 조사 서브에이전트). 대회 데이터 값은 검색어에 넣지 않고 일반 개념어로만 검색함.
- 겹침 회피: `lit_survey_2026-10-01/B_해외논문.md`(배지 온도·근권 가온), `C_데이터셋_대회_실무.md`(AGC 데이터셋·GroSens 설치), `연구실/코덱스/research/ec_comprehensive_20260930_v1/`(EC 정의·온도보상·Hilhorst·GS3/TEROS/WET 매뉴얼·금실/설향 배액률·Depardieu 2016·Udagawa 1991 등)에 이미 있는 출처는 다시 쓰지 않음. 필요할 때 "기존 조사"로만 언급.
- 접근 방법: WebSearch → OpenAlex API(초록 복원) → 열리는 곳은 PDF 본문 추출(pypdf). ScienceDirect·Springer·Tandfonline·ASHS 본문은 막힘. MDPI는 `www.mdpi.com`이 403이지만 `mdpi-res.com` PDF 경로로 본문을 읽음. 중간 파일은 세션 scratchpad에만 둠.
- **확인 수준 표기**: [본문 확인] 본문(PDF·HTML)을 직접 읽음 / [초록만] 초록만 읽음 / [목록만·접근 실패] 제목·검색 요약만. 수치는 [본문 확인]인 것만 적음(초록에 있는 수치는 "초록 수치"라고 따로 표시).
- **출처 성격**: 심사 논문 / 비심사 예비본(preprint) / 공공기관 지침 / 업계(업체) 자료를 각 항목에 적음. 업계 자료는 토마토 암면 기준이 많아 딸기 코이어에 바로 옮길 수 없음.

---

## 0. 요약 (가장 쓸모 있는 5가지)

1. **"어둡고 다습한(수동적인) 날에는 배액이 0에 가깝고 배지 EC가 오른다 → 밝은 날에 바로잡힌다"**는 운영 규칙이 업계 자료에 명시되어 있음(E12, 업계·[본문 확인]). 밀폐 날·겨울에 고EC가 몰리고 며칠 지속되는 우리 관찰과 방향이 맞음. 같은 자료는 "겨울에 흡수가 너무 적으면 배지 EC가 **내려갈** 수도 있다"고도 써서, 방향은 운영 방식(급액 시작·종료 기준, 최소 급액)에 달려 있음. 대분 딸기 지침(E11)도 1~2월 저일조기에 배액 EC가 **떨어지는** 사례가 많다고 적음 → 방향은 일반 법칙이 아니라 그 온실의 급액 규칙에 따라 정해짐.
2. **급액 정보가 있어도 배지 EC 예측 오차는 0.07~0.08 dS/m 수준**(E01, 파프리카 암면, 배지 EC 3.3~5.1 dS/m, 과거 배지 EC까지 입력, [본문 확인]). 배액 EC·일 누적 급액량을 빼면 결정계수가 0.05 떨어졌고, 기후 변수 하나를 빼면 0.02~0.04 떨어짐. **기후만으로 배지 EC를 추정해 성능을 보고한 논문은 찾지 못함.** "EC 예측" 논문 대부분은 같은 센서의 bulk EC·함수율을 입력으로 씀(E03, E04).
3. **사람의 운영 개입이 계단형 수준 변화를 만든다**: 타이머 급액은 주 단위 등으로 수동 조정되어 "일정 기간 같은 일 급액량"이 이어지고(E09), 대분 지침은 **1주 평균으로 1주 단위 조정**(E11), 생육 단계별 급액 농도 단계(30→50→80→100%, E09; 1.0→1.2→1.0 dS/m, E10), 빗물 세척 급액(E09: 특정 이틀) 등. 우리 데이터의 "동별 상태가 며칠 지속되고 0시부터 높음"과 모양이 맞음. 맞다면 오차는 기후 함수가 아니라 **숨은 계단 상태**임.
4. **폐쇄·반폐쇄 급액에서는 흡수되지 않는 이온(Ca, SO₄, Na)이 쌓여 시즌 후반 EC가 단조 상승**(E05 [초록만], E10 [본문 확인]: 딸기 폐쇄형 양액재배에서 2월 이후 배지 EC 상승, 처리에 따라 2.1~3.4 dS/m까지). 우리의 "학습한 계절 지표가 크게 도움"(C6.157)과 기작이 맞음.
5. **하루 수준 편향 보정 문헌의 교훈**: 최근 관측으로 편향을 따라가는 칼만 필터·7일 이동평균 보정은 **체제 전환(regime change) 때 오차가 급변하는 것을 못 따라감**. 입력이 비슷했던 과거 사례의 관측을 쓰는 아날로그(analog) 방식이 이를 보완함(E22, 기상 예보 후처리, [초록만]). 팀의 "직전 정답 상태 특징이 일관되게 해로움"(6.151, C6.166)과 같은 방향의 근거. 혼합효과 모델(E23, E24)은 **정답이 있는 그룹에만** 효과를 줌 → 정답이 없는 새 날에는 고정효과로 돌아감(카탈로그 6.39와 일치).

---

## 1. 출처 목록

### 1-A. 배지(근권) EC 예측·추정 (질문 1)

**E01.** Moon T., Ahn T.I., Son J.E. (2018). Forecasting Root-Zone Electrical Conductivity of Nutrient Solutions in Closed-Loop Soilless Cultures via a Recurrent Neural Network Using Environmental and Cultivation Information. *Frontiers in Plant Science* 9:859. https://doi.org/10.3389/fpls.2018.00859 (PMC6021533) — 심사 논문. **[본문 확인]**(PMC 본문, WebFetch 요약 경유)
- 파프리카, 암면 슬래브, 폐쇄형 순환. 2014-10-15~12-31(77일), 10초 측정 → 시간 평균, 1,416점(학습 900·검증 396·시험 120). 배지 EC 센서는 FDR(CoCo 100B, 한국 미래센서).
- 입력 20개: 배지 EC(과거값, 자기회귀)·배지 함수율·배액탱크 EC·부피·일 누적 배액량·혼합탱크 EC·부피·혼합량·일 누적 급액량·급액 일사 적산 설정값·회당 급액량·CO₂·광·온도·습도·정식 후 일수·초장·마디 수. 24시간 입력 → 24시간 출력.
- 단층 LSTM(64): 검증 R² 0.92·RMSE 0.07 dS/m, 시험 R² 0.72·RMSE 0.08 dS/m. 배지 EC 범위 3.3~5.1 dS/m.
- 입력 제외 실험(시험 R²): 배액탱크 EC 제외 0.67, 일 누적 급액량 제외 0.67(각 −0.05), 온도 0.69, 습도 0.68, 광 0.68, CO₂ 0.70. 오전 6~12시 오차는 증산 증가 탓이라고 저자가 해석. 한계로 "짧은 기간, 한 재배 라인, 계절 영향을 덮으려면 긴 자료 필요"를 명시.

**E02.** Moon T., Ahn T.I., Son J.E. (2019). Long short-term memory for a model-free estimation of macronutrient ion concentrations of root-zone in closed-loop soilless cultures. *Plant Methods* 15. https://doi.org/10.1186/s13007-019-0443-7 — 심사 논문. **[초록만]**
- 같은 그룹. 근권 이온 농도를 환경·생육 입력 LSTM으로 추정. 초록 수치: 입력 간격 1시간, 시간 단계 168(1주)이 사용됨. 다층보다 단층, 짧은 간격이 나았다고 함. "누적된 환경 조건"의 영향을 LSTM이 해석한다는 관점.

**E03.** Sodini M., Cacini S., Navarro A., Traversari S., Massa D. (2024). Estimation of pore-water electrical conductivity in soilless tomatoes cultivation using an interpretable machine learning model. *Computers and Electronics in Agriculture*. https://doi.org/10.1016/j.compag.2024.108746 — 심사 논문. **[초록만]**
- 방울토마토, 암면, 폐쇄형. 저가 센서의 bulk EC → 공극수 EC 변환을 Hilhorst 식 대신 GAM·XGBoost로. Hilhorst 식은 고함수율·고유전율 배지에서 부적합하다고 확인. XGBoost가 최선, 특히 극단 EC에서. (입력은 센서 bulk EC → 기후 전용 추정이 아님.)

**E04.** Zhao J., Tian P., Sun J., Wang X., Deng C., Yang Y. (2025). A Predictive Method for Greenhouse Soil Pore Water Electrical Conductivity Based on Multi-Model Fusion and Variable Weight Combination. *Agronomy* 15(5):1180. https://doi.org/10.3390/agronomy15051180 — 심사 논문. **[본문 확인]**(방법·상관 분석 절)
- 장미 온실 토양, 2024년 1년, 8,482 시간 기록. 공극수 EC와의 상관: bulk EC 0.93, 함수율 −0.6, 토양온도 0.49, 기온 0.34, VPD·습도는 매우 약함. 입력으로 함수율·토양온도·bulk EC·기온을 사용 → R² 0.9778.
- 함의: 높은 성능은 같은 센서의 bulk EC를 입력으로 쓴 덕분. **기후 입력만으로는 상관이 약하다**는 반대 방향 근거로 쓸 만함.

**E05.** Carmassi G., Incrocci L., Maggini R., Malorgio F., Tognoni F., Pardossi A. (2005). Modeling Salinity Build-Up in Recirculating Nutrient Solution Culture. *Journal of Plant Nutrition* 28(3). https://doi.org/10.1081/PLN-200049163 — 심사 논문. **[초록만]**
- 물질수지 모델: 다량 양이온은 "흡수 농도(양분 흡수/물 흡수)"를 상수로 두고 물 흡수에 선형, Na는 비선형. 증산으로 줄어든 물을 완전 양액으로 채우는 폐쇄계에서 **흡수 농도가 급액 농도보다 낮은 이온이 쌓여 EC가 점진적으로 오름**. 누적 증발산량으로 "세척이 필요한 EC에 도달하는 시점"을 예측함. 여러 계절 실험으로 검증.

**E06.** Ahn T.I., Shin J.H., Son J.E. (2021). Theoretical and Experimental Analyses of Nutrient Control in Electrical Conductivity-Based Nutrient Recycling Soilless Culture System. *Frontiers in Plant Science* 12:656403. https://doi.org/10.3389/fpls.2021.656403 — 심사 논문. **[초록만]**
- 배지 안 용질·물 이동 + 자동 양액 조제 + 흡수를 묶은 통합 모델로 개방·반폐쇄·폐쇄계를 비교. 양분 변동은 "계 경계로의 양분 공급 함수"로 통합된다고 결론. 물질수지형 가상 센서의 틀.

**E07.** Ahn T.I., Yang J.-S., Park S.H., Moon H.W., Lee J.Y. (2020). Translation of Irrigation, Drainage, and Electrical Conductivity Data in a Soilless Culture System into Plant Growth Information… *Agronomy* 10(9):1306. https://doi.org/10.3390/agronomy10091306 — 심사 논문. **[초록만]**
- 급액·배액량·EC 자료로 "양분 흡수 지표"를 온라인 계산. 기존 지표(증산·급액·배액률·세척률·배액 EC)보다 수량과 상관이 높았다고 함. 역방향(EC 변화 → 흡수)으로 쓰는 물질수지 해석의 예.

**E08.** Heinen M., de Willigen P. — FUSSIM2(2차원 물 흐름·용질 이동·뿌리 흡수 모델), 암면 토마토 적용. ResearchGate 목록. **[목록만·접근 실패]**
- 검색 요약상 Richards 식 + 대류·분산 용질 이동. 물리 모델로 슬래브 EC를 풀려면 급액량·급액 EC·배액이 경계조건으로 꼭 필요함(일반 지식).

### 1-B. 고형배지 딸기·배지 EC의 계절·기후·운영 의존 (질문 2)

**E09.** Bonelli L., Montesano F.F., D'Imperio M., Gonnella M., Boari A., Leoni B. (2024). Sensor-Based Fertigation Management Enhances Resource Utilization and Crop Performance in Soilless Strawberry Cultivation. *Agronomy* 14(3):465. https://doi.org/10.3390/agronomy14030465 — 심사 논문. **[본문 확인]**(mdpi-res PDF)
- 딸기 'Sabrosa', 개방형 자유배액, 겨울~봄(처리 2022-02-09~05-16). 실내 일평균 기온 6~24℃, DLI 0.58~19(평균 6) mol/m²/d. TEROS 12(FDR, 15분 주기)로 함수율·EC·온도, bulk EC → Hilhorst 식으로 공극수 EC.
- 급액 농도를 생육 단계별로 원액의 30%(정식 후 1~24일) → 50%(25~84일) → 80%(85~149일) → 100%(150~182일)로 **계단식** 변경(원액 EC 1.40 mS/cm).
- 타이머 처리: 하루 3~14회·회당 1분, 세척률 약 20%를 목표로 "주기적으로(대개 주 단위 이상) 수동 조정" → 조정 사이 기간은 **일 급액량이 고정**. 배액이 줄거나 없을 때 빗물 세척 급액을 4월 8일·27일 두 번 줌. 센서 처리: 함수율 0.45 아래면 급액, 공극수 EC가 설정값(1~1.2 dS/m) 위면 빗물로 세척 → EC가 떨어졌다가 양액 공급으로 다시 오르는 **톱니 모양**.
- 누적 급액은 겨울→봄 전환에 급증. 처리 기간 총 급액(양액+물) 타이머 22.4 vs 센서 16.6 L/식물체. 저자도 Hilhorst 공극수 EC는 함수율이 높고 일정할 때만 믿을 만하다고 씀.

**E10.** 直井昌彦(Naoi M.)·畠山昭嗣·岡村昭子·稲葉幸雄·植木正明 (2008). イチゴの閉鎖型養液栽培に適した培養液処方 (Nutrient Solution Formula for a Closed Hydroponic System for Strawberries). 栃木県農業試験場研究報告 63:59–68. https://www.pref.tochigi.lg.jp/g61/seika/documents/1241743284768.pdf — 공공기관 연구보고(일본어, 영문 요약). **[본문 확인]**
- 크립토모스 배지 딸기 폐쇄형 양액재배. 기존 처방(오오츠카 A)에서 **급액량이 늘어나는 2월 이후 배지 내 EC가 상승**, Ca·SO₄-S가 배지에 뚜렷이 축적. 한 처리는 생육 후반 배지 EC가 2.1~3.4 dS/m까지 상승, SO₄를 절반으로 줄인 처방에서는 2.0 dS/m 이하로 억제.
- 권장 급액 EC: 정식~정화방 개화 1.0 → 개화~1월 말(또는 2월 말) 1.2 → 2월(3월) 이후 1.0 dS/m(본문과 요약에 시점 표기가 조금 다름). → **급액 EC 자체가 계절 계단으로 바뀌는 운영**의 실례.

**E11.** 竹中智哉·山田晴夫 (2019 작성, 2020 개정 4판). イチゴ高設栽培における排液計測を活用したかん水・肥培管理マニュアル. 大分県産業科学技術センター·農林水産研究指導センター. https://www.pref.oita.jp/uploaded/attachment/2096158.pdf — 공공기관 지침(일본어). **[본문 확인]**
- 관리 기준: 배액 EC 0.3~0.6 mS/cm, 배액률 10~30%. 배액률×배액 EC 표로 급액량(±20~50%)·시비를 조정하되 **"1주 평균값으로 1주 단위 조정"**.
- 사가호노카 1주당 하루 흡수량(mL): 10월 122, 11월 98, 12월 74, 1월 83, 2월 91, 3월 183, 4월 250(추정). 12~2월이 최저, **2월부터 급증하고 날마다 크게 변동**.
- "1~2월 저일조기에 1~3번 과방 비대기에 배액 EC가 떨어지는 경우가 많다"(현지 사례 그림). 배액 EC는 낮에 변동이 심하므로 21시~다음 날 7시 측정을 권장. 휴대 측정기와 비교한 정확도는 대략 ±0.1 mS/cm.

**E12.** Beerens J. (Grodan), Veenman J. (Ridder). (연도 표기 없음, 파일 URL 기준 2022년 2월 게시 추정). Save energy in greenhouses without affecting plant activity. Grodan 기사 PDF. https://www.grodan.com/syssiteassets/downloads/downloads-en/grodan_article_save-energy-in-greenhouses_en.pdf — **업계 자료**(암면 토마토 중심). **[본문 확인]**
- "겨울에 낮 동안 흡수가 너무 적으면 배지 EC가 **내려갈** 수 있다(식물이 비료를 모두 흡수). 여름과 정반대." 2~3일 EC가 내려가면 급액보다 기후 전략을 먼저 보라고 권고.
- "**어둡고 외기 습도가 높은 날은 온실 안이 매우 수동적**이 되어 급액이 덜 필요하고, **배액이 적거나 0이어도 괜찮다. 배지 EC가 오를 수 있지만 더 밝은 날에 바로잡아야 한다.**"
- "배지 EC와 배액 EC는 특히 겨울에 같지 않다." 어두운 날(일사 <400 W/m², 광량 <800 J/day[원문 단위 그대로]) 배액이 1 L/m² 미만이면 배액 EC가 배지 EC보다 상당히 높음.
- 일사 수준별 허용 배지 EC(토마토, mS/cm): 200 W/m² → 8, 400 → 6, 600 → 5, 800 → 4, 1000 → 3.8. 즉 **어두운 계절에 더 높은 배지 EC를 일부러 허용**하는 관행.

**E13.** Grodan (연도 미상, 파일 날짜 2024-07). Steering and control of EC and WC in the slab under summer conditions (백서). https://www.grodan.com/syssiteassets/downloads/downloads-en/whitepapers-en/grodan-whitepaper-steering-and-control-of-ec-and-wc-in-the-slab-under-summer-conditions.pdf — **업계 자료**. **[본문 확인]**
- 급액 시작 약 200 W/m², 첫 배액 약 600 W/m², 종료 약 300 W/m²(외부 일사 기준).
- 24시간 배지 EC 변동 0.3~0.8 mS/cm는 정상, <0.3이면 급액 과다, ≥1.0이면 급액 부족 신호. 정상일 그림에서 **"일사가 가장 높을 때 배지 EC가 가장 낮다"**(급액·배액이 많은 시간).
- 급액 용량이 부족하면 오후에 함수율이 내려가고 EC가 오르며, 이를 늦은 종료·급액 EC 인하로 보상하다 보면 **"날마다 오르는 EC"**가 생긴다고 서술.

**E14.** Grodan 지식 페이지. Effect of early stop time and gift/radiation on EC behaviour. https://www.grodan.com/global/knowledge/root-zone-management/irrigation-and-nutrients/what-is-ec/effect-of-early-stop-time-and-giftradiation-on-ec-behaviour/ — **업계 자료**. **[본문 확인]**(웹페이지, WebFetch 요약 경유)
- 같은 급액 EC(3.2)에서 고정 종료(14:30)는 밝은 날 배지 EC가 통제 없이 오르고, 일사 기준 종료(외부 250~200 W/m² 남을 때)는 안정. 일사 기준이면 종료 시각이 밝은 날과 어두운 날 사이에 ±1.5시간 달라짐.

**E15.** Sonneveld C., Welles G.W.H. (1988). Yield and quality of rockwool-grown tomatoes as affected by variations in EC-value and climatic conditions. *Plant and Soil*. https://doi.org/10.1007/BF02182034 — 심사 논문. **[목록만·접근 실패]**(Springer 막힘, 초록 비공개)
- 검색 요약에 따르면 저광 조건에서는 높은 근권 EC의 수량 감소가 작았다고 함 → 겨울에 EC를 높게 운영하는 관행의 근거로 자주 인용됨. 본문 미확인이라 수치는 적지 않음.

**E16.** Lieten P. (2012). Advances in Strawberry Substrate Culture during the Last Twenty Years in the Netherlands and Belgium. *International Journal of Fruit Science*. https://doi.org/10.1080/15538362.2012.697024 — 심사 논문(리뷰). **[초록만]**
- 초록은 역사·품종 위주. EC 운영 수치는 초록에 없음. 본문(Tandfonline)은 막힘.

> 기존 조사에 이미 있는 딸기 배지 자료(Depardieu 2016 PLOS ONE, 금실 배액률 2022, 설향 일사비례 2022, 설향 배지 연수 2019, SRUC TN655)는 여기서 다시 쓰지 않음.

### 1-C. 급액을 기후로 추정·일사비례 급액의 흔적 (질문 3)

**E17.** Nikolaou G., Neocleous D., Katsoulas N., Kittas C. (2019). Irrigation of Greenhouse Crops. *Horticulturae* 5(1):7. https://doi.org/10.3390/horticulturae5010007 — 심사 논문(리뷰). **[본문 확인]**(mdpi-res PDF, 관련 절)
- 고형배지 급액은 보통 **일출 1시간 뒤 시작·일몰 1시간 전 종료**, 일사가 강하면 1시간 이하 간격. 피드백이 없는 온실은 고정 1회량에 빈도만 바꿔 시간으로만 자동화하는 경우가 흔함.
- 일사 적산 방식: 실내 일사 적산 0.4~0.6 MJ/m²(폐쇄형, 배액 30%)·1.4~1.8 MJ/m²(개방형, 15%)마다 급액(Schröder & Lieth 인용). 암면 자유배액은 0.8 MJ/m² + 최소 휴지시간(밝은 날 20분·어두운 날 50분, Lee 인용). 임계값은 작물 계수·재배 작업에 따라 자주 재평가 필요.

**E18.** Xiao L., Ma Y., Feng Q., Gao X., Shi H., Liu X., Yin Y. (2026). Radiation-Driven Prediction of Daily Irrigation Demand under Different Electrical Conductivity Scenarios in Greenhouse Tomato. *bioRxiv* 예비본. https://doi.org/10.64898/2026.01.23.701235 — **비심사 예비본**. **[본문 확인]**
- 중국 허베이 유리온실, 2023-11~2024-05, 고형배지 토마토. 일 급액량(I)과 일 적산 일사(G, PAR에서 환산)의 선형 관계: 처리별 R² 0.52~0.79.
- 5겹 시간 블록 교차검증 RMSE 0.815~1.393 L/일/조(평균의 약 18~21%), NSE 0.407~0.730. 급액 EC를 넣으면 한 처리에서만 ΔRMSE −0.014로 소폭 개선.
- 함의: 일사로 일 급액량의 50~80%를 설명하나, 나머지는 운영자 판단·처리 차이. 급액량을 일사로 "추정"하면 일 단위 오차가 20% 안팎 남음.

**E19.** Sim H.S., Jo J.S., Moon Y.H., Jung S.B., Lee T.Y., Shin H.R. 외 (2025). Development of a strawberry transpiration model based on a simplified Penman–Monteith model under different irrigation regimes. *Horticulture, Environment, and Biotechnology*. https://doi.org/10.1007/s13580-025-00677-z — 심사 논문. **[초록만]**
- 고형배지 딸기에서 일사·VPD·LAI를 넣은 수정 Penman–Monteith 증산 모델. 일사+VPD 보정형이 정확했다고 함(초록에 수치 없음). 충분/부족 관수 두 처리.

**E20.** Jing Z., Yang Y., Song J., Song C., Qian J., Shan G. 외 (2026). Simulation of Root Zone Soil Moisture Dynamics and Optimization of Irrigation Scheduling for Greenhouse Strawberries Based on HYDRUS-3D. *Horticulturae* 12(6):715. https://doi.org/10.3390/horticulturae12060715 — 심사 논문. **[초록만]**
- U자형 재배대 딸기의 근권 함수율을 HYDRUS-3D로 모사(초록 수치: R² ≥ 0.8302, RMSE ≤ 0.0309, NSE ≥ 0.5979). EC·용질은 다루지 않음. 물리 모델은 급액 입력이 있어야 함.

**(참고)** 기존 조사(코덱스 agronomy.md)에 설향 일사비례 150/200 J/cm² 원논문, 코이어 로드셀 원논문이 있음.

### 1-D. 하루 수준 오프셋·재귀 추정·자료 동화 (질문 4)

**E21.** Delle Monache L., Nipen T., Deng X., Zhou Y., Stull R. (2006). Ozone ensemble forecasts: 2. A Kalman filter predictor bias correction. *Journal of Geophysical Research: Atmospheres* 111. https://doi.org/10.1029/2005JD006311 — 심사 논문. **[초록만]**
- 최근 예보와 관측으로 칼만 필터가 "다음 편향"을 추정해 빼는 예측자 모드. 계통 오차는 칼만 필터로, 비계통 오차는 앙상블 평균으로 줄였다고 함.

**E22.** Delle Monache L., Nipen T., Liu Y., Roux G., Stull R. (2011). Kalman Filter and Analog Schemes to Postprocess Numerical Weather Predictions. *Monthly Weather Review* 139(11). https://doi.org/10.1175/2011MWR3653.1 — 심사 논문. **[초록만]**
- 시간 순서 칼만 필터(KF)와 7일 이동평균 보정은 **체제 전환 때 오차의 급변을 예측하지 못함**. 대신 "현재 예보와 닮은 과거 예보(아날로그) 10개가 맞았을 때의 관측 가중평균(AN)"과 아날로그 순서로 KF를 돌린 ANKF를 제안.
- 초록 수치: AN은 중심화 RMSE 기준으로 ANKF·KF·7일 보정·원시 예보 대비 각각 10·20·25·35% 개선(풍속, 400개 관측소, 6개월).

**E23.** Hajjem A., Bellavance F., Larocque D. (2014; 온라인 2012). Mixed-effects random forest for clustered data. *Journal of Statistical Computation and Simulation*. https://doi.org/10.1080/00949655.2012.741599 — 심사 논문. **[초록만]**
- 랜덤 포레스트 + 그룹 랜덤효과(EM). 랜덤효과가 무시할 수 없을 때 RF보다 크게 개선(모의실험). 새 그룹은 랜덤효과 0(고정부)으로 예측됨(일반 지식).

**E24.** Sigrist F. (2022). Gaussian Process Boosting. *Journal of Machine Learning Research* 23. http://jmlr.org/papers/volume23/20-322/20-322.pdf (GPBoost 라이브러리) — 심사 논문. **[목록만·접근 실패]**(검색 결과로 존재·내용 범위만 확인)
- 트리 부스팅 + 그룹 랜덤효과(중첩·교차)·가우스 과정. **팀은 이미 날짜 랜덤효과로 시도해 실패(카탈로그 6.39)**.

**E25.** Aljoumani B., Sánchez-Espigares J.A., Cañameras N., Josa R. (2014). An advanced process for evaluating a linear dielectric constant–bulk electrical conductivity model using a capacitance sensor in field conditions. *Hydrological Sciences Journal*. https://doi.org/10.1080/02626667.2014.932053 — 심사 논문. **[초록만]**
- 현장 정전용량 센서에 Hilhorst식 선형 관계를 적용하니 잔차에 강한 양의 자기상관. 오프셋을 **시간에 따라 변하는 동적 선형 모델(DLM) + 칼만 필터·평활**로 추정. 오프셋은 깊이마다 달랐고 원인으로 토양 온도를 추정.
- 함의: 센서 EC의 "느리게 움직이는 오프셋"을 상태로 두는 것은 문헌에 있는 기법. 단 관측(EC)을 계속 받아야 쓸 수 있음.

**E26.** Jiang Z., Huang Q., Li G., Li G. (2019). Parameters Estimation and Prediction of Water Movement and Solute Transport in Layered, Variably Saturated Soils Using the Ensemble Kalman Filter. *Water* 11(7):1520. https://doi.org/10.3390/w11071520 — 심사 논문. **[초록만]**
- HYDRUS-1D + 앙상블 칼만 필터(앙상블 50~100). 함수율보다 **염분 모사가 자료 동화로 크게 좋아짐**. 관측이 성긴 경우에도 정확한 상태 관측이 도움.

### 1-E. 출처 혼합·도메인 이동 (질문 5)

**E27.** Loer P., Daniels A., Fink M., García-Mañas F., Wollherr D., Rodríguez F. (2025). Transfer Learning for Time-series Forecasting of Greenhouse Microclimate. *Jornadas de Automática* 46. https://doi.org/10.17979/ja-cea.2025.46.12186 — 학회 논문. **[본문 확인]**(초록·서론)
- 877 m² 온실(81일)로 학습한 트랜스포머를 1,900 m² 온실(48일)로 옮기며 미세조정. 대상 온실 **2일치 자료만으로 충분**했다고 보고. 예측 대상은 기온·습도(EC 아님).

**E28.** Moon T., Son J.E. (2021). Knowledge transfer for adapting pre-trained deep neural models to predict different greenhouse environments based on a low quantity of data. *Computers and Electronics in Agriculture*. https://doi.org/10.1016/j.compag.2021.106136 — 심사 논문. **[목록만·접근 실패]**

**E29.** Sugiyama M., Krauledat M., Müller K.-R. (2007). Covariate Shift Adaptation by Importance Weighted Cross Validation. *Journal of Machine Learning Research* 8. — 심사 논문. **[초록만]**
- 학습·시험 입력 분포가 다르고(예: 학습 범위 밖 외삽) 조건부 분포는 같을 때, 보통 교차검증은 편향됨 → 중요도 가중 교차검증(IWCV) 제안. 비정상성이 강한 세션 간 이동에 적용 예.

**E30.** Truong C., Oudre L., Vayatis N. (2020; 온라인 2019). Selective review of offline change point detection methods. *Signal Processing* 167. https://doi.org/10.1016/j.sigpro.2019.107299 (arXiv 1801.00718) — 심사 논문(리뷰). **[목록만·접근 실패]**(서지만 확인)
- 오프라인 변화점 탐지 방법 정리(ruptures 라이브러리). 계단형 수준 변화를 찾는 데 쓸 수 있는 일반 도구라 목록에만 둠.

### 1-F. 센서(bulk vs 공극수, 온도·함수율) 함의 (질문 6)

> GS3·TEROS12·WET·WET150 매뉴얼, Hilhorst 2000 원저, Archie, Rhoades 1976, Alem 2011 ASHS 초록은 기존 조사(코덱스 sensor.md·agronomy.md)에 있음.

**E31.** Scoggins H.L., van Iersel M.W. (2006). In Situ Probes for Measurement of Electrical Conductivity of Soilless Substrates: Effects of Temperature and Substrate Moisture Content. *HortScience* 41(1):210–214. https://doi.org/10.21273/HORTSCI.41.1.210 — 심사 논문. **[초록만]**(ASHS PDF는 HTML로 막힘)
- 탐침 4종 비교. 공극수 EC를 재는 탐침(SigmaProbe, WET)은 **함수율이 오르면 EC가 내려가고**(희석), bulk EC 탐침(HI 76305, Field Scout)은 **함수율이 오르면 EC가 올라감**(물이 전류 통로). **초록 수치: 체적함수율 35% 이상에서는 함수율의 영향이 작음.** 한 탐침은 온도에 매우 민감했음.

**E32.** Bañón S., Álvarez S., Bañón D., Ortuño M.F., Sánchez-Blanco M.J. (2021). Assessment of soil salinity indexes using electrical conductivity sensors. *Scientia Horticulturae*. https://doi.org/10.1016/j.scienta.2021.110171 — 심사 논문. **[초록만]**
- 저울(실제 염 수지)과 센서를 동시에 써서 염도 지표를 평가. bulk EC는 함수율이 일정할 때만(높을수록 정확) 좋은 지표. Hilhorst 공극수 EC는 함수율이 높고 일정할 때만 믿을 만함. **현재 함수율로 보정하거나 두 급액 사이 평균을 내면** 이 조건이 필요 없어짐.

**E33.** Li Y.L., Stanghellini C., Challa H. (2001). Effect of electrical conductivity and transpiration on production of greenhouse tomato. *Scientia Horticulturae*. https://doi.org/10.1016/S0304-4238(00)00190-4 — 심사 논문. **[목록만·접근 실패]**(초록 비공개)

**E34.** Shin J.H., Son J.E. (2015). Changes in electrical conductivity and moisture content of substrate and their subsequent effects on transpiration rate, water use efficiency, and plant growth in the soilless culture of paprika. *Horticulture, Environment, and Biotechnology*. https://doi.org/10.1007/s13580-015-0154-6 — 심사 논문. **[목록만·접근 실패]**

---

## 2. 질문별 정리 (문헌이 말하는 것 / 말하지 않는 것)

**질문 1 (기후만으로 배지 EC 추정)**
- 찾은 EC 예측 연구는 모두 급액·배액 기록(E01, E07), 같은 센서의 bulk EC·함수율(E03, E04), 또는 과거 배지 EC(E01)를 입력으로 씀. **기후 데이터만으로 배지 EC를 추정해 성능을 보고한 연구는 찾지 못함**(검색 범위 안에서).
- 급액 정보까지 있는 E01도 시험 RMSE 0.08 dS/m(EC 3.3~5.1 dS/m 범위, 5일 시험). 작물·EC 범위가 달라 직접 비교는 할 수 없음. 다만 "기후 함수만으로 0.06 수준"은 문헌 기준으로도 어려운 목표로 보임(추정). 상위 팀의 차이는 기후 물리보다 **하루 수준 상태를 맞히는 다른 정보원**에서 나왔을 가능성이 큼(추정, 문헌이 직접 말하지 않음).

**질문 2 (딸기 배지 EC의 계절·기후 의존)**
- 같은 현상에 두 방향의 서술이 모두 있음: 어둡고 다습한 날 배액 0 → EC 상승(E12), 겨울 흡수 부족 → EC 하락(E12), 1~2월 저일조기 배액 EC 하락(E11), 2월 이후 비흡수 이온 축적으로 상승(E10). **방향은 그 온실의 급액 규칙(시작·종료 기준, 최소 급액, 급액 EC 계절 단계, 세척 여부)이 정함.**
- 운영 개입은 대부분 **주 단위·생육 단계 단위의 계단**(E09, E10, E11). 하루 안 기후 변화보다 긴 시간 척도.
- 흡수량은 12~2월이 최저이고 2월부터 급증(E11, 사가호노카). 겨울이 "급액·흡수가 모두 작아 작은 불균형도 며칠 쌓이는 시기"라는 해석과 맞음(추정).

**질문 3 (기후로 급액 추정·일사비례의 흔적)**
- 일사로 일 급액량의 약 50~80%를 설명(E18, 예비본). 급액은 일출 1시간 뒤~일몰 1시간 전, 일사 적산 임계값마다(E17), 업계는 외부 일사 약 200 W/m²에서 시작·300(250~200) W/m²에서 종료(E13, E14).
- 일사비례 급액이 잘 맞으면 "일사 최고일 때 EC 최저", 24시간 EC 변동 0.3~0.8 mS/cm(E13, 토마토 암면). 부족하면 **날마다 오르는 EC**(E13, E14).
- 난방으로 생긴 증산(VPD)은 일사비례 급액에 반영되지 않음 → 밀폐·난방·저일사 날에 급액이 증산보다 적어질 수 있다는 가설은 문헌의 직접 진술이 아니라 이 조사자의 추론(E12의 "어둡고 다습한 날 배액 0"과 E19의 VPD 항에서 유도).

**질문 4 (하루 수준 오프셋)**
- 문헌 기법: (a) 최근 관측으로 편향을 따라가는 칼만 필터(E21), (b) 입력이 닮은 과거 사례의 관측을 쓰는 아날로그(E22), (c) 그룹 랜덤효과(E23, E24), (d) 오프셋을 동적 상태로 두는 DLM/칼만(E25), (e) 물리 모델 + 앙상블 칼만 자료 동화(E26).
- (a)(c)(d)는 **같은 그룹의 최근 정답이 예측 시점에 있어야** 효과가 있음. E22는 시간 순서 보정이 체제 전환에서 실패한다고 명시 → 팀의 "직전 정답 특징 해로움"(6.151, C6.166)과 같은 방향. (b) 아날로그는 시간 이웃이 아니라 **입력 지문이 닮은 과거 날의 잔차**를 쓰므로 이 실패를 피할 수 있다고 주장됨.

**질문 5 (출처 혼합·도메인 이동)**
- 온실 간 이전은 소량 자료 미세조정으로 가능하다는 보고(E27, 기온·습도). 외삽 상황에서는 중요도 가중 검증이 필요(E29). 변화점 탐지(E30)는 계단 상태를 찾는 도구.
- 기록 안에 하루씩 번갈아 이어붙은 두 출처를 입력만으로 식별하는 문헌은 찾지 못함(검색 범위 안에서).

**질문 6 (센서)**
- bulk EC는 함수율이 오르면 올라가고 공극수 EC는 내려감. VWC 35% 이상에서는 함수율 영향이 작음(E31). bulk·Hilhorst 공극수 EC 모두 함수율이 높고 일정할 때만 염도 지표로 믿을 만함(E32, E09). Hilhorst식은 고함수율 배지에서 부적합(E03). 센서 오프셋은 시간·깊이에 따라 변하는 상태일 수 있음(E25).
- 함의: 목표값의 **하루 수준 오프셋 일부가 염 농도가 아니라 "그날의 함수율 수준" 또는 센서 오프셋일 가능성**. 하루 안 2%/℃ 동행은 코덱스 D13에서 새벽(1~5시, 급액 없는 시간) 1차 차분 기울기도 같게 나왔으므로 급액보다 온도 효과(미보상 또는 보상식 차이) 쪽과 더 맞음(기존 분석 + 이 조사자의 해석).

---

## 3. 우리 데이터로 확인할 수 있는 가설·기법

규정 요약: 평가 행 특징은 **같은 온실의 현재·이전 입력만**. 학습 정답은 특징으로 사용 가능. "이미 시험"란은 카탈로그에서 비슷한 것을 찾은 결과(완전히 같은 시험이라는 뜻은 아님).

| 가설/기법 | 맞다면 데이터에 보일 흔적(또는 기대 효과) | 보이면 안 되는 것 | 필요한 입력 | 대회 규정상 주의 · 이미 시험 |
|---|---|---|---|---|
| **H1. 어둡고 다습한 날 연속 → 배액 0 → EC 누적, 첫 밝은 날 회복**(E12, E14) | 하루 EC 수준이 "직전 k일의 저일사·환기 0 일수"에 따라 오르고, 일사 큰 날 다음 날에 떨어짐. 상승은 점진, 하락은 급함(비대칭) | 당일 일사가 같은데 직전 날씨 이력과 무관하게 고EC가 나옴. 밝은 날 연속인데 EC 계속 상승 | 외기 일사 일합, 환기 0 시간, 실내 RH(달력상 이전 날들) | 외기는 두 동이 공유하므로 달력상 이전 날 외기는 "같은 온실의 이전 입력"으로 쓸 수 있음(확인 필요). 실내·구동기 이력은 동이 바뀌어 섞임 주의. 비슷한 것: 6.38(당일 누적), 6.113(H17 7일 기억, −1.6~−1.8%이나 전칸 실패) |
| **H1′. 방향 판별: 겨울 흡수 부족 → EC 하락형인지 상승형인지**(E11, E12) | 상승형이면 저일사 연속 뒤 EC↑, 하락형이면 EC↓. 우리 관찰(밀폐 날 고EC)은 상승형 쪽 | 같은 동에서 저일사 연속 뒤 방향이 에피소드마다 뒤바뀜(→ 운영자 개입이 지배) | H1과 같음 | 진단만(특징 추가 전). 정답으로 집단을 나누면 평균 회귀 착시(6.148) → 입력 조건부로만 나눔 |
| **H2. 난방 증산 − 일사비례 급액 불일치**(E17, E19에서 추론) | 밀폐·난방 많고 일사 적은 날의 다음 날 EC가 높음. "난방 시간 × 실내 VPD ÷ 일사합" 같은 비율이 하루 수준 잔차와 같은 부호로 움직임 | 난방이 적은 밀폐 날에도 같은 크기로 고EC | 난방 가동률, 실내 온습도(VPD), 외기 일사 | 이전 날 실내·난방은 동 교체 문제(1.14: 같은 출처 전날은 대개 d−2). 비슷한 것: 6.38 TR(일사·VPD 누적 비율, 판별 불가), D14 heating×VPD(0~6시만, 실패). 다일(多日) 판은 미시험으로 보임 |
| **H3. 주 단위 수동 조정 → 계단형 수준**(E09, E11) | 학습 정답의 하루 수준에서 변화점이 7일 안팎 간격으로 나오고, 구간 안은 평평. 두 동의 변화점 시점이 서로 다를 수 있음 | 매끈한 연속 표류만 있음. 변화점이 날씨 급변일과 항상 겹침(→ H1 쪽) | 학습 정답(하루 평균), 기록 일차·추정 달력 | 진단은 학습 정답만으로 가능. 예측에 쓰려면 평가일이 어느 구간인지 입력으로 알아야 함 → 6.150(입력만으로 동 식별 불가)이 걸림돌 |
| **H4. 세척 급액 → 급락 후 점진 재상승(톱니)**(E09, E05) | 학습 정답에 "하루 만에 큰 하락 → 며칠 상승"이 비대칭으로 반복. 하락일은 날씨로 설명 안 됨 | 상승·하락이 대칭이고 날씨로 대부분 설명됨 | 학습 정답 | 진단용. 하락일을 입력으로 알 길이 없으면 예측 기법이 아니라 오차 해석에만 씀 |
| **H5. 비흡수 이온 축적 → 시즌 후반 단조 상승**(E05, E10) | 계절 지표가 후반에 단조 상승 모양을 학습. 2월 이후(흡수·급액 증가기) 기울기 증가. 양액 교체 시점에 리셋 | 후반에 수준이 평평하거나 하락 | 학습한 계절 지표(C6.157) | 이미 계절 지표로 반영됨(C6.157, 6.167). 추가로 "리셋(급락) 후 재축적" 모양이 있는지 H4와 함께 진단 |
| **H6. 하루 안 EC–배지온도 2%/℃는 센서 온도 효과**(E31, 기존 D13) | 급액이 없는 밤 시간대에도 같은 기울기(D13: 1~5시 0.0213). 기울기가 날의 EC 수준과 무관 | 기울기가 낮 급액 시간에만 있음, 일사 최고 시 EC 최저(E13형) | 시간별 배지온도(평가 입력에는 없음 — 예측 배지온도 사용 필요) | 이미 대부분 확인(6.130, D13). 온도 보정 모델은 6.28에서 기각. 평가 입력에 배지온도가 없으면 예측 온도의 오차가 EC로 전파됨 |
| **H7. 하루 수준의 일부는 함수율 수준(bulk EC) 또는 센서 오프셋**(E25, E31, E32) | 고EC 날에 하루 안 진폭·온도 기울기가 달라짐(함수율이 다르면 bulk EC의 온도·수분 감도가 바뀜). 고EC 사슬이 센서 교체·재설치일 근처에서 시작 | 하루 수준이 바뀌어도 하루 안 모양·기울기가 똑같음 | 학습 정답의 시간별 모양 | 진단용. 정답 모양 특징을 평가 행에 쓰면 규정 위반 아님(학습 정답)이나 평가일 자체 정답은 없음 |
| **H8. 아날로그 잔차 보정(AN)**(E22) | 평가일 입력 지문(0시까지·같은 온실)과 가장 닮은 학습일 K개의 OOF 하루 잔차 평균을 더하면 하루 수준 오차 감소. KF·직전 정답과 달리 체제 전환에 덜 민감 | 지문 거리가 먼 평가일(추운 외삽, 6.150의 거리 52~326)에서 개선 없음 또는 악화 | 학습일 OOF 잔차(학습 정답 사용), 입력 지문 | 학습 정답·OOF 잔차는 사용 가능. 단 기존 모델의 kNN·트리와 겹칠 수 있고, 6.81(군집×계절 잔차 보정 기각)·6.110과 유사 → 사전 고정 규칙으로 작은 실험만 |
| **H9. 혼합효과(날·사슬 랜덤효과)**(E23, E24) | 정답이 있는 그룹에만 이득. 새 날은 고정부 예측과 같아짐 | — | 그룹 ID | 6.39에서 이미 실패. 문헌상 기대와 일치 → 재시험 비권장 |
| **H10. 칼만·DLM 오프셋 추적**(E21, E25, E26) | 같은 동의 최근 정답이 1~2일 전이고 체제 전환이 드물 때만 개선 | 정답 간격이 멀거나 비어 있는 평가 배치에서 악화 | 같은 동 직전 정답 | 6.151(+35% 악화), C6.166(PAR1 악화)과 같은 계열 → E22가 말하는 실패 유형. 재시험 비권장 |
| **H11. 외삽 대비 중요도 가중**(E29) | 평가와 닮은(춥고 난방 많은) 학습일에 가중을 주면 추운 외삽 검증기(EXT)에서 개선, DIAG10은 비슷 | 모든 검증기에서 비슷하거나 악화 | 학습·평가 입력 분포(밀도비 추정) | 평가 입력의 분포를 쓰는 것은 정답을 쓰지 않으므로 대체로 허용으로 보이나 팀 규칙 확인 필요. 비슷한 것: 6.85(자정 ≤10℃ 학습일 가중 2배, 재시험 권장 상태) |
| **H12. 변화점 탐지로 동별 상태 구간 찾기**(E30) | 학습 정답에서 찾은 변화점이 입력 지문(구동기 설정 변화)의 변화점과 같은 날 | 입력 쪽 변화점과 무관 | 학습 정답, 0시 구동기 설정 | 입력 쪽 변화점만 평가에 쓸 수 있음. 6.150·6b.20과 겹침 → 낮은 우선순위 |

**우선순위 제안(이 조사자 의견)**: H1/H1′·H2는 문헌(업계 운영 규칙)과 우리 관찰이 가장 직접 맞고, 다일 버전은 미시험으로 보임. 다만 동 교체 문제 때문에 **외기(공유) 이력 → 실내·구동기 이력** 순서로 시험하는 것이 안전함. H3·H4·H7은 학습 정답만으로 하는 싼 진단이라 먼저 돌려 "숨은 계단 상태가 얼마나 지배적인가"를 숫자로 확인할 가치가 있음. H9·H10은 문헌과 팀 결과가 같은 방향으로 부정적.

---

## 4. 못 본 것 / 다음에 조사할 곳

| 대상 | 이유 |
|---|---|
| Sonneveld & Welles 1988(E15), Li·Stanghellini·Challa 2001(E33) 본문 | 저광기 고EC 운영의 원 실험 수치. Springer·Elsevier 막힘. WUR 리포지터리(E33은 green OA 표시) 재시도 가치 |
| Scoggins & van Iersel 2006(E31) 본문 | 탐침별 온도 감도 수치. ASHS PDF가 HTML로 막힘 |
| Moon & Son 2021(E28) | 온실 간 전이학습 수치 |
| Incrocci et al. 2020, Agric. Water Manag. 242:106393 (Irrigation management of European greenhouse vegetable crops) | 유럽 고형배지 급액 관행·계절 배액률 정리. WUR 포털 요약만 확인(서지 확인), 본문 미확보 → 위 목록에서 제외 |
| Sonneveld & Voogt (2009) 『Plant Nutrition of Greenhouse Crops』 | 흡수 농도·계절별 근권 EC 관리의 표준 교재. 단행본이라 접근 불가 |
| 영국·네덜란드 코이어 딸기 실무 자료(AHDB 등) | 겨울 배지 EC 관리 수치. 이번 검색에서 못 찾음 |
| 기후만으로 배지 EC를 추정한 연구 | 검색어 6종에서 찾지 못함. 없다는 증거는 아님 |

---

## 5. 검색 기록(요약)

| 순번 | 검색어(요지) | 검색처 | 결과 | 열어 본 것 |
|---|---|---|---|---|
| 1 | substrate EC prediction greenhouse ML soilless | WebSearch | 9 | E01(PMC 본문), E03(OpenAlex 초록) |
| 2 | virtual sensor substrate EC climate without irrigation | WebSearch | 9 | 업체 페이지 위주, 관련 논문 없음 |
| 3 | strawberry coir substrate EC seasonal winter | WebSearch | 9 | 기존 조사 출처(Depardieu) 중복 |
| 4 | OpenAlex: substrate EC prediction soilless | OpenAlex | 2,674 | E06, E02 초록 |
| 5 | Carmassi salinity build-up / Heinen rockwool model | WebSearch | 10/9 | E05 초록, E08 목록 |
| 6 | slab EC winter low radiation rockwool | WebSearch | 9 | E12·E13 PDF 본문, E18 본문 |
| 7 | 日本語: イチゴ 高設栽培 培地EC 冬季 上昇 | WebSearch | 9 | E11·E10 PDF 본문, 熊本(연작, EC 동태 없음 → 제외) |
| 8 | strawberry transpiration Penman-Monteith / EC between irrigation | WebSearch | 9/9 | E19 초록, E07 초록 |
| 9 | Bonelli 2024 MDPI | mdpi-res PDF | 1 | E09 본문 |
| 10 | Kalman filter nutrient / KF bias correction / EnKF salinity | WebSearch + OpenAlex | 9/9/9 | E21·E22·E26 초록 |
| 11 | transfer learning greenhouse / GPBoost | WebSearch | 9/10 | E27 PDF 본문, E24 목록 |
| 12 | pore water EC dielectric / temperature effect substrate | OpenAlex | 33/5,981 | E31·E32·E25 초록 |
| 13 | Nikolaou 2019 review | mdpi-res PDF | 1 | E17 본문(관련 절) |
| 14 | Grodan early stop time page | WebFetch | 1 | E14 |
| 15 | arXiv EC forecasting hydroponic | WebSearch | 9 | E04 PDF 본문(방법 절) |
