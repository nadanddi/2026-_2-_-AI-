# B. 해외 학술 문헌 조사 — 배지(근권) 온도와 실내 공기 온도의 '분리' 현상

- 작성: 2026-10-01, 집 클로드(문헌 조사 서브에이전트)
- 범위: 영어 중심 + 일본어·중국어·네덜란드어. 한국 학회지 논문은 범위 밖이지만 영어 초록으로 걸린 것만 참고로 표시(★국내).
- 대회 데이터 내용은 검색어에 쓰지 않았음. 일반 개념어로만 검색.
- 접근 제약: ScienceDirect·Springer·Tandfonline·MDPI·ASHS 본문은 WebFetch에서 403/리다이렉트로 막힘 → 해당 출처는 OpenAlex/Semantic Scholar API의 **초록만** 확인. PMC·J-STAGE·WUR edepot·NARO·ALIC·북방원예 PDF는 본문을 직접 읽음.

---

## 0. 요약 (5줄)

1. **가장 유력: 기록되지 않은 근권/크라운 국소 가온(온수관·전열선)이 처리구별로 켜지고 꺼짐.** 국소 가온은 공간 난방과 별개로 배지를 18~21℃로 붙잡아 두고, 실내 공기는 4~10℃여도 상관없음. 가온한 베드와 안 한 베드의 차이는 약 5℃(B05: 13~15 vs 18~21℃), 크라운은 21℃(B08, 공기 4~7℃). 가온을 꺼도 새벽에 +1.1℃가 남는 관성이 있음(B03). → "공기 10℃인데 배지 19℃"와 처리구 이어붙임 구조를 함께 설명함.
2. **고설 베드는 땅의 열을 받지 못함 + 공간 난방의 열은 위로 감.** 그래서 공기를 데워도 배지는 공기보다 낮게 남음. 온수관 근권 가온을 해도, 가장 추운 날에 공기 13.6~14.3℃ vs 배지 11.5~12.0℃로 약 2℃ 낮았음(B06). 덕트·난방관 위치에 따라 아래쪽은 차갑게 유지됨(B02, B15). → "난방을 많이 한 날 배지가 공기보다 약 2.3℃ 낮게 수렴"과 크기·방향이 맞음.
3. **겨울철 관수·양액이 차가움(10~15℃, B03)과 배지의 큰 열 관성.** 배지는 공기보다 2~5시간 늦게 따라감(B10). 관수 때마다 배지 온도가 출렁이고(B03), 깊은 층은 회복에 24~48시간이 걸림(B18). → 하루 안의 계단형 하강과 느린 냉각("18.5→17.5℃ 천천히 식음")에 부합.
4. **측정 위치와 베드(槽) 위치에 따른 차이도 1~4℃.** 같은 온실 안에서도 槽 사이 최저온이 1℃ 이상 차이 남(B06). 테이블 아래 공기도 2~4℃ 차이(B16). 실내 기준 센서는 하부 군락·근권을 대표하지 못함(B15).
5. "**난방 100%·한낮 공기 16℃인데 배지 8℃**"(8℃ 차)를 직접 설명하는 문헌은 찾지 못함. 문헌에 나온 크기(무가온 고설 베드 −2℃, 차가운 양액, 증발냉각 −1~5℃)를 다 합쳐도 모자람. 그래서 무가온 구획의 별도 센서이거나 센서 이탈·건조 같은 기록 생성 문제일 가능성을 우선 검토해야 함(추정).

---

## 1. 검색 기록 표

| 순번 | 검색어 | 검색처 | 날짜 | 결과 수(대략) | 열어 본 것 |
|---|---|---|---|---|---|
| 1 | strawberry root-zone temperature elevated bed substrate temperature greenhouse winter air temperature difference | Web(일반 검색) | 2026-10-01 | 9 | Springer(Jo&Shin) 리다이렉트 실패, Tandf(Shin 2023) 403, MDPI 403, PMC(Myung 2024) 본문 |
| 2 | substrate temperature energy balance model soilless greenhouse slab heat transfer | Web | 2026-10-01 | 9 | 목록만 |
| 3 | strawberry crown temperature control local heating tabletop hydroponics | Web | 2026-10-01 | 9 | 목록만(ASABE Moon 초록 요지) |
| 4 | DOI 조회(Jo&Shin 2022, Shin 2023, Kalorizou 2025) | Semantic Scholar API | 2026-10-01 | 3 | 초록(Jo&Shin은 초록 비공개) |
| 5 | strawberry+root+zone… 등 3건 | Semantic Scholar 검색 API | 2026-10-01 | 0(429 rate limit) | 실패 |
| 6 | strawberry root-zone temperature substrate heating | OpenAlex | 2026-10-01 | 2,969 | 상위 25 목록 → Kim 2009 ×2, Hidaka 2017, Moritani 2018 |
| 7 | substrate temperature model greenhouse soilless heat balance | OpenAlex | 2026-10-01 | 4,715 | 상위 25 목록(대부분 무관) |
| 8 | root zone temperature prediction greenhouse machine learning | OpenAlex | 2026-10-01 | 12,281 | 상위 25 목록(무관) |
| 9 | Kawasaki & Yoneda 2019 Hort. J. 리뷰 | J-STAGE | 2026-10-01 | 1 | **PDF 본문 전체**(pdftotext) |
| 10 | Kim et al. 2009 園芸学研究 8(2) | J-STAGE | 2026-10-01 | 1 | 초록 |
| 11 | 高設栽培 イチゴ 培地温度 気温 冬季 夜間 培地加温 | Web | 2026-10-01 | 9 | ALIC 2012(수상) 본문, NARO 근중국 보고서 PDF(글꼴 깨짐→메타데이터만) |
| 12 | イチゴ クラウン温度制御 温湯 チューブ 培地温度 推移 | Web | 2026-10-01 | 9 | NARO 카탈로그 67 본문 |
| 13 | 草莓 高架栽培 基质温度 气温 冬季 变化规律 | Web | 2026-10-01 | 9 | 北方园艺 2022 He Fen 등 **PDF 본문 전체** |
| 14 | substrate temperature LSTM/random forest root zone hydroponic inputs | Web | 2026-10-01 | 9 | PMC(Cheng 2024) 본문 요약, RF-XGBoost 초록 |
| 15 | root zone temperature model soilless substrate heat transfer… | Web | 2026-10-01 | 9 | Cross 2025 초록(OpenAlex), Garrido 2005/2006 서지만 |
| 16 | substrate temperature rockwool slab rail pipe lower than air | Web | 2026-10-01 | 10 | 목록만(무관) |
| 17 | OpenAlex 배치 4건(strawberry RZ heating / crown heating elevated bench / heat balance / heated substrate cable) | OpenAlex | 2026-10-01 | 627~20,356 | 각 상위 10 → Moon 2016·2019(★국내), Hidaka 2016, Garrido |
| 18 | DOI 8건 초록 조회 | OpenAlex | 2026-10-01 | 8 | 초록 6건(Garrido 2건은 초록 없음) |
| 19 | OpenAlex 배치 6건(수직 온도구배, 관수수온, 양액온도, 토양열류, 암면 슬래브, 이력곡선) | OpenAlex | 2026-10-01 | 953~30,739 | 상위 10(대부분 무관, 검색엔진 잡음 큼) |
| 20 | hot air heater vertical temperature distribution bench colder | Web | 2026-10-01 | 9 | 목록·요지(MSU 기사 본문 로드 실패) |
| 21 | cold irrigation water winter substrate temperature drop | Web | 2026-10-01 | 9 | California Agriculture(Wierenga & Hagan) PDF 본문 |
| 22 | 高設栽培 培地温度 気温より低い 放射冷却 ベンチ 夜間 | Web | 2026-10-01 | 9 | NARO 2000(野茶研), NARO 2005(近中四) 성과정보 본문 |
| 23 | Myung 2024 AoB Plants 겨울 실험 재확인 | PMC | 2026-10-01 | 1 | 본문(방법·결과) |
| 24 | Kempkes temperature distribution heating pipe position | Web | 2026-10-01 | 10 | PLOS ONE van Westreenen 2020 본문 요약 |
| 25 | mattemperatuur substraat kas verwarming … WUR | Web | 2026-10-01 | 9 | edepot 289333(KEMA 2009), 415513(GTB-1439, 2017), 24884(Rapport 415, 2005) **PDF 본문** |
| 26 | Wageningen root zone slab heating pipe strawberry | Web | 2026-10-01 | 9 | 목록만 |
| 27 | OpenAlex 배치 6건(Diurnal RZT strawberry / vertical gradients / thermal screen / table-top / sensor placement / warm air stratification) | OpenAlex | 2026-10-01 | 108~12,609 | 상위 5 → González-Fuentes 2016(초록 없음), Deschamps 2019 초록 |
| 28 | dynamic model substrate temperature rockwool slab energy balance | Web | 2026-10-01 | 9 | 목록만 |
| 29 | Moritani root-zone cooling elliptic curve | Web·OpenAlex | 2026-10-01 | 10 | Moritani 2018 초록, 2023 서지만 |

---

## 2. 출처 표

확인 수준: **전문** = 본문을 직접 읽음 / **본문일부** = 본문을 읽었으나 그림 수치는 못 읽음 / **초록** / **목록** = 검색 결과의 제목·요지만 / **실패** = 접근 실패.
신뢰도: 상(심사 논문·공공기관 실측) / 중(공공기관 기술정보·보고서, 실측이지만 간략) / 하(요지·2차 인용).

| ID | 제목 | 저자 | 연도 | 매체 | DOI/URL | 언어 | 유형 | 확인 수준 | 신뢰도 |
|---|---|---|---|---|---|---|---|---|---|
| B01 | Local Temperature Control in Greenhouse Vegetable Production | Kawasaki Y., Yoneda Y. | 2019 | The Horticulture Journal 88(3):305-314 | 10.2503/hortj.utd-r004 | 영어 | 리뷰 | 전문 | 상 |
| B02 | (B01 안에서 인용) Kawasaki et al. 2011 덕트 위치에 따른 국소 난방 / Kempkes et al. 2000 난방관 위치 모델 / Sato & Kitajima 2010 크라운 전열가온 / Dan et al. 2015 미야기 크라운 온도제어 | — | 2000~2015 | 2차 인용 | B01 참고문헌 | 영어/일본어 | 실험(2차 인용) | B01을 통한 2차 확인 | 중 |
| B03 | Development of a root-zone temperature control system using air-source heat pump and its impact on the growth and yield of paprika | Myung J. et al. | 2024 | AoB Plants 16(5) plae047 | PMC11489771 | 영어 | 실험(코이어 슬래브, 온실) | 전문 | 상 |
| B04 | Effects of supplemental root-zone pipe heating systems on the growth and development of strawberry plants in a greenhouse during the winter season | Shin J., Lee B., Cui M., … Chun C. | 2023 | NZ J. Crop Hort. Sci. 53(4) | 10.1080/01140671.2023.2224035 | 영어 | 실험(코이어, 딸기) | 초록 | 상 |
| B05 | Spot Heating Technology Development for Strawberry Cultivated in a Greenhouse by Using Hot Water Pipe (★국내) | Moon J.P. et al. | 2016 | J. Korean Soc. Agric. Eng. 58(5) | 10.5389/ksae.2016.58.5.071 | 한국어(영문초록) | 실험 | 초록 | 상 |
| B06 | 不同栽培模式下温室草莓根区加温环境测试与分析 (Test and analysis of strawberry root-zone heating environment under different cultivation modes) | 何芬, 侯永, 尹义蕾, 田婧, 丁小明, 李中华 | 2022 | 北方园艺 2022(01):59-64 | 10.11937/bfyy.20212016 | 중국어 | 실측(베이징 연동온실, 온수관 근권가온) | 전문 | 상 |
| B07 | Effects of Root Zone Heating during Daytime (at Different Growth Stages) on the Flowering, Growth and Yield of Strawberry 'Akihime' Grown in Substrate Culture (2편) | Kim Y.S., Endo M., Kiriiwa Y., Chen L., Nukaya A. | 2009 | 園芸学研究 8(2):193, 8(3):315 | 10.2503/hrj.8.193 / 10.2503/hrj.8.315 | 일본어 | 실험 | 초록 | 상 |
| B08 | イチゴの高設栽培における省エネ加温技術(クラウン加温) | 水上宏二(福岡農総試) | 2012 | ALIC 野菜情報 2012年6月 | vegetable.alic.go.jp/yasaijoho/joho/1206_joho01.html | 일본어 | 기술해설(실측 그림) | 본문일부 | 중 |
| B09 | Energy Saving Effect for High Bed Strawberry Using a Crown Heating System (★국내) | Moon J.P., Park S.H., Kwon J.K., Kang Y.K. | 2019 | Protected Hort. & Plant Factory 28(4) | 10.12791/ksbec.2019.28.4.420 | 한국어(영문초록) | 실험 | 초록 | 상 |
| B10 | Parsimonious models of root zone temperature in soilless substrates through ensemble machine learning | Cross J.F., Owen J.S., Shreckhise J.H., Fields J.S. | 2025 | Smart Agricultural Technology | 10.1016/j.atech.2025.101289 | 영어 | 예측모델(노지 용기) | 초록(뒷부분 잘림) | 상 |
| B11 | Root-zone Cooling Evaluation Using Heat Pump for Greenhouse Strawberry Production | Moritani S., Nanjo H., Itou A., Imai T. | 2018 | HortTechnology 28(5) | 10.21273/horttech04007-18 | 영어 | 실험 | 초록 | 상 |
| B12 | Crown-cooling Treatment Induces Earlier Flower Bud Differentiation of Strawberry under High Air Temperatures | Hidaka K., Dan K., Imamura H., Takayama T. | 2017 | Environ. Control Biol. 55(1) | 10.2525/ecb.55.21 | 영어 | 실험 | 초록 | 상 |
| B13 | 気化潜熱を利用したイチゴ高設ベンチの培地冷却法 | NARO 野菜茶業研究所 | 2000(성과정보) | NARO 연구성과정보 | naro.go.jp/…/vegetea00-014.html | 일본어 | 실측 요약 | 전문(짧은 페이지) | 중 |
| B14 | 高設ベンチの強制気化冷却による促成イチゴ一次腋花房の出蕾の前進化 | NARO 近畿中国四国農研 | 2005 | NARO 연구성과정보 | naro.go.jp/…/wenarc05-29.html | 일본어 | 실측 요약 | 전문(짧은 페이지) | 중 |
| B15 | Substantial differences occur between canopy and ambient climate: Quantification of interactions in a greenhouse-canopy system | van Westreenen A., Zhang N., Douma J.C., Evers J.B., Anten N.P.R., Marcelis L.F.M. | 2020 | PLOS ONE 15(5): e0233210 | 10.1371/journal.pone.0233210 | 영어 | 실측·통계 | 본문 요약(WebFetch) | 상 |
| B16 | Energiebesparing door lokale verwarming — Test op teelttafels bij Elstgeest Potplanten (GTB-1439) | Raaphorst M., van Noort F. | 2017 | Wageningen Plant Research 보고서 | edepot.wur.nl/415513 | 네덜란드어(영문초록) | 현장 실측 | 전문 | 중 |
| B17 | Planttemperatuur als stuurparameter in kasklimaatregelingen (Rapport 415) | Campen J.B., Kempkes F.L.K., Houter B., Rijpsma E.C. | 2005 | Agrotechnology & Food Innovations(WUR) | edepot.wur.nl/24884 | 네덜란드어 | 현장 실측 | 본문일부(관련 절) | 중 |
| B18 | Effects of cold irrigation water on soil temperature and crop growth | Wierenga P.J., Hagan R.M. | 1960년대(정확 연도 미확인) | California Agriculture | californiaagriculture.org/article/112825 | 영어 | 노지 실측 | 전문(OCR 품질 낮음) | 중 |
| B19 | Beknopte literatuurstudie invloed bodem op kasklimaat bij energie-arme teelten | KEMA Nederland | 2009 | 보고서(Kas als Energiebron) | edepot.wur.nl/289333 | 네덜란드어 | 문헌리뷰 | 본문일부(2.2절) | 중 |
| B20 | Soil Marginal Effect and LSTM Model in Chinese Solar Greenhouse | Cheng W. et al. | 2024 | Sensors 24 | PMC11280564 | 영어 | 실측+LSTM | 본문 요약(WebFetch) | 상 |
| B21 | A Hybrid RF–XGBoost Model for Soil Temperature Prediction in Solar Greenhouses | Liu X. et al. | 2026 | Agriculture 16(7):774 | 10.3390/agriculture16070774 | 영어 | ML 예측 | 초록 | 중 |
| B22 | Hydroponic Thermal Regulation for Low-Energy Winter Strawberry Production in Mediterranean Coastal Infrastructures | Kalorizou H. et al. | 2025 | Horticulturae 11(11):1383 | 10.3390/horticulturae11111383 | 영어 | 실험(양액 가온) | 초록 | 중 |
| B23 | 自然エネルギーを利用したイチゴのクラウン温度制御 | NARO(みどり技術カタログ67) | 2020년대 | NARO 카탈로그 | naro.go.jp/smart-nogyo/midori/catalog/catalog_67.html | 일본어 | 기술 소개 | 전문(짧음) | 중 |
| B24 | Optimized Design of Irrigation Water-Heating System and Its Effect on Lettuce Cultivation in a Chinese Solar Greenhouse | Guo L.B. et al. | 2024 | Plants 13(5):718 | 10.3390/plants13050718 | 영어 | 실험 | 초록 | 중 |
| B25 | Modelling the Transient Thermal Behaviour of Sand Substrate heated by Electric Cables / Application of a Heat Transfer Model for Heated Substrates… | Garrido Fernández M.D., Ramiro Rodríguez M. et al. | 2005 / 2006 | Biosystems Engineering | 10.1016/j.biosystemseng.2004.11.005 / 10.1016/j.biosystemseng.2006.05.010 | 영어 | 열전달 모델 | 목록(서지만; 초록 미확보) | — |
| B26 | Effect of root-zone heating using positive temperature coefficient film on growth and quality of strawberry in greenhouses | Jo W.J., Shin J.H. | 2022 | Hortic. Environ. Biotechnol. | 10.1007/s13580-021-00384-5 | 영어 | 실험 | 목록(검색 요지: 코이어가 암면보다 가온 온도를 잘 유지) | 하 |
| B27 | Diurnal root zone temperature variations affect strawberry water relations, growth, and fruit quality | González-Fuentes J.A. et al. | 2016 | Scientia Horticulturae | 10.1016/j.scienta.2016.03.039 | 영어 | 실험 | 목록(초록 미확보) | — |
| B28 | Heating and cooling methods for the subirrigation of strawberry plants using air and geothermal heat pumps | Moritani S. et al. | 2023 | Environ. Dev. Sustain. | 10.1007/s10668-023-03006-5 | 영어 | 실험 | 목록 | — |
| B29 | 促成イチゴの高設栽培における連続出蕾性に与える定植後の培地昇温抑制と施肥時期の効果 | 山崎敬亮, 熊倉裕史, 濵本浩 | 2019 공개 | 近中四農研報告 7:35-47 | 10.24514/00001646 | 일본어 | 실험 | 메타데이터만(PDF 글꼴 깨짐) | — |
| B30 | Twofold Increase in Strawberry Productivity by Integration of Environmental Control and Movable Beds | Hidaka K. et al. | 2016 | Environ. Control Biol. 54(2) | 10.2525/ecb.54.79 | 영어 | 실험 | 초록(근권온도 언급 없음) | 상(관련성 낮음) |
| B31 | Metalized-striped Plastic Mulch Reduces Root-zone Temperatures… Winter Strawberry | Deschamps S.S., Agehara S. | 2019 | HortScience 54(1) | 10.21273/hortsci13583-18 | 영어 | 노지 실험 | 초록 | 상(관련성 낮음) |

---

## 3. 주장 → 흔적 표

"우리 데이터에서 보여야 할 흔적"은 **그 주장이 우리 온실에도 맞다면** 나타나야 할 패턴이다(추정). "입력 변수로 확인 가능 여부"는 대회 입력(외기·실내기상·구동기)만으로 그 흔적을 볼 수 있는지를 뜻한다.

### 3-A. 숨은 국소(근권·크라운) 가온 — 질문 1

| 출처 | 관련 주장(수치, 근거 위치) | 우리 데이터에서 보여야 할 흔적 | 입력 변수로 확인 가능? |
|---|---|---|---|
| B05 | 겨울철 온수관(22~24℃, 480 L/h, 하루 종일 순환) 근권 가온 베드: 야간 근권 18~21℃. 무가온 대조 베드: 13~15℃. 크라운 가온 베드 13~17℃ vs 대조 8~14℃ (초록) | 가온 구획은 밤에 배지 18~21℃로 거의 평평하고, 같은 날 같은 공기에서 무가온 구획보다 약 5℃ 높음. 구획(처리구)별로 배지-공기 차의 "층"이 생김 | 직접 확인 불가(가온 신호 없음). 간접 확인: 밤의 배지-공기 차, 배지 야간 하강 기울기(≈0), 배지 일교차 축소로 날을 군집 |
| B08 | 하우스 기온을 4℃·7℃로 낮춰도 크라운 가온(21℃)이면 크라운 온도가 높게 유지됨. 20:00~08:00 야간 비교(그림 2, 표 1). 연료 52~86% 절감 | 공간 난방 설정이 낮은(공기 ≈ 10℃ 이하) 밤에 배지만 19℃ 근처를 유지하는 날이 있음. 그런 날은 난방 지표가 낮거나 0이어도 배지가 높음 | 부분 가능: 난방 지표가 낮은데 배지가 높은 밤을 고르면 후보 날짜 목록이 나옴 |
| B03 | 가온 처리(SSH, 06–08·16–20시 순환)는 20:00에 대조보다 +3.2℃. 순환을 멈춘 밤에도 배지가 계속 내려가지만 06:00에 여전히 +1.1℃ (본문 결과, Fig. 5) | 가온 정지 뒤 배지는 지수적으로 천천히 식음(수 시간~하루 규모의 관성). 가온 "켜짐→꺼짐" 전환일에는 하루 이상 차이가 서서히 줄어듦 → 며칠 지속처럼 보임 | 간접: 배지-공기 차의 일별 시계열이 계단+지수감쇠 모양인지 확인 |
| B04 | 코이어 재배 딸기에서 크라운 기반(CBH)·뿌리 기반(RBH) 보조 파이프 가온을 비교. RBH가 2~4월 적정 근권온도 유지 시간이 가장 김(초록) | 시험 온실에서 근권 가온은 2~4월 등 한 기간 동안 처리구 단위로 운영되는 것이 흔함 → 분리일이 특정 월·구획에 몰림 | 날짜·구획 분포로 확인 가능(온실 ID·날짜 축의 집중도) |
| B07 | 낮 시간 배지 가온 처리: 무가온 16.2℃ ~ 강가온 22.0℃. 처리 기간을 생육 단계별로 다르게 둠(9월~4월, 12월~4월 등) | 낮 시간만 가온하는 설계면 낮에만 배지가 공기와 떨어짐. 처리가 생육 단계(날짜 구간)로 켜지고 꺼짐 | 간접: 분리가 낮/밤 중 어디에 나타나는지 시간대별로 분해 |
| B09 | 공간 4℃ + 크라운 가온 13~15℃(순환펌프 on/off 제어)로 8℃ 공간 난방 대비 29.7% 절감 | 공간 난방 설정이 처리구마다 다를 수 있음(4/6/8℃). 같은 날 구획 간 공기 온도의 바닥값이 다름 | 가능: 구획별 야간 실내기온 하한(설정값) 군집 |
| B23, B12 | 지하수(15~18℃)를 크라운 튜브에 흘려 가을엔 냉각, 겨울엔 가온. 냉각 튜브 수온 10~25℃ 처리(B12) | 가을(고온기)엔 배지·크라운이 공기보다 낮게, 겨울엔 높게 유지 → 계절에 따라 분리 방향이 뒤바뀜. 지하수 온도(≈15~18℃) 근처에 수렴 | 간접: 분리일 배지 온도가 15~18℃ 근처로 모이는지 히스토그램 확인 |

### 3-B. 고설 베드의 열 고립 + 공간 난방의 성층화 — 질문 3

| 출처 | 관련 주장(수치, 근거 위치) | 우리 데이터에서 보여야 할 흔적 | 입력 변수로 확인 가능? |
|---|---|---|---|
| B06 | 베이징 최한기 실측. 온수관 근권 가온이 있어도 기간 평균은 공기 14.4/15.5℃ vs 배지 13.8℃. 극한 저온일(외기 일평균 −12.8℃)에는 공기 일평균 14.3/13.6℃ vs 배지 일평균 12.0/11.5℃(**배지가 약 2℃ 낮음**). 공기는 야간 가온으로 약 13℃로 안정. 고설 槽는 "지온 지원이 없고 배지 부피가 작아 환경 영향을 쉽게 받음"(3절 토의) | 외기가 매우 낮고 난방을 많이 하는 날일수록 배지가 공기보다 1.5~2.5℃ 낮게 유지됨. 차이의 크기는 외기온(또는 난방 가동률)과 단조 관계 | **가능**: (배지−공기) vs 외기온·난방 지표의 일별 관계. "난방 많은 날 −2.3℃"와 직접 비교 가능 |
| B06 | A자형 3槽 배치에서 가운데 槽 최저온이 좌우보다 1℃ 이상 높음. A자형은 최저값 편차 4.5~25.3%(평균 17.4%), H형은 0~11%(평균 1.1%) | 같은 온실에서도 베드 위치에 따라 배지 최저온이 약 1℃ 차이 남 → 구획(처리구)마다 고정된 오프셋 | 간접: 구획을 구분할 수 있다면 구획별 상수 오프셋으로 검정 |
| B01/B02 | 덕트를 지면에 깐 일반 난방과 달리, 덕트를 군락 가까이 매달면 생장점·꽃은 따뜻해지지만 **하부는 낮게 유지**됨(Kawasaki et al. 2011, Fig. 2–3). 연료 26% 절감. 난방관을 군락 가까이 두면 상부 잎 온도는 오르지만 기온은 조금만 오름(Kempkes et al. 2000) | 난방 열원의 높이·방향에 따라 같은 기온 센서값이어도 배지 높이의 공기는 더 차가울 수 있음. 온풍 난방 가동률이 높을수록 (기온 센서 − 배지) 차가 커짐 | 가능: 난방 지표 구간별 (공기−배지) 차의 기울기. 유동팬 가동 시 차가 줄면 성층화 근거 |
| B15 | 실내 기준 센서(군락 위)와 군락 내부 공기 차이: 맑은 날 정오에 최대 5℃(군락이 더 차가움), 흐린 날 2℃. 군락 위→아래로 온도가 낮아짐. 밤에는 차이의 방향이 뒤바뀌고 열복사가 지배. 난방관 온도가 높으면 차이가 줄어듦 | 낮에 일사가 클수록 (실내 기온 − 배지) 차가 커짐. 밤에는 방향이 바뀜. 측정된 '실내 기온'은 배지 높이 환경을 대표하지 못함 | 가능: 일사·시간대별 잔차 패턴 |
| B19 | 비가온 PE 온실 CFD: 맑은 밤(하늘 −10℃)에는 실내 기온이 외기보다 2.5K 낮아지는 역전 발생. 흐린 밤은 실내가 3.6K 높음. 토양→실내 공기 열류는 20 W/m²로 가정. 일주기 토양 온도 변동의 유효 깊이는 약 25cm(2.2.2–2.2.3절) | 고설 베드는 이 20 W/m² 규모의 바닥 열을 받지 못하고, 맑은 밤에 지붕으로 복사 손실을 봄 → 맑고 추운 밤, 보온커튼이 열린 밤에 배지가 더 차가움 | 가능: 보온커튼 상태 × 외기·운량 대리지표(전날 일사·습도)별 야간 (배지−공기) |
| B16 | 테이블 매트 가온 7 W/m² → 화분 온도 약 +1℃. 테이블 아래 난방관 구획의 테이블 아래 공기가 다른 구획보다 2~4℃ 높음. 같은 칸 안에서도 수평 온도 구배가 있음(본문 Fig. 4–5) | 국소 가온은 W/m² 단위만으로도 수 ℃의 근권-공기 차를 만듦. 같은 칸 구획 사이 공기도 2~4℃ 차이 날 수 있음 → 처리구 이어붙임에서 구획 간 공기 값 자체가 다를 수 있음 | 간접 |

### 3-C. 관수·양액 온도와 열 관성 — 질문 1·2

| 출처 | 관련 주장(수치, 근거 위치) | 우리 데이터에서 보여야 할 흔적 | 입력 변수로 확인 가능? |
|---|---|---|---|
| B03 | 겨울 무가온 탱크의 양액은 10~15℃. 무가온 대조 배지는 16~19℃(밤 기온 <15℃). 08~17시 매시간 관수(2→10분). 가온 양액을 줘도 관수 때마다 배지가 매시간 출렁임(본문 결과) | 관수 시간대(낮)에 배지가 계단형으로 출렁이거나 낮 상승이 억눌림. 겨울에는 차가운 양액 때문에 낮 배지가 공기보다 낮음 | 직접 불가(관수 기록 없음). 간접: 배지의 시간별 1차 차분에 주간 고주파 성분이 있는지 확인 |
| B17 | 토마토 암면 매트 온도는 관성이 매우 큼. 관수 시점이 매트 온도에 드러나지 않음. 하루 대부분 매트가 줄기보다 낮음. 양액 온도와 매트 온도를 측정·조절하라고 권고(3.5절, 결론) | 슬래브형 배지라면 관수 신호가 거의 보이지 않고, 배지는 공기보다 낮은 쪽으로 평활화됨 | 해당 없음(이 출처는 관수 흔적이 없을 수도 있다는 반증으로 사용) |
| B18 | 차가운 관수 뒤 토양 온도가 수원 온도 근처까지 떨어짐. 10cm 깊이는 24시간 안에 회복, 30cm 깊이는 최소 48시간 필요. 24인치에서 82→73°F(−5℃) 사례(본문) | 큰 관수(또는 배지 교체·세척) 뒤 배지 온도가 1~2일에 걸쳐 회복 → "2~3일 지속"의 일부를 설명할 수 있음 | 간접: 분리 시작일의 배지 급락 뒤 회복 곡선 모양 |
| B10 | 용기 배지 온도는 공기 온도·하향 단파복사·VPD와 강하게 상관하며, 상관이 최대가 되는 것은 **2~5시간 앞선** 값. 공기온도+단파복사 2변수 모델로 좋은 성능(초록, 성능 수치는 잘려 미확인). 관수 정보는 입력에 없음 | 정상일에는 배지 ≈ 2~5시간 지연된 공기·일사 함수로 잘 맞아야 하고, 분리일에만 큰 잔차가 생김 → 분리는 "입력의 함수"가 아니라 숨은 상태로 판단 | **가능**: 지연 특징(2~5h)으로 적합한 정상일 모델의 잔차로 분리일 판정 |
| B11 | 공기-토양 온도 관계를 타원(이력 루프)으로 표현. 타원 중심 = 대표 일 토양온도, 기울기 = 안정도. 냉각수 9.4℃에서 토양이 가장 낮음 | 날마다 (공기, 배지) 루프의 중심과 기울기를 계산하면, 숨은 가온·냉각이 있는 날은 중심이 수직으로 이동하고 기울기가 0에 가까워짐(평평) | **가능**: 일별 타원 중심·기울기를 특징으로 분리일 군집 |
| B13, B14 | 고설 베드 증발냉각(젖은 부직포 ± 송풍): 맑은 날 최고기온 때 배지가 5~10℃ 낮아짐, 밤에는 약 1℃. 냉각 능력은 최대 약 300 W/m². 송풍 강제냉각은 낮 평균 최대 약 5℃ 낮춤 | 정식 직후(가을)에 배지가 공기보다 낮과 밤 모두 낮고, 일사가 클수록 차가 커지는 날이 몇 주 이어짐 | 가능: 가을 분리일의 (공기−배지) vs 일사 기울기 |
| B22, B24 | 양액·관수수를 20~22℃로 데우면 생육이 좋아짐. 겨울 관수수가 차가운 것이 저온 스트레스 요인(초록) | 양액 가온 처리구가 있으면 낮(관수 시간)에만 배지가 공기 위로 떠 있음 | 간접 |

### 3-D. 배지 온도 예측 사례 — 질문 4

| 출처 | 관련 주장(수치, 근거 위치) | 우리 데이터에서 보여야 할 흔적 | 입력 변수로 확인 가능? |
|---|---|---|---|
| B10 | 1~2개 예측변수(공기온도 + 단파복사, 2~5h 지연)로 배지 온도 예측. 관수·가온 정보 없음. 지면 덮개 차이는 배지 온도에 작은 영향만 줌(초록) | 우리 '정상일' 성능의 상한 참고치. 분리일 오차는 입력 부족 때문이지 모델 용량 문제가 아님 | 가능 |
| B20 | 중국 일광온실 토양(지중) 온도 LSTM: MAE 0.05℃, R² 0.9984. 남쪽 가장자리 2.6m 구간은 저온대(11.06~19.05℃), 같은 깊이에서 수평으로 최대 9.42℃ 차이 | 위치(가장자리 vs 중앙)만으로 근권 온도가 수 ℃ 다를 수 있음. 자기회귀 입력(과거 배지 온도)이 있으면 오차가 매우 작아짐 → 우리 대회는 과거 배지가 입력이 아니므로 직접 비교 불가 | 해당 없음(참고) |
| B21 | 일광온실 20cm 토양온도의 12시간 앞 예측: RF-XGBoost 하이브리드 R² 0.9927, 12h 최대오차 2.52℃(단독 모델 3.12~4.60℃). "열 이력(thermal hysteresis)"을 핵심 난점으로 지목 | 지연·이력 특징이 핵심. 단, 토양(대용량)이라 고설 배지보다 관성이 큼 | 참고 |
| B25 | 전열 케이블로 가온한 모래 배지의 과도 열거동 모델, 케이블 간격 설계 모델(서지만 확인) | 물리모델로 가온 배지를 다룰 때 가온 전력이 핵심 입력임 → 우리처럼 가온 정보가 없으면 물리모델도 이 항을 추정 상태로 둬야 함 | 해당 없음 |

### 3-E. 열수지 항의 상대적 크기(문헌에서 얻은 범위) — 질문 2

| 항 | 문헌 값 | 출처 | 비고 |
|---|---|---|---|
| 국소 가온(근권·크라운) | 7 W/m² 매트 → +1℃. 온수관 22~24℃ 순환 → 대조 대비 +5℃. 크라운 21℃ 유지(공기 4~7℃) | B16, B05, B08 | 우리 입력에 없는 항 중 가장 큼 |
| 증발냉각(배지 표면) | 낮 최대 약 300 W/m², −5~10℃. 밤 −1℃ | B13 | 정식 직후 처리 |
| 관수·양액 | 겨울 양액 10~15℃. 관수마다 배지 출렁임. 회복 24~48h | B03, B18 | 관수량·빈도에 비례 |
| 바닥(지중) 열 | 비가온 온실 가정치 20 W/m²(토양→공기). 고설 베드는 이 지원이 없음 | B19, B06 | 고설일 때 배지가 공기보다 낮아지는 이유 |
| 공기 대류·성층화 | 군락 위·아래 기온 차 최대 5℃(낮), 밤은 반대. 테이블 아래 공기 2~4℃ 구획 차 | B15, B16 | 기준 센서 대표성 문제 |
| 장파 복사(밤) | 맑은 밤 실내 역전 −2.5K(외기 대비) | B19 | 보온커튼으로 줄어듦 |
| 열 관성 | 배지는 공기보다 2~5h 늦음. 가온 정지 뒤 수 시간 이상 잔존(+1.1℃ @06시) | B10, B03 | 지속성 일부 설명. 다만 "2~3일"은 관성만으로 설명하기 어려움(추정) |

> 각 항의 W/m² 크기를 하나의 고설 딸기 시스템에서 동시에 정량화한 문헌은 찾지 못함. 위 범위는 서로 다른 시스템에서 모은 값이라 직접 더할 수 없음.

---

## 4. 아직 못 본 곳 / 다음에 조사할 곳

| 대상 | 이유 |
|---|---|
| Shin et al. 2023(B04) 본문 | CBH/RBH 처리별 배지 온도 일변화 그림이 있을 것. 처리구 운영 기간·시간이 우리 구획 구조와 비교할 핵심 자료. Tandfonline 403으로 막힘 |
| Jo & Shin 2022 PTC 필름(B26) 본문 | 코이어·암면 배지의 가온 유지 차이, 공기 대비 배지 온도 수치. Springer 막힘, 초록도 비공개 |
| Cross et al. 2025(B10) 본문 | 2변수 모델의 RMSE 수치와 지연 구조. 우리 '정상일' 기준선으로 쓸 수 있음. ScienceDirect 403(OA라는 표시가 있으므로 다른 경로로 재시도 가치 있음) |
| Moritani 2018(B11)·2023(B28) 본문 | 공기-배지 타원 지표의 구체적 계산식. ASHS·Springer 막힘. HortTechnology는 ashs downloadpdf 경로로 재시도 가능 |
| González-Fuentes et al. 2016(B27) | 근권 온도 일변동 처리(딸기). 초록 미확보 |
| Garrido Fernández 2005/2006(B25) | 가온 배지 열전달 모델의 항 구성(대류계수, 증발). 초록 미확보 |
| NARO 近中四 보고서(B29) 본문 | 고설 베드 시트 외측 냉각·培地昇温抑制 실측 그래프. PDF 글꼴이 깨져 판독 불가 → OCR 필요 |
| 奈良県 イチゴ高設栽培の手引き(2004) PDF, 愛知県 環境制御ガイドライン PDF, NARO 大規模いちご生産マニュアル PDF | 실용 지침에 배지-기온 실측 그래프가 있을 가능성(검색 결과로만 확인: "最低気温12℃, 培地温度15℃가 目安", "クラウン表面温度 18℃ 目標(20時~8時)") |
| Kawasaki et al. 2011, Sato & Kitajima 2010, Dan et al. 2015 원문 | B01에서 2차 인용만 함. 덕트 위치별 하부 온도, 크라운 전열 21℃의 실측 원자료 |
| Acta Horticulturae "Vertical temperature gradients in heated greenhouses"(ISHS 70_12) | 난방 방식별 수직 구배 수치. ISHS 초록 페이지 미열람 |
| CNKI: 高架草莓 基质温度 夜间 / 根区加温 对比 空气温度 | 북방원예 외에 중국 고설 딸기 실측 논문이 더 있을 가능성(B06 참고문헌 [7][8][17]: 徐川 2015, 林晓 2014, 高敏 2015) |
| Semantic Scholar 검색 API | 429 rate limit으로 실패. 시간 간격을 두고 재시도하면 OpenAlex보다 관련도 높은 결과를 얻을 수 있음 |

### 메모(작업 부수효과)
- 일본어 PDF를 추출하려고 시스템 Python에 `pypdf`를 pip로 설치했음(프로젝트 파일은 변경하지 않음).
- 중간 텍스트 파일은 세션 scratchpad에만 저장함.
