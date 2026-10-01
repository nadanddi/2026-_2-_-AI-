# C. 공개 데이터셋·대회·실무(산업) 자료 조사 — 배지 온도 '분리' 현상

- 작성: 2026-10-01, 집 클로드(자료 조사 하위 작업). 검색어에는 대회 데이터 내용을 넣지 않고 일반 개념만 썼음.
- 확인 수준 표기: **원문 확인**(본문을 직접 텍스트로 뽑아 읽음) / **내용 확인(요약)**(웹 도구의 요약을 거쳐 읽음 — 원문 대조는 안 함) / **설명 페이지만** / **검색 결과 목록만**(검색 스니펫) / **접근 실패**.
- 주의: 웹 도구 요약은 PDF를 못 읽을 때 그럴듯한 내용을 지어낸 적이 있었음(C35·C24에서 확인). 그래서 수치는 '원문 확인' 항목만 그대로 믿을 것.

## 0. 요약 (5줄)
1. 국내 연구용 딸기 온실 실험(문종필 외 2016·2019, 이태석 외 2021)에서는 **같은 온실 안 베드나 같은 시기 옆 동마다 근권·관부 가온을 처리구로 다르게 걸었음**. 근권 가온 베드의 근권 온도는 밤 18~21℃(주야 평균 약 20℃)로 거의 일정했고, 무처리 베드(13~15℃)보다 주야 평균이 6~9℃ 높았음. 이는 "밤 공기 10℃인데 배지 19℃"와 같은 크기와 방향임.
2. 가온은 축열조 온수(20~24℃)를 24시간 돌리거나, 관부 근처 센서로 순환 펌프를 켜고 끄며(시간대별 13/15℃) 운전했음. 운전 기간은 대략 11월~2·3월이고, 일본 지침에는 야간(20~08시)에만 가온하는 안도 있음. 그런데 **이 가온은 우리 입력(난방 복합 지표·공기 온도)에 나타나지 않는 별도 회로**임.
3. 반대 방향(난방을 세게 하는데 배지가 차가움)은 **차가운 양액 급액**(온수 급액 시 지하수보다 8.7℃ 높고 베드 약 5℃ 상승 → 거꾸로 찬물이면 하강), **배지 냉각**(효고현, 최대 3℃), **센서가 배지 밖으로 빠진 경우**(온도는 계속 기록되고 EC·함수율은 0 근처)로 설명할 수 있음.
4. 공개 데이터셋 중 배지 온도가 있는 것은 Wageningen 자율온실 챌린지 2회(GroSens 슬래브 온도·EC·함수율)와 4회(화분 3개 토양센서 온도, 난방관 온도), AI Hub 수직농장 딸기(배지·양액 온도)임. 변수 목록과 '배지와 공기 관계 분석'까지 직접 확인한 것은 4회 논문 하나뿐이고 나머지는 설명 페이지 수준임.
5. 그룹 단위 오프셋을 맞힌 대회 해법은 배지 온도 대회에서는 찾지 못했음. 비슷한 사례인 ASHRAE·DrivenData의 상위 해법은 그룹 수준을 **표적의 과거값 정규화**나 **같은 사이트 동시 이상 제거**로 맞췄음. 앞의 것은 우리 규정(평가 행 특징은 입력만)으로 쓸 수 없고, 뒤의 것(같은 날짜에 여러 온실이 동시에 분리되는지)은 학습 데이터로 바로 점검할 수 있음.

---

## 1. 검색 기록 표

| 순번 | 검색어 | 검색처 | 날짜 | 결과 수(대략) | 열어 본 것 |
|---|---|---|---|---|---|
| 1 | Autonomous Greenhouse Challenge dataset 4TU slab temperature substrate | WebSearch | 2026-10-01 | 10 | 4TU 2회 페이지(C02), WUR 4회 페이지(C03) |
| 2 | greenhouse dataset substrate temperature root zone temperature sensor open data Zenodo | WebSearch | 2026-10-01 | 10 | 없음(목록만: 토양·시뮬레이션 위주, 온실 배지 온도 데이터 없음) |
| 3 | 데이콘 스마트팜 경진대회 배지 온도 예측 해법 | WebSearch | 2026-10-01 | 30(3회 재검색) | 없음(DACON 온도추정 대회 목록만, C30) |
| 4 | strawberry root zone temperature extension greenhouse substrate heating | WebSearch | 2026-10-01 | 9 | eXtension(C10), NZJCHS(C38, 실패) |
| 5 | Autonomous Greenhouse Challenge second edition GrodanSens.csv slab temperature … | WebSearch | 2026-10-01 | 10 | Grodan 블로그(C06) |
| 6 | Grodan GroSens slab sensor temperature placement rockwool … | WebSearch | 2026-10-01 | 8 | GroSens 2.2 설치 안내 PDF(C07) |
| 7 | "slab temperature" greenhouse rockwool irrigation cooling night heating pipe … | WebSearch | 2026-10-01 | 10 | Grower2Grower(C09) |
| 8 | 딸기 고설재배 배지 온도 관리 근권 가온 농업기술센터 | WebSearch | 2026-10-01 | 8 | PMC10183158(C22), 월간원예 Q&A(C26, 실패) |
| 9 | Kaggle Autonomous Greenhouse Challenge 2nd edition GrodanSens notebook … | WebSearch | 2026-10-01 | 10 | Kaggle 미러(C05, 실패) |
| 10 | virtual sensor substrate temperature prediction greenhouse machine learning | WebSearch | 2026-10-01 | 9 | 없음(공기 온도 가상센서·CFD뿐, 배지 온도 없음) |
| 11 | AI Hub 스마트팜 딸기 환경 데이터 배지온도 근권 데이터셋 | WebSearch | 2026-10-01 | 10 | AI Hub 수직농장 딸기(C23) |
| 12 | いちご 高設栽培 培地加温 温湯管 培地温度 管理 マニュアル 農業技術センター | WebSearch | 2026-10-01 | 9 | ALIC(C19), 奈良(C20, 실패), 北海道(C18), 愛知(C17) |
| 13 | Kaggle competition soil temperature prediction winning solution group offset per site | WebSearch | 2026-10-01 | 30(3회) | 없음(해당 대회 없음) |
| 14 | 스마트팜 경진대회 배지 EC 배지 함수율 예측 수상 코드 공유 | WebSearch | 2026-10-01 | 9 | 없음(기사 목록만) |
| 15 | Priva rootzone heating "grow pipe" substrate temperature control … | WebSearch | 2026-10-01 | 10 | Springer PTC 필름(C37, 실패), ATTRA PDF(C11) |
| 16 | substrate temperature strawberry tabletop heating pipe air temperature difference night … | WebSearch | 2026-10-01 | 9 | MDPI agronomy(C39, 실패) |
| 17 | 딸기 고설 배지 온도 기온 차이 야간 측정 겨울 온풍난방 … | WebSearch | 2026-10-01 | 9 | KCI ART003013077(C24, 엉뚱한 논문 반환), JESI PDF(C35) |
| 18 | Wageningen greenhouse experiment compartments … "substrate temperature" hourly 4TU | WebSearch | 2026-10-01 | 10 | AGC 4회 논문 PMC(C04) |
| 19 | substrate sensor temperature reading wrong sensor dried out slab removed exposed air … | WebSearch | 2026-10-01 | 9 | Growlink Acclima 문제 해결 안내(C12) |
| 20 | DACON 딸기 환경 데이터 예측 경진대회 private 1위 코드 | WebSearch | 2026-10-01 | 40(4회) | 없음(해당 대회 없음) |
| 21 | Kaggle greenhouse climate prediction competition winning solution | WebSearch | 2026-10-01 | 9 | 없음 |
| 22 | nutrient solution temperature irrigation effect substrate temperature drip strawberry … | WebSearch | 2026-10-01 | 9 | greenhousemag(빈 페이지, 실패) |
| 23 | Autonomous Greenhouse Challenge 2019 data paper "GrodanSens" slab temperature columns | WebSearch | 2026-10-01 | 40(4회) | MDPI Sensors 2020(C40, 실패) |
| 24 | 2025 스마트농업 AI 경진대회 딸기 원격 재배 알고리즘 근권 온도 배지 | WebSearch | 2026-10-01 | 9 | 없음(기사 스니펫, C31) |
| 25 | 스마트팜 딸기 배지 온도 센서 설치 위치 매뉴얼 농촌진흥청 | WebSearch | 2026-10-01 | 9 | 없음(스니펫만) |
| 26 | growing pipe heating rail pipe position substrate gutter … "grow pipe" | WebSearch | 2026-10-01 | 9 | 없음(스니펫, C42) |
| 27 | 딸기 고설 베드 온수 배관 근권 난방 배지 온도 기온 차이 실증 | WebSearch | 2026-10-01 | 9 | 문종필 2016(C13), 문종필 2019(C14), 이태석 2021(C15), 이종원 2013(C16) |
| 28 | substrate temperature lag air temperature thermal mass daily amplitude … | WebSearch | 2026-10-01 | 9 | 없음(스니펫, C41) |
| 29 | ASHRAE Great Energy Predictor III 1st place solution … | WebSearch | 2026-10-01 | 10 | arXiv 2202.02898 초록, buds-lab 분석(C27), Kaggle 1위 글(C28, 실패) |
| 30 | DrivenData cold start energy forecasting winning solution … | WebSearch | 2026-10-01 | 10 | 수상자 블로그(C29) |
| 31 | open dataset Chinese solar greenhouse substrate temperature hourly figshare strawberry | WebSearch | 2026-10-01 | 10 | 없음(공개 배지 온도 데이터 못 찾음) |
| 32 | 4TU "Greenhouse data experiment drip irrigation 2016" variables … | WebSearch | 2026-10-01 | 9 | 없음(스니펫, C36) |
| 33 | irrigation water temperature cold drip lowers substrate temperature … | WebSearch | 2026-10-01 | 9 | 없음(관련 실측 없음) |
| 34 | 農研機構 施設園芸 環境データ オープンデータ 培地温度 … | WebSearch | 2026-10-01 | 9 | 없음(공개 배지 온도 데이터 못 찾음) |
| 35 | HortiDaily root temperature strawberry substrate heating tips … | WebSearch | 2026-10-01 | 10 | HortiDaily 2건(C32·C33), Hort Americas PDF(C34) |
| 36 | substrate sensor placement representative slab avoid ends … Priva Hoogendoorn | WebSearch | 2026-10-01 | 40(4회) | 없음(Priva·Hoogendoorn 공개 설치 지침 못 찾음) |
| 37 | 배지 온도 센서 위치 고설 베드 직사광 영향 측정 오차 딸기 스마트팜 | WebSearch | 2026-10-01 | 9 | 없음(스니펫만) |

---

## 2. 출처 표

| ID | 제목 | 작성자/기관 | 연도 | 유형 | URL | 언어 | 확인 수준 | 신뢰도 |
|---|---|---|---|---|---|---|---|---|
| C01 | Autonomous Greenhouse Challenge, First Edition (2018) | WUR / 4TU | 2020(공개) | 데이터셋 | https://data.4tu.nl/articles/_/12717758/1 | 영 | 검색 결과 목록만 | 높음(기관) |
| C02 | Autonomous Greenhouse Challenge, Second Edition (2019) | WUR / 4TU | 2020 | 데이터셋 | https://data.4tu.nl/articles/_/12764777/2 | 영 | 설명 페이지만(7z 8.4MB, 변수 목록 미확인) | 높음 |
| C03 | 4th Autonomous Greenhouse Challenge: Dwarf Tomato Timeseries and Images | WUR | 2024~25 | 데이터셋 | https://research.wur.nl/en/datasets/4th-autonomous-greenhouse-challenge-dwarf-tomato-timeseries-and-i/ | 영 | 설명 페이지만 | 높음 |
| C04 | Autonomous Greenhouse Cultivation of Dwarf Tomato: Performance Evaluation of Intelligent Algorithms for Multiple-Sensor Feedback | WUR 외 | 2025 | 데이터셋 논문 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12299141/ | 영 | 내용 확인(요약) | 높음 |
| C05 | Autonomous Greenhouse Challenge(AGC) - 2nd Edition (Kaggle 미러) | Kaggle 사용자 piantic | - | 데이터셋 | https://www.kaggle.com/datasets/piantic/autonomous-greenhouse-challengeagc-2nd-2019 | 영 | 접근 실패(제목만 반환) | 중 |
| C06 | Autonomous Greenhouses International Challenge underlines the importance of meaningful data | Grodan | ~2020 | 비공식(업체 블로그) | https://www.grodan.com/global/our-thinking/grodan-blogs/autonomous-greenhouses-international-challenge/ | 영 | 내용 확인(요약) | 중 |
| C07 | GroSens 2.2 Sensor Installation Guide | Grodan(ROCKWOOL) | 2023 | 매뉴얼 | https://www.grodan.com/syssiteassets/downloads/downloads-en/installation-guides-en/grodan_grosens2.2_sensor-installation-guide_en.pdf | 영 | 원문 확인(pypdf 추출, 10쪽 대부분이 그림) | 높음 |
| C08 | GroSens MultiSensor 설치 매뉴얼 / 브로슈어 | Grodan | 2019~24 | 매뉴얼 | https://www.grodan.com/nl/syssiteassets/downloads/downloads-nl/brochures-grodan-nl/downloads-multi-sensor-systeem/installation-manual-grodan-grosens-multisensor-system-122019.pdf | 영 | 검색 결과 목록만 | 높음 |
| C09 | Why Maintaining Cool Slab Temperatures Is Critical for Greenhouse Crops | Grower2Grower(NZ) | 연도 미상 | 비공식 | https://www.grower2grower.co.nz/why-maintaining-cool-slab-temperatures-is-critical-for-greenhouse-crops/ | 영 | 내용 확인(요약) | 낮음 |
| C10 | Root Zone Heating Systems for Greenhouses | Farm Energy eXtension | 연도 미상 | 교육자료 | https://farm-energy.extension.org/root-zone-heating-systems-for-greenhouses/ | 영 | 내용 확인(요약) | 중 |
| C11 | Root Zone Heating for Greenhouse Crops (ATTRA) | S. Diver, NCAT/ATTRA | 2002 | 교육자료 | https://ceac.arizona.edu/sites/default/files/Root%20zone%202.pdf | 영 | 원문 확인(pypdf) | 중 |
| C12 | Troubleshooting Guide – Acclima Substrate Sensors | Growlink | 연도 미상 | 매뉴얼 | https://www.growlink.com/troubleshooting-guide-time-domain-reflectometer-tdr-substrate-sensor | 영 | 내용 확인(요약) | 중 |
| C13 | 온수배관을 이용한 시설딸기 부분난방기술 개발 (한국농공학회논문집 58(5):71-79) | 문종필·강금춘·권진경·백이·이태석·오성식·남명현(국립농업과학원, 충남농기원) | 2016 | 실증 연구(실무) | https://www.koreascience.kr/article/JAKO201630762633051.pdf | 한 | 원문 확인(pypdf) | 높음 |
| C14 | 고설 딸기 관부 난방시스템의 에너지 절감 효과 (시설원예·식물공장 28(4):420-428) | 문종필·박석호·권진경·강연구·이재한·김형권(국립원예특작과학원, 국립농업과학원) | 2019 | 실증 연구(실무) | https://cdn.apub.kr/journalsite/sites/phpf/2019-028-04/KSBEC-28-4-420/KSBEC-28-4-420.pdf | 한 | 원문 확인(pypdf) | 높음 |
| C15 | 관부 난방시스템과 온수 양액 공급이 온실 에너지 사용량, 딸기 생육 및 생산성에 미치는 영향 분석 (생물환경조절학회지 30(4):271-277) | 이태석·김진구·박석호·이재한·문종필 | 2021 | 실증 연구 | https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002770766 | 한/영 | 설명 페이지만(초록, 요약 경유) | 중상 |
| C16 | 시설딸기 고설재배시스템의 베드 종류별 상토 내부온도 변화 (농공학회 학술대회 초록) | 이종원·나욱호 | 2013 | 학회 초록 | https://kiss.kstudy.com/Detail/Ar?key=3593823 | 한 | 설명 페이지만(초록, 요약 경유) | 중 |
| C17 | Ⅲ イチゴにおける環境制御ガイドライン | 愛知県(JAあいち経済連・愛知農総試) | 2018~19 | 교육자료(지침) | https://www.pref.aichi.jp/uploaded/attachment/412059.pdf | 일 | 원문 확인(pypdf, 크라운 가온 부분) | 높음 |
| C18 | 北海道における太陽光利用型の施設園芸導入マニュアル（いちご） | 北海道農政部·GB産業化設計 | 연도 미상 | 교육자료 | https://www.naro.go.jp/publicity_report/publication/files/Large-scale_facility_gardening_manual_Hokkaido.pdf | 일 | 원문 확인(키워드 부분만) | 중상 |
| C19 | イチゴの高設栽培における省エネ加温技術 | ALIC(野菜情報) | 2012 | 교육자료 | https://vegetable.alic.go.jp/yasaijoho/joho/1206_joho01.html | 일 | 내용 확인(요약) | 중 |
| C20 | イチゴ高設栽培（ピートベンチ栽培）の手引き | 奈良県農業技術センター | 2004 | 교육자료 | https://www.pref.nara.jp/secure/261432/itigokousetu0115.pdf | 일 | 접근 실패(인증서 만료), 검색 스니펫만 | 중 |
| C21 | いちご「べにたま」栽培マニュアル | 埼玉県農業技術研究センター | 2024 | 교육자료 | https://www.pref.saitama.lg.jp/documents/104573/benitamamanyuaru.pdf | 일 | 검색 결과 목록만 | 중 |
| C22 | Estimating the impact of environmental management on strawberry yield using publicly available agricultural data in South Korea | (PMC 논문) | 2023 | 데이터 활용 연구 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10183158/ | 영 | 내용 확인(요약) | 중상 |
| C23 | 지능형 수직농장 통합 데이터(딸기) | AI Hub(NIA) | 2021~ | 데이터셋 | https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=596 | 한 | 설명 페이지만 | 중 |
| C24 | 관부 난방 시스템 적용으로 인한 고설 딸기의 재배 환경 변화와 그에 따른 출뢰, 개화 및 수확량 비교 분석 | (KCI ART003013077) | 미상 | 실증 연구 | https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003013077 | 한 | 접근 실패(도구가 다른 논문 내용 반환), 검색 스니펫만 | 중 |
| C25 | 딸기 고설재배, 배수관 송풍에 의한 배지냉각 시스템(효고현 기술) | 농사로 | 미상 | 교육자료 | https://nongsaro.go.kr/portal/ps/psz/psza/contentSub.ps?cntntsNo=28867&menuId=PS00098 | 한 | 검색 결과 목록만 | 중 |
| C26 | 농업 궁금증 Q&A (딸기 배지 온도) | 월간원예 | 미상 | 비공식 | http://www.hortitimes.com/news/articleView.html?idxno=31644 | 한 | 접근 실패(인증서), 검색 스니펫만 | 낮음 |
| C27 | ashrae-great-energy-predictor-3-solution-analysis | BUDS Lab(NUS) | 2020~22 | 해법 분석 | https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis | 영 | 내용 확인(요약) | 중상 |
| C28 | ASHRAE GEPIII 1st Place Solution (Isamu & Matt) | Kaggle | 2019 | 해법 | https://www.kaggle.com/competitions/ashrae-energy-prediction/discussion/124709 | 영 | 접근 실패(제목만); 검색 스니펫만 | 중 |
| C29 | Meet the winners of Power Laws: Cold Start Energy Forecasting | DrivenData | 2018 | 대회·해법 | https://drivendata.co/blog/power-laws-cold-start-winners/ | 영 | 내용 확인(요약) | 중 |
| C30 | AI프렌즈 시즌 공공 데이터 활용 온도 추정 AI 경진대회 | DACON | 2020 | 대회 | https://dacon.io/en/competitions/official/235584/overview/rules | 한 | 검색 결과 목록만 | 중 |
| C31 | 2025 스마트농업 AI 경진대회(딸기 원격재배) 기사 | 뉴스핌 등 | 2026 | 대회(기사) | https://www.newspim.com/news/view/20260212001379 | 한 | 검색 결과 목록만 | 낮음 |
| C32 | Can substrate heating contribute to earlier strawberry production or more vegetative crops? | HortiDaily(HAS 학생 시험) | 미상 | 비공식(업계 기사) | https://www.hortidaily.com/article/9598403/ | 영 | 내용 확인(요약) | 낮음 |
| C33 | Taking a look inside a Dutch thermal berry greenhouse | HortiDaily | 미상 | 비공식 | https://www.hortidaily.com/article/9421082/ | 영 | 내용 확인(요약) | 낮음 |
| C34 | Growing strawberries (Hort Americas) | Hort Americas | 2019 | 업체 자료 | https://hortamericas.com/wp-content/uploads/2017/08/Hort-Americas-Growing-strawberries-042019.pdf | 영 | 원문 확인, 관련 내용 없음 | 낮음 |
| C35 | 육묘기 저온 처리가 '설향' 딸기의 화아분화 감응과 과실 수량에 미치는 영향 (JESI 33(10)) | 박수정 외 | 2024 | 연구 | https://journal.kenss.or.kr/xml/42344/42344.pdf | 한 | 원문 확인, 이 현상과 무관 | - |
| C36 | Greenhouse data experiment drip irrigation 2016 | 4TU(Wipfler 외) | 2020 | 데이터셋 | https://data.4tu.nl/articles/dataset/Greenhouse_data_experiment_drip_irrigation_2016/12708971 | 영 | 검색 결과 목록만(배지 온도 없음으로 보임) | 중 |
| C37 | Effect of root-zone heating using PTC film on growth and quality of strawberry in greenhouses (HEB) | (국내 연구진 추정) | 2021 | 연구 | https://link.springer.com/article/10.1007/s13580-021-00384-5 | 영 | 접근 실패(로그인 리다이렉트) | - |
| C38 | Effects of supplemental root-zone pipe heating systems on … strawberry … winter (NZJCHS 53(4)) | - | 2023 | 연구 | https://www.tandfonline.com/doi/abs/10.1080/01140671.2023.2224035 | 영 | 접근 실패(403) | - |
| C39 | Effect of Greenhouse Cladding Materials and Thermal Screen Configuration on Heating Energy and Strawberry Yield in Winter (Agronomy 11:2498) | - | 2021 | 연구 | https://www.mdpi.com/2073-4395/11/12/2498 | 영 | 접근 실패(403) | - |
| C40 | Cherry Tomato Production in Intelligent Greenhouses—Sensors and AI … (Sensors 20:6430) | Hemming 외 | 2020 | AGC2 논문 | https://www.mdpi.com/1424-8220/20/22/6430 | 영 | 접근 실패(403) | - |
| C41 | Temperature of substrates in relation to trough characteristics | (ResearchGate) | 미상 | 연구 | https://www.researchgate.net/publication/286853123 | 영 | 검색 결과 목록만 | 중 |
| C42 | Grow pipes – an energy advantage / Greenhouse heat distribution | Greenhouse Canada / UMass | 미상 | 교육자료 | https://www.greenhousecanada.com/grow-pipes-the-energy-advantage-20303/ | 영 | 검색 결과 목록만 | 중 |

---

## 3. 주장 → 흔적 표

"입력으로 확인·적용 가능?"은 평가 행 특징 규정(같은 온실의 현재·이전 입력만)을 기준으로 적음. "학습 데이터 점검 가능"은 표적(배지 온도·EC)을 써서 학습 구간에서만 확인할 수 있다는 뜻임.

| 출처ID | 관련 주장/기법 | 우리 데이터에서 보여야 할 흔적 또는 적용 방법 | 입력 변수로 확인·적용 가능 여부 |
|---|---|---|---|
| C13 | 연구 온실 한 동 안에 베드 4개를 두고 근권 가온/관부 가온/둘 다/무처리를 동시에 처리함. 축열조 온수 22~24℃를 24시간 순환(480 L/h). 근권 가온 베드는 밤 근권 18~21℃(주야 평균 약 20℃, 무처리보다 혹한기 +9.1℃·2월 +6.1℃). 무처리 베드는 밤 13~15℃ | '배지 > 공기' 분리일에는 배지 온도가 **18~21℃ 근처에서 거의 평평**하고, 하루 변동폭이 정상일보다 작으며, 외기가 −0℃든 +4℃든 수준이 비슷해야 함. 같은 온실 ID 안 '다른 구획' 기록이 이런 가온 베드라면 분리가 **구획 교대 경계와 맞물려** 시작·종료되어야 함 | 공기·난방 입력으로는 구분 불가(별도 회로). 학습 데이터 점검 가능: 분리일 배지 온도의 일 최소값 분포, 일 변동폭, 구획 경계와 분리 시작 시점의 일치 |
| C13 | 관부 가온만 한 베드도 근권 온도가 무처리보다 높음(관부 가온 베드의 근권이 근권 가온 베드보다 4~5℃ 낮다는 서술에서 역산) | 분리 크기가 두 계층(예: +3~5℃와 +6~9℃)으로 나뉠 수 있음 | 학습 데이터 점검 가능: 분리 크기 히스토그램이 여러 봉우리인지 |
| C14 | 3개 동(같은 시기, 같은 설계)에 공간 난방 설정 4/6/8℃ + 관부 가온 유무를 처리함. 관부 가온은 관부 옆 온도 센서로 순환 펌프를 켜고 끄며, **시간대별 설정(05~10시 15℃, 22~05시 13℃)**. 무처리 동의 관부 온도는 "온실 야간 기온과 거의 같음"(7~8℃). 가온 동은 공간 설정보다 3~8℃ 높음 | 정상일에는 배지 ≈ 공기여야 하고(우리 '정상 80~90%' 날과 일치), 가온일에는 밤 배지 온도에 **05시·22시 무렵 계단**, 켜고 끄는 톱니(1~2℃ 폭)가 보여야 함. 또 처리 동의 난방 복합 지표는 **오히려 낮게**(설정 4℃) 나타나므로 '난방 적은데 배지 따뜻' 조합이 나와야 함 | 시각 계단은 학습 데이터 점검 가능. '난방 지표 낮음 + 공기 낮음'은 입력으로 일부 확인 가능하지만 그것만으로 가온 여부를 판정할 수는 없음 |
| C14 | 옆 연동 온실 그늘 때문에 한 동이 겨울 아침 광량 부족 → 동 사이 차이 | 같은 외기 일사에서도 구획마다 아침 배지 상승 속도가 다름(구획 고정 차이) | 입력(실내 온도 아침 상승)으로 일부 확인 가능 |
| C15 | 관부 가온 동(실내 밤 5.7℃)의 베드가 대조 동(7.1℃)보다 약 2℃ 높음(12.7 vs 10.8℃). **온수 양액**을 낮에 공급하면 지하수보다 평균 8.7℃ 높고 베드는 약 5℃ 높아짐 | 낮 배지 온도가 **급액 시각(아침 첫 급액, 일사 비례 급액) 직후 계단식으로** 움직임. 반대로 찬 원수를 급액하는 구획/시기에는 '난방 100% 한낮 공기 16℃인데 배지 8℃'처럼 **낮에 배지가 공기보다 낮게 고정**될 수 있음. 이때 일사가 클수록(급액이 많을수록) 더 차가워져야 함 | 관수 입력 없음. 일사 적분은 입력에 있으므로 '배지<공기' 크기와 일사 적분의 관계는 학습 데이터로 점검 가능. 배지 EC 동반 변화(급액 직후 EC 변화)도 점검 가능 |
| C16 | 무가온일 때 근권은 야간에 실내보다 약 1.5℃, 주간에 약 4.3℃ 높음. 베드 재질(PE 타포린·플라스틱·스티로폼)에 따라 반응 속도가 다름 | 정상일에도 **구획(베드 종류)마다 고정 오프셋과 반응 지연이 다름**. 이어붙인 구획이 바뀌면 오프셋이 함께 바뀌어야 함 | 구획 식별이 입력에서 추정 가능하면 적용 가능. 학습 데이터 점검: 구획별 '배지−공기' 평균과 지연 |
| C17 | 크라운 가온은 35℃ 온수 연질 튜브로 크라운 표면 15~18℃를 유지. 기간은 **10월 말~2월 말**, 시설 최저 5℃(가온 시) 대 8℃(무가온). 2018년에는 **야간(20~08시)에만** 가온한 처리도 있었음. 처리구(가온/무가온/야간가온)를 같은 시설에서 비교함. 급액은 06시 강제 1회 + 외부 적산일사 4MJ/m²당 정량 | 분리일이 **계절 창(11~2월)에 몰리고**, 시작·종료가 날짜로 깨끗이 나뉨. 야간 가온형이면 분리가 **20시 전후 시작, 08시 전후 종료**로 하루 안에서 시간대가 갈림. 06시 급액 직후 배지 변화 | 날짜·시각은 입력이므로 계절·시각 구조는 확인 가능. 가온 여부 자체는 입력으로 알 수 없음 |
| C18 | 가온 촉성의 최저 조건은 "기온 12℃, 배지 온도 15℃"(기온과 배지를 따로 설정). 여름에는 배지 냉각으로 배지 온도를 낮춤 | 겨울 분리일 배지가 **15℃ 근처 바닥값**을 가질 수 있고, 여름에는 '배지 < 공기' 분리 | 학습 데이터 점검: 배지 일 최저가 특정 값(15℃ 등)에 쌓이는지 |
| C19 | 크라운 가온(전열선) 설정 21℃, 하우스 난방을 10℃→4℃로 낮춰도 같은 수량. 11월 상순~외기 최저 5℃ 이상인 시기까지 운전 | 처리구에서는 공기 난방은 낮고 근권·관부만 따뜻함 → 입력상 '난방 적음'이 배지 '따뜻함'과 같이 나올 수 있음 | 입력으로 반대 방향 증거만 볼 수 있음(학습 점검용) |
| C25 | 효고현: 배지 바닥 배수관에 송풍해 기화열로 배지를 최대 3℃ 낮춤(화아분화 촉진, 가을) | 가을(9~10월)에 '배지 < 공기' 분리가 하루 내내 지속. 송풍이 꺼지면 바로 원래 상태로 돌아감 | 송풍 입력이 없으면 불가. 유동팬 입력과는 다른 장치임(주의) |
| C10, C11 | 근권 가온은 백(bag)이나 상토에 꽂은 센서로 펌프를 켜고 끄며, 켜짐·꺼짐 간격은 1~2℃. 바닥 가온 사례에서 바닥 74°F(23℃)·군락 55°F(13℃)·1.2 m 높이 48°F(9℃)로 수직 차이가 약 10℃ | 가온일에는 배지 온도에 **1~2℃ 폭 톱니(히스테리시스)**. 공기 센서 높이에 따라 '배지−공기' 차이가 10℃까지 커질 수 있음 | 톱니는 1시간 해상도에서 흐려질 수 있음. 학습 데이터로 점검 가능 |
| C07 | 기후 센서는 직사광에서 최대 5℃ 오차가 날 수 있음. 근권 센서는 슬래브 높이에 맞춘 정렬판으로 설치함. 센서 위치는 '열 12, 오프셋 4'처럼 위치명으로 관리하도록 권장 | 일사 시간대에만 + 편차가 나오는 분리(직사광 영향)는 우리 현상(하루 내내·밤에도 지속)과 **모양이 다름** → 직사광 설명은 기각 쪽 근거. 위치명 관리 권장은 센서를 옮기면 기록이 다른 위치값으로 이어진다는 뜻 | 일사 입력으로 확인 가능(분리가 밤에도 유지되면 이 설명 탈락) |
| C12 | 탐침이 배지에 제대로 꽂히지 않거나 빠지면 함수율 0%, EC 0이 되지만 **온도는 계속 보고됨** | 센서가 빠진 날에는 배지 온도가 **공기 온도를 바짝 따라가야** 함(분리가 아니라 오히려 결합). 따라서 '공기와 수℃ 어긋나며 2~3일 지속'은 센서 이탈보다 다른 원인일 가능성이 큼. 단 **EC 표적이 같은 날 0 근처나 급변**이면 센서 문제 쪽 | 학습 데이터 점검 가능: 분리일과 EC 이상일의 동시 발생 비율(Codex의 EC 결과와 교차 확인 필요) |
| C09 | 슬래브 온도는 아침 일사·차광·관수 시점·난방관 사용에 좌우되고, 하루 중 열이 쌓임 | 정상일에는 배지 일 최고가 공기보다 늦게(오후) 나옴 → 지연 특징이 유효 | 입력으로 적용 가능(지연·누적 일사 특징) |
| C41 | 용기 속 배지는 토양보다 일교차가 약 10℃ 큼. 부피가 작을수록 변동이 큼 | 구획(용기·베드 크기)마다 진폭이 다름 → 진폭 차이로 구획을 구분할 수 있음 | 학습 데이터 점검 가능 |
| C04 | 4회 챌린지: 5분 간격, **화분 3개에만** 토양 센서(온도·유전율·EC) 설치, 난방관 온도도 기록. 데이터 지연과 센서 연결 끊김이 가끔 있었음 | 배지 온도가 소수 센서(1~3개)의 값이라면 센서 하나의 위치나 상태가 온실 전체 값을 정함 → 센서를 교체하거나 옮긴 날을 경계로 수준 이동 | 공개 데이터로 '배지 vs 공기 vs 난방관' 관계 사전 학습에 쓸 수 있음(미다운로드) |
| C02, C06, C40 | 2회 챌린지(방울토마토 6구획): GroSens로 슬래브 온도·EC·함수율, 일부 팀은 자체 중량·온도 센서까지 추가. 팀(구획)마다 관수·난방 전략이 다름 | (외부 데이터) 같은 외기에서 **구획 전략 차이만으로** 슬래브−공기 차이가 얼마나 벌어지는지 보는 기준으로 쓸 수 있음 | 데이터 다운로드 후 분석 필요(이번엔 안 함) |
| C22, C23 | 국내 공개 스마트팜 데이터는 농가마다 설치 센서가 다르고 기간 결측이 많음. 'Internal soil temperature'는 일부 농가에만 있음. AI Hub 수직농장 딸기에는 배지 온도·양액 온도·배지 EC가 함께 있음 | 양액 온도와 배지 온도를 같이 가진 공개 데이터(AI Hub)로 **'양액 온도 → 배지 온도' 결합 강도**를 외부에서 가늠할 수 있음 | 외부 분석용(우리 입력에는 양액 온도 없음) |
| C27, C28 | ASHRAE GEPIII 상위 해법: 이상 데이터 제거(상수 구간, 0, 급등락)가 순위를 가른 요인. **같은 사이트 여러 건물에서 같은 시각에 이상이 나면 진짜 이상으로 판단.** 건물·계기별 모델 | 우리 분리일이 **여러 온실 ID에서 같은 날짜에 동시 발생**하는지 점검 → 동시라면 운영 이벤트(난방 회로 일괄 가동, 측정 시스템 교체) 쪽. 학습에서 분리일을 빼거나 가중치를 낮추는 실험 | 학습 데이터 점검 가능. 학습 행 제거·가중은 규정상 허용 범위(평가 특징 규정과 무관) |
| C29 | Cold Start 3위: 건물 수준을 **짧은 과거 소비량의 최소·최대·평균으로 정규화**. 2위: 건물 과거 특징은 쓰지 않고 시간·기온·일정 특징 + 임베딩 | 그룹 수준을 맞히는 데 표적의 과거값을 쓰는 방법은 우리 규정상 **불가**(평가 행 특징은 입력만). 쓸 수 있는 것은 2위 방식(입력 기반 + 구획 임베딩)뿐 | 표적 기반 정규화는 적용 불가. 입력 기반 구획 표현은 적용 가능 |
| C30 | DACON 온도 추정: 짧은 기간만 있는 목표 지점을 비슷한 지점으로 사전학습한 뒤 미세조정(LSTM 전이학습) | 목표 센서의 일부 기간 값이 학습에 있어야 성립 → 평가 온실 2개가 학습에도 있는 우리 구조에서는 '온실별 미세조정'에 해당 | 온실 단위 미세조정은 가능(평가 온실 학습 행 사용). 그날의 분리를 맞히는 데는 도움이 안 됨 |

---

## 4. 아직 못 본 곳 / 다음에 조사할 곳

| 대상 | 이유 | 다음 행동 |
|---|---|---|
| AGC 2회 데이터 본체(4TU 7z, 8.4MB)의 GrodanSens·Irrigation·GreenhouseClimate 파일 | 6구획 × 같은 외기에서 슬래브 온도와 공기·난방관 온도의 관계를 **직접 계산**해 볼 수 있는 유일한 공개 연구 온실 데이터. 이번에는 다운로드하지 않음(파일 다운로드는 승인 필요) | 사용자 승인 후 다운로드 → '슬래브−공기' 일평균 분포, 구획 간 차이, 분리일(수℃·수일 지속) 존재 여부 확인 |
| AGC 4회 Timeseries.zip | 토양 온도 3개, 난방관 온도, 5분 간격 → '난방관 온도 ↔ 배지 온도' 결합 정도 | 위와 같음 |
| C40 Hemming 외 2020(Sensors), C38 NZJCHS 2023, C37 PTC 필름, C39 MDPI | 403·로그인으로 접근 실패. C38은 딸기 근권 파이프 가온의 배지·기온 실측이 있을 가능성이 큼 | 브라우저나 기관 접근으로 원문 확보 |
| C24 KCI 관부 난방 고설 딸기(ART003013077) | 도구가 엉뚱한 논문을 반환함. 스니펫상 야간 베드 +2.4℃ | KCI 원문 PDF를 직접 받기 |
| C20 奈良 피트벤치 수인, C26 월간원예 Q&A | 인증서 만료로 접근 실패 | 다른 경로(캐시, 미러) |
| 농촌진흥청 '스마트온실 매뉴얼-딸기', 농사로 스마트팜 센서 설치 기준 | 국내 배지 센서 설치 위치·개수 기준(베드 앞쪽 1세트라는 스니펫만 봄) | 농사로·농업과학도서관 PDF |
| Priva·Hoogendoorn·Ridder 설치/운전 매뉴얼(근권 가온 회로, 'grow pipe' 제어) | 공개 문서를 찾지 못함(제품 소개 페이지만 있음) | 업체 지식베이스·대리점 PDF(Hortispares 등) 검색 |
| METER(TEROS 12)·Delta-T(WET) 배지 센서 매뉴얼의 온도 측정·설치 깊이 | 이번 검색에서 직접 열지 못함 | 매뉴얼 PDF 확인 |
| AI Hub 지능형 수직농장(딸기)·지능형 스마트팜 통합(토마토·파프리카) 데이터 | 배지 온도와 양액 온도가 같이 있어 '양액 → 배지' 결합을 외부에서 가늠할 수 있음. 설명 페이지만 봄 | AI Hub 데이터 명세서(PDF) 확인. 데이터는 신청이 필요해 사용자 판단 |
| 국내 '스마트농업 AI 경진대회'(2021~2025) 예선 데이터·수상 해법 | 기사만 봄. 예선에 우수 농가 환경 데이터 예측 과제가 있었는지 미확인 | 대회 공식 페이지·발표 자료 |
| Kaggle ASHRAE 1위 원문(C28) | 페이지가 제목만 반환됨. 사이트 수준 이상 제거 절차의 세부 내용 미확인 | 1위 팀 GitHub 원문 |
| 중국·일본 공개 온실 시계열(배지 온도 포함) | 이번 검색으로 공개 데이터를 찾지 못함 | Science Data Bank(중국), 農研機構 データカタログ 검색 |
| 스마트팜코리아 '농촌진흥청 데이터셋' | 지시에 따라 원천 추적은 하지 않음(데이터마트는 원천 아님). 일반 변수 구성 참고로만 볼 수 있음 | 필요하면 변수 정의서만 열람 |
