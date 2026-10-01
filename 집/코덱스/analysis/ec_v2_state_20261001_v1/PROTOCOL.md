# EC 전용 v2_state_v1 — 단일 변경 및 전달 사전 고정

사용자 요청: 새 v2를 생성하고 EC만 예측해 팀원의 온도 모델과 결합할 자리를 채운다. EC CSV/재현 ZIP/한국어 설명자료 생성은 이번 요청으로 허용된다. 실제 플랫폼 제출은 하지 않는다. 산출물은 집/코덱스/local/ec_v2_state_20261001_v1/artifact/에 새 이름으로 둔다.

가설1개: 같은 온실의 예측일보다 앞선, 현재 fold 학습에서 사용 가능한 가장 최근 완료일의 EC 평균과 경과시간이 하루 EC 상태를 보완한다. 같은 출처임을 가정하지 않는다. 특징 두 개: prev_public_ec, prev_public_gap_hours. 완료일 전체 평균은 day<query_day만 허용한다. 자기일/미래/검증/잠금/±1 purge일 라벨은 접근 불가. 과거일이 없으면 두 특징 NaN, fold 안 median imputer로 처리한다. 온도 정답은 읽지 않는다.

모델 변경은 고정 후처리 단계의 ET 차이 교정 하나다. seed별 candidate = clip(final_v2 + 0.48*(final_ET40−final_ET38), train_y_min, train_y_max). 0.48은 기존 v2의 ET 비중 .8*.6으로 고정하며 튜닝하지 않는다. ET40=기존38열+과거EC2열. ET38/40은 동일 600trees, leaf1, seed7/101/2024. final_ET는 .5현재+.5당일까지예측평균 후 clip, final_v2는 기존 .8R3+.2TabPFN 후 동일 후처리. 이것은 raw ET를 pre-clip 혼합식에서 교체한 것과 clipping 경계에서 다르다. 후처리된 고정 v2의 변화량 모델로 정의하며 CV/최종예측에 똑같이 적용한다. 앙상블 평균은 seed별 candidate 평균이다.

DIAG10×10/A×5/B×5/EXT10/12 기존22fold 그대로, 모든 학습에서 잠금40일±1 및 검증일±1 purge. 이전 v2 예측캐시 SHA/row_id/라벨 일치 확인 후 사용. old ET 재학습 결과와 baseline ET 캐시 일치 확인. 기존v2평가예측은이번에동일레시피로새학습한다. 최종EC모델도잠금40일±1제외하므로잠금라벨을패키지에포함하지않는다.

실행전 main에 이 프로토콜/코드만 경로지정 커밋. 시드별/검증기별 방향, DIAG10 온실×5일block bootstrap20000 p_worse<.025(가설1), 95%차이구간 기록. A/B pooled발생단위와고유행평균점수구분. 이전자료재사용·미사용확인없음은명시. 전시드×전검증기개선,EXT무악화,DIAG기준미충족이면 '개선 검증 실패한 요청 생성 모델'이라고 표기한다. 사용자가생성을요청했으므로예측/재현물은생성하되채택으로바꾸지않는다. 최종잠금채점0,미사용시드확인미실시이면정식채택없음.

누수검사: source_day<query_day,source_key∈fold_train,외부온실/현재미래라벨변조불변,행순서불변,검증잠금행사용0. 기존 input causal/MASK 검사 재사용+새모듈검사. 평가CSV는sample_submission 순서 row_id/sub_ec1440행만. 재현ZIP은EC전용코드·필터된학습입력/EC라벨·평가입력·잠금목록·로컬TabPFN checkpoint·requirements·한국어문서·검증점수/해시를 포함. 깨끗한추출폴더에서다시전체예측을생성하고최대차이≤1e−8 및CSV해시/ID검사를수행한다. 원본산출물덮어쓰기금지.
