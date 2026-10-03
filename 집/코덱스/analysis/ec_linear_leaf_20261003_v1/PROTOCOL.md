# EC 전체 특징의 상수 잎/선형 잎 비교 — 실행 전 고정

2026-10-03 집 코덱스. 지속목표 6·7번째 후보 GB_CONST20/GB_LINEAR20. 기존 온도 linear_tree 실험은 기각됐고 EC 현재 계절v2의 FULL 특징으로 동일설정 대조는 카탈로그에서 확인되지 않았다. 과거 실패를 일반 불가능성으로 해석하지 않는다.

동기: 순위 정보는 충분한데 크기 추정이 부족하고, 작은 inner표본으로 만든 수준보정은 고EC에 반대 보정을 학습했다. 보정함수·표본축소 대신 허용 전체학습행으로 piecewise linear leaf를 직접 학습하는 계산방식 변경. 일반leaf 대조를 함께 실행하며 차이는 linear_tree bool과 해당정규화뿐이다. [LightGBM 공식 parameter 문서](https://lightgbm.readthedocs.io/en/v4.5.0/Parameters.html)의 linear_tree/linear_lambda 설명 확인. Local Linear Forests의 완전재현이라고 부르지 않는다.

- 기존 MASK FULL38/day대신season, farm 추가없음. 동일22fold×3seed(7/101/2024). trainfold median imputer keep_empty_features=True 및 StandardScaler를 두안에 동일 적용.
- LGBMRegressor objective regression,400iterations,lr.035,leaves15,min_child_samples100,reg_lambda15,max_bin127,subsample.8/freq1,colsample.8,deterministic=True/force_col_wise=True,n_jobs2,seed별random_state. CONST linear_tree=False;LINEAR True/linear_lambda10. 검증set early stopping 없음.
- raw새모델 예측→기존 causal shrink 한 번→candidate=clip(.8*현재계절v2+.2*new_shrunk,학습EC범위). 동일비중으로확정,시험후비중조정없음. PFN재학습없음.
- family7(앞5안+CONST/LINEAR). 각안전15칸개선,DIAG온실층화5기록일 bootstrap20,000 p<.025/7 및99.285714% CI상한<0. 작은평균개선만으로채택하지않음. 공개통과전EL1미채점,소모final lock미사용;최종채택에는EL1·재현감사필수.
- 학습/전처리fit IDs buffer 감사,첫fold 각안seed7 독립재학습,모델확인 linear leaves 유무,raw/shrink/혼합식scalar 및score fsums/독립 bootstrap 검산. 저온·후반·고EC/일반일·농장별 보고.
- query자신의현재·이전MASK정보만;다른온실query/미래/외기없음. supervised training labels는공개OOF라벨만.
- ownlocal/ec_linear_leaf_20261003_v1/GB_CONST20,GB_LINEAR20 체크포인트,코드해시동일때만재사용. 새원시EC/잠금읽기/제출/test예측없음.
