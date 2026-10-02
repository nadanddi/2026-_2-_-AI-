# T-H1/T-H2 온도 잔차 수준·모양 분리 · 2026-10-03 실행 전 고정

- 사용자 '더 해봐' 직접 지시. 물리식/혼합비율 재조정이나 동일CB/ET 파라미터 재탐색이 아니라 학습 타깃 구조 변경2안. 기존6b.20/22/33/38은 잔차 사후예측/출처식별이고, 이번에는 기본 회귀기 자체를 level+shape로 학습한다. 모든유사연구부재주장없음.
- 원래 CODEX 물리Ridge(alpha100)/19열 그대로. fold 학습일 잔차 r=y-physics를 하루평균 L과 centered S=r-L로 나눈다. L은 하루1표본의 0시 입력으로 학습, S는 행단위 현재까지입력으로학습. 쿼리는자기0시입력/현재까지입력만. 평가정답/미래예측/미래입력사용0.
- daylevel 0시특징: original FEATURE_COLUMNS의각14센서_h0 + farm_id/day/second. 모델은온실표시를기존CODEX대로사용하며 외부메타추가0. h0부터현재까지가용. 평균타깃은학습정답으로만만들며 검증정답을daylevel특징으로넣지않음.
- T-H1: level LGB200/lr.03/leaf7/minchild20/L2=20/subsample.8 freq1/colsample.8; T-H2: level median+scaler+Ridge100. shape는공통LGB220/lr.035/leaf12/minchild100/L2=15,기존CODEX와동일. 시드726/727(기준BASE7/101짝). H2·물리·shape는샘플링없어seed가실질독립아닐수있음을명시,BASE/PFN변화별로판정.
- 후보 W30G=(.4+.1g)BASE+(.6-.4g)(physics+Lhat+Shat)+.3gPFN. CODEX만교체,BASE/PFN/g/비중고정. day정규화가저온외삽을망가뜨릴수있으므로EXT10/12전행판정필수.
- 기존 DIAG10/EXT10/EXT12 +/-1buffer, BASE/PFN캐시동일행확인. 2시드×2PFN문맥×3검증기=후보별12칸. 이전5안+이번2안 k7보수보정: 전12칸RMSE개선 및각DIAG p_worse<.025/7,99.285714% ΔMSE CI상한<0. 20k 온실5일블록bootstrap RNG20261003. fold표준편차·학습검증격차·온실/후반/시간대/저온분해서술.
- 추가 날씨guard: train_X 날씨4×24 zRMSE<=.05 연결성분(weather_groups.csv,정답없이split정의). 원DIAG검증일의weather group과공유하는학습일을±1buffer후추가제외. 쿼리변경0. 여기선CODEX/기존CB·ET/신규H1·H2 physics+잔차 **구성원**을동일guard로재학습비교. BASE/PFNcache는guard를거친것이아니므로guard결과를W30G전체검증점수로표현하지않음.
- guard는전검증방향선정용새혼합비중을만들지않는의존성스트레스진단이다. 원12칸이통과하면정식채택전에전체W30G의동일guard재학습검증까지이어간다(자동계속,제출물생성금지). 원12칸실패면그두안기각.
- weatheraudit 400일122group/491유사쌍·438완전날씨쌍,해당쌍중전체입력/온도정답완전복사0. 날씨공유=누수단정금지. unsupervised 전체train weather scaling은split정의용만·학습전처리아님. test날씨/라벨/EC최종잠금파일사용0.
- 기존input-only TF가중치globalrank는원모델고정조건으로유지. 새로운imputer/scaler/physics/level/shape는fold학습자료만fit. 정답복제·타깃인코딩·OOF메타라벨간접누수없음. code/기준main사전커밋.
