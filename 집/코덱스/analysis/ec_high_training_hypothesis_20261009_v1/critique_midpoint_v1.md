# 고EC 가설 첫 폴드 독립 중간 비평 v1
2026-10-09 · 원본 run_v4 / 독립 verifier v5 · 아직 최종판정 아님

판정: **데이터·누수·계산 차단 문제 없음. 사전 계획대로 나머지9폴드 진행 승인.** 첫폴드 성능으로 성공/기각하거나 기준·선정을 변경하지 않는다.

## 실제 검산
verify_independent_v4는 기본 Python의 numpy 경로가 없어 수치계산 전에 중단. 기존 파일은 보존하고 공용 env bootstrap만 추가한 verify_independent_v5.py로 실행했다. 생산자/recipe import 없음. midpoint_independent_v5.json에 검산 PASS 저장.
- 검증912행/38일, 일반888행/37일, 고EC24행/1일. pass2 검증은 이번 폴드에 없음.
- 생산자96개 그룹의 RMSE·ET평활 RMSE·상수 RMSE·bias를 csv/fsum으로 독립 재계산, 모두2e-12 이내 일치. rawET 및 farm×pass 세분표를 보충.
- farm×day//5 paired bootstrap 20000회/동일RNG를 독립 재생, 세비교 p/CI 일치. 같은 16블록, seed평균예측 사용.
- 모든 source/cache SHA, baseline replay등록 SHA, 후보 train IDs와 삭제집합/24행삭제/농장×구간 control수/5NN self·인접금지/유한거리/중앙값 구조 확인.
- baseline ET replay maxdiff0, 기존 final baseline maxdiff2.22e-16. 손상PFN메타1개 대체검증은 preflight3의 제한 유지.

## 중간 수치와 반론
ensemble 일반RMSE: BASE0.1154501364, ALL0.1095920056, DISCORD0.1155800606, CONTROL0.1191504972.
ALL−BASE delta−0.0058581308/p_worse.0366, DISCORD−BASE+.0001299243/p.5796, DISCORD−CONTROL−.0035704366/p.0425. 사전alpha.008333에는 모두 미통과지만 한폴드이므로 최종판정하지 않는다.
고EC 단1일 RMSE는 BASE.4444215686→ALL.4771217645 손해, DISCORD.4355049586. ALL의 일반일 이득과 고EC 손해가 함께 있으므로 손해날을 채점에서 빼거나 ‘평가에 없을 것’이라고 해석하지 않는다.
일반raw ET RMSE는 BASE.1267250841→ALL.1127225725, DISCORD.1267391236. 선택적 삭제의 개선을 지금 주장할 근거가 없으며 CONTROL의 일반 손해도 target분포와 단일삭제샘플 교란을 고려한다.

## 이어서 확인할 것
최종360일에서 pass2 guard·각 seed 방향·farm×pass·고EC손해·161제외·normal fold분산 확인. source/selection/alpha 고정. 이미 노출된 DIAG진단이며 A/B 채택조건은 미실시. 결과가 좋아도 채택0, 평가 고EC 부재 입증불가.
