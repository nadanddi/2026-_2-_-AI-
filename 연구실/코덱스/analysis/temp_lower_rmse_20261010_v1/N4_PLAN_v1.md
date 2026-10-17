# N4: 일사량 반응과 기존 온도 조건을 연결한 전체 조합 예비 검증

2026-10-10 연구실 코덱스. 실행 전 고정. 제출물 생성 없음.

## 근거와 한계
IRR4는 CODEX 구성원 DIAG10 -0.7864%, EXT10 +0.1351%, EXT12 -1.2388%였다. 추운 행에서의 불안정성과 9회차의 기존 8~10℃ gate를 연결한다. BASE에 일사량이 이미 있으므로 새 관측정보라는 주장은 하지 않는다. N2/N3 실패를 되돌려 채택하지 않는다.

후보 하나: `c_new=c+g*(c_irr-c)`, `g=clip((현재 in_temp-8)/2,0,1)` (결측1). 기준 `0.4*BASE+(0.2+0.4*(1-g))*c+0.4*g*PFN`, 후보의 c만 c_new로 교체. 농장별 선택, 추가 threshold, 결과를 본 계수 최적화 없음. 0~8℃에서는 기준과 완전히 같아야 한다. IRR4 계산/설정은 기존 등록 그대로. 물온도·멀칭·센서 위치 추정값을 만들지 않는다.

## 검증
공식 train_X/train_y의 sub_temp/test_X 및 같은 원ZIP checkpoint만. sub_ec 값은 읽지 않는다. 모든 보류·buffer·공식 test 입력은 TRAIN world에서 NaN. QUERY world는 같은 온실 현재·과거 입력만. query label은 특징 생성에서 제거한다. 가중치/season은 fold 학습행으로만 계산. day grouping은 원 day/t 사용, season 치환은 CODEX/PFN feature만.

DIAG10은 정렬된 각 농장 일자 순번 i에 `(i+2)//5 %10`, F47는 fold index에3 추가. 기존 일자와 겹치므로 완전히 새로운 데이터라고 하지 않는다. EXT10/12는 원검증 정의의 ph_in_temp_3 일별 최솟값 threshold 그대로. split_mask buffer1. Bootstrap 농장×raw day//5 block 20,000회, seed20261010. 여섯 후보 누적 보정 alpha=.025/6. 세 검증 모두 RMSE 감소, DIAG p_worse<alpha와 CI 상한<0 필요.

BASE는 원3seed 7/101/2024. CODEX seed726 및 IRR4 seed9109. PFN CPUv2 n_estimators4/context2000/신규seed49 **한 표본**. 이 결과는 9회차 원8표본의 성능이나 hidden RMSE를 입증하지 못한다. 통과하면 별도 등록으로 원8표본과 추가 실제 stochastic seed/검증배치 확인. 이 단계에서는 채택하지 않는다.

첫 fold 후 독립 중간 비평, full 후 최종 비평. 비평가 사용량 제한으로 계획 비평이 불가하면 신규 fit을 시작하지 않고 코드/재개기록만 보존. 원 재현 소수6자리 불일치도 유지.

## 실행 전 검사
원소스/입력/checkpoint hash 고정. 전체 모델 특징에서 query 미래 입력·다른 농장 변화와 TRAIN world 독립성 비교. query target perturbation에 frame 불변 확인. label 열 query frame에 없어야 함. 첫 fold 기준 CODEX 결과를 동일 입력 재실행으로 비교. 캐시는 recipe+source+row IDs+SHA 확인 후만 재사용. N4는 이전 OOF cache를 기준 예측으로 쓰지 않는다.
