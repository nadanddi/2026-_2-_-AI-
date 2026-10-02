# F1 온실별 물리 기울기 · 실행 전 고정

사용자 추가탐색 지시. loss3안의 방향 불일치 뒤 별도 순차 가설. 기존 카탈로그/struct_temp/cold_v5/re22_physics_channels 검색에서 온실 더미절편·cold×heating·다른온실전이는 확인했으나 이 18개 물리기울기 부분공유 구체안은 확인하지 못했다. 모든유사연구 부재 주장은 하지 않는다.

원 CODEX Ridge19열은 온실절편만 다르고 공기/일사/난방과 reset입력 기울기는 공유한다. PHYSICS_COLUMNS의 farm_id를 제외한18열 각각 ×(farm_id−.5)를 추가해 두기록의 물리반응 기울기를 달리한다. 같은날 level 예측을 별도로 만들지 않고 관측 온도 전체를 원 pipeline으로 예측한다. 두기록이 여러출처를 섞은 합성이므로 실제온실고유 열용량 측정이라고 해석하지 않는다.

물리 median/scaler/Ridge100 및 후속 LGB220/.035/leaf12/minchild100/L2=15, MASK89특징/가중치 그대로. 추가되는 것은 물리18상호작용만. 원 W30G의 CODEX만 교체, 다른멤버/게이트/비중고정.

원 DIAG10/EXT10/EXT12 ±1일buffer, 726/727(BASE7/101), PFN1~8/17~24 후보12칸. 전칸RMSE개선 및각DIAG20k농장5일블록 p_worse<.025/11,99.545455% ΔMSE CI상한<0. RNG20261003. 누적11가설. 기존3loss안은최소alpha10에서도실패이며재채택하지않음. 원12칸통과시전체W30G 날씨GUARD추가검증 자동계속,그전정식채택없음.

학습전처리fold내fit,추가변환현재입력만, test평가예측/제출물/EC잠금0. 반복CV·독립홀드아웃아님,TF입력globalrank고정한계. rawlabel/혼합/fsum/bootstrap 및첫fold재학습 독립검산. source main사전커밋.
