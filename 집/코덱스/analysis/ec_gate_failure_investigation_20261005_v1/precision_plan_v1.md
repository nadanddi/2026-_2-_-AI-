# MLP 수치 정밀도 감사 보완

원 verify_v1은 메타 MLP 일부의 gradient>=5e-4로 중단했다. 예측/손실 공식/스케일러/비용은 앞선 검사와 일치했다. 원270fit/원행/기존모델/수렴기준은 수정하지 않는다.

verify_v2는 모든 원행·분할·목적함수를 그대로 검산하고 허용gradient를 초과한 MLP를 명시적 최적화 flag로 남긴다. 이 flag를 PASS 학습최적성으로 취급하지 않는다.

독립 목적함수 CE+기존alpha10 L2의 analytic gradient를 201파라미터 finite difference로 검사한 뒤, 모든90 MLP 메타셀 중 비상수모델을 원계수에서 재최적화한다. 정규화/피처/라벨/학습행은 동일, L-BFGS maxiter5000/ftol1e-14/gtol1e-8/maxls50, success 및 maxgradient<=1e-6 요구. 정확도 개선 대조이며 새 제출/채택 후보가 아니다. 성능에 따라 일부 모델만 고르지 않는다.

외부(meta-valid) 정답은 새목적함수 fit에 들어가지 않는다. 원/정밀해 g 및 EC예측의 차이·RMSE·기울기·목적함수 감소를 전수 남긴다. optimizer 수렴이 진단의 주요 결론을 바꾸는지 확인한 뒤 결론을 낸다. LR/LGB의 원기준 검사는 유지한다. 실패로그와 v1은 보존한다.
