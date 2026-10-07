# 독립 경량검산 v1 실패 보존

2026-10-07 `critic_verify_support_identity_v1.py` 실행 exit1. 지원·trace 대조 후 raw-input identity 절의 `assert g[0]['in_co2']!=''`가 실패했다. 검토자가0h CO2가400일 모두 관측됐다고 추가 가정한 결함이다. root `input_identity_v1.py`는 `equal_nan=True`를 명시하므로 root 계산이나 모델 결함으로 판정하지 않는다.

실패 원코드를 보존하고 새 `critic_verify_support_identity_v2.py`에서 CO2 관측·결측 분모를 구별한다. 모델·결과·문턱·학습은 변경하지 않는다. v1을 PASS로 인용하지 않는다.
