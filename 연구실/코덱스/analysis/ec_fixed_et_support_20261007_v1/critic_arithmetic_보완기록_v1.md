# 독립 산술 검산기 보완 기록

2026-10-07 연구실 코덱스. 학습/모델worker·분할·시드·원결과·검산식은 변경하지 않았다.

`critic_arithmetic_v1.py` 최초 일반권한 실행은 기존 분석라이브러리 `six.py` 읽기 PermissionError로 import단계에서 종료했다. 모델 또는 결과 결함이 아니다. 같은 runtime과코드에 읽기검산범위의승격승인후 실행했다.

v1은각조합mask에서NPZ배열을반복조회해동일압축배열을재압축해제하는비효율이있었다. 검토자본인의검산session2563을중단했고v1코드는보존한다. v1완료PASS나전체검산수치는없다. 모델학습worker는중단하지않았다.

새 `critic_arithmetic_v2.py`는각coalition NPZ를한번만읽어dictionary에보관한다. 전수mask/fsum/평활식/고EC날지원/Shapley/비가산성/끝점/서명 검사식은동일하다. 출력은새 `critic_arithmetic_v2.json`으로분리했다. 환경설치나전역설정변경은없고,기존라이브러리읽기권한승인후실행했다.

이기록은검토자검산기의실패·성능보완출처이며원실험성능이나tree routing 감사판정을바꾸지않는다.
