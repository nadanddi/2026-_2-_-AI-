# Float32 attention 동등 연산 확인

원 패키지가 Torch GQA를 사용하면 float32가 math SDPA로 내려갈 수 있다. KV head를 명시적으로 repeat_interleave한 동일 attention 식은 efficient SDPA를 이용할 수 있다. dtype/n_estimators/표본/모델/날짜/가중치 변경 없음. 패키지 파일은 수정하지 않고 프로세스 내부에서 gqa_is_supported만 False로 연결한다.

통계 점수로 연산 옵션을 고르지 않는다. 기존 첫fold/시드1의 BASE·계절 예측과 maxdiff <= 1e-4℃, 각 팔 시간이 기존 약21초보다 작고 메모리 오류가 없을 때만 새 worker 버전에서 같은 float32 동등 연산을 사용할 수 있다. 실수 연산 오차를 보고하고 채택 가능 후보는 원 CPU 실행 재현을 여전히 요구한다. 기존 paired 체크포인트를 보존하며 최종 캐시 비교에서 각 fold의 차이를 기록한다. 조건 불만족 시 원 worker 계속.
