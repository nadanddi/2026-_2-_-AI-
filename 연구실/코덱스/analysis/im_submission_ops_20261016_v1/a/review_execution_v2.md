# C1/A2 실행 코드 보완 확인

review_execution_v1.md의 필수 보완을 결과 열람 전에 재검토했다. 검토대상은 c1/train_eval_v3.py, a/score_eval_v2.py, a/references_v1.py이다. 결과는 아직 없다.

- C1 v3와 A2 v2는 baseline 참조 파일의 존재, 파일명/라벨/p9 대응을 필수 검사한다. R5는 y/kind/group/p9를 대조한다. 조용한 검산 생략이 제거됐다.
- A2 참고평가에 gpt1/gpt3/gptr hold/짝진짜 hold가 추가됐다. 특징/사진/라벨에 동일 hold 선택을 적용하며 기존 필수 판정조건은 변경하지 않았다.
- references_v1.py는21세트 특징/사진/라벨 대응을 후보 점수 생성 전에 검사한다. 부모 에이전트가 서버PASS를 전달했다.
- 부모 에이전트가 전달한 CPU preflight는4944사진/1344마스크 크기일치, source파생 동일fold, fold별양음성 존재 및 원모델SHA 보존PASS였다. 이는 아직 첫학습 결과나 후보 성능 통과가 아니다.

추가 필수 수정은 발견하지 못했다. 예정대로 C1 첫fold와 A2 첫fit까지만 실행하고 중간 비평으로 이어간다. 모델/문턱/필수 판정식의 사후 변경을 요청하지 않는다.
