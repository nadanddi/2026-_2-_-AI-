# family20 마지막 실행 전 감사 해시 보완 · 2026-10-04

사전v3의 제목·본문에 섞인 외국어 단어만 한국어로 정정한다. 내용·식·문턱·실행 소스 변경은 없다. v1~v3 문서와 모든 기존 source/prepare를 보존한다.

실제 소스는 **run_v3.py**, 준비파일은 **preparation_v3.json**이다. 첫 감사 JSON의 수치/필수검사/status/signature 확인에 더해 정확한 파일 SHA를 cell metadata에 저장하고 재개 시 대조한다. 저장 순서는 cell CSV→첫 감사 JSON→첫 감사 SHA를 포함한 metadata다. 어느 부분에서 중단돼도 3artifact guard가 재개 적합을 거부한다. 준비v3와 실제 실행 직전 manifest/hash/runtime의 완전 일치도 유지한다.

모델·학습·문턱·채택식은 사전v1과 같다. 새 source/사전/prepare/정적검산을 main에 커밋한 뒤 최초 모델 fit을 실행한다. 이 문서 시점 fit/predict0.
