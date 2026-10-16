# A2 실행 코드 및 C1/D 진단 독립 검토

검토 대상: A2 tokens_v1.py/sample_v1.py/train_v1.py/eval_support_v1.py/score_eval_v1.py/run_v1.sh, C1 train_eval_v2.py, D diagnostics/build_v1.py. 파일 수정 없이 정적 검토. GPU 실행 검증은 아직 아님.

## 결론
토큰 추출·마스크 정렬·OOF 선택·고정 판정식에 실행을 즉시 막는 계산 오류는 발견하지 못했다. 다만 평가 실행 전 baseline 검증의 조용한 생략을 차단하고, 계획에 명시한 GPT 참고 평가 누락을 보완해야 한다. 후보 계산 변경은 필요하지 않다.

## A2 확인
- 저장소 submit_img09/vit.py는 batch-first transformer 뒤 전체 [N,197,768]에 ln_post를 적용한다. 해당 hook에서 CLS를 제외하는 구현은 올바르다. conv flatten 순서와 label_cells의 reshape/mean/ravel은 행 우선 14x14 순서로 일치한다. 서버 vit.py도 같은 내용인지 SHA 확인 필요.
- RGB 정규화와 224/112 grid는 기존 inference.py와 같다. alignment는 hook 전후 CLS bitwise 동일과 두 합성 마스크 셀을 검사한다. 실제 사진과 마스크 크기 일치 assert가 있다.
- sample은 같은 source를 같은 group 폴드에 두고, 표본 선택은 점수에 의존하지 않는다. fold별 scaler/LR는 학습 token만 사용한다. 진짜 OOF는 각 사진의 held fold 모델로 전체 54x196 token max를 계산한다.
- threshold는 게이트 안 학습 진짜1800 OOF와 보정 진짜1395의 final 점수이다. h3j는 계수 학습에만 들어간다. rc 추가 문턱이 없다. max 결합과 필수 조건은 PLAN_v2와 일치한다.
- max-logit 후 sigmoid는 모든 token sigmoid의 max와 수학적으로 같다. affine 재정리의 부동소수점 차이는 있을 수 있으므로 패키지도 동일 계산을 사용하고 저장 점수 재현을 검사해야 한다.

## 평가 전 보완
1. A2와 C1 모두 baseline CSV가 없으면 if ref.is_file()로 검산을 조용히 생략한다. HERE.parents[1] 경로는 서버 codex 아래의 이전 실험 폴더를 가리켜 구조상 맞다. 하지만 예상 참조 파일 존재를 먼저 assert하고, file/source 또는 파일 순서/label까지 대조한 후 p9를 비교해야 한다. R5는 별도 NPZ 검산이 필수로 실행되나 순서 근거는 동결 manifest와 연결해야 한다.
2. A2 PLAN_v2는 처리 진짜 및 GPT류 참고 보고를 약속했으나 SETS에는 GPT류가 없다. 이미 노출된 GPT 참고 세트를 결과 열람 전 추가하거나 별도 고정 점수보고 실행을 등록한다. 이를 필수 판정으로 바꾸면 안 된다.
3. A2 샘플 자료 건수/종류는 manifest를 따르므로 학습 전 예상 h3real1800+h3j1800+pstyle501+v7393+h3s450과 실제 선택 수를 대조한다. 정확하지 않으면 원인 확인 후 진행하며, 결과를 보고 표집을 수정하지 않는다.
4. 샘플 단계는 매 사진 전체 token을 CPU로 복사한 뒤 일부만 보관한다. 누수나 계산 오류는 아니지만 누적 전송 비용이 크다. 본 계획의 실행 시간은 실제 첫200장 속도로 확인해야 하며, 새 backbone pass가 없다는 사실을 추가 비용0으로 설명하지 않는다.

## C1
- h3/h3j/G의 source 폴드 분리, Ef C=.5, H2식 진짜 OOF+미사용3종 θ풀, rc별 분위, 다른 전문가 유지 및 gate 밖 동일 검사는 적절하다.
- filter 손실은 base정답→candidate놓침 전이로 집계하므로 개선으로 상쇄하지 않는다. 모든 guard는 고정 기준과 일치한다.
- 첫fold JSON의 finite는 불리언 기록뿐이다. 중간 검토에서 true를 확인하고 진행해야 하며, 거짓이면 중단한다. 최종 단계에서는 전체 OOF finite assert가 존재한다.

## D1/D2 최종
build_v1.py는 SHA가 고정된 원행의 R2~R5 label1,p9<.5만 선택하고 dg1/gap finite 및 고정 경계값으로 found를 정의한다. 유형별 n 합181, found 합37, 37/181=.2044198895이므로 A2 분기가 정확하다. rc17은 별도 집계다. D2는 이전 표의 ② 행을 수치 변경 없이 복사하며 새 독립 평가로 해석하지 않는다. 저장 마스크 결측은 None→NA로 유지한다. 결과 JSON471행과 요약5줄 생성 구조는 요구에 부합한다.
