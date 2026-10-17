# E_gpt 실행 전 등록 — 2026-10-16 13:42 KST

- 사용자 지정 단일 가설 검증. 나단9 + E_gpt 하나. 기존 N11(ppre 제거)와 무관한 새 폴더. 실패·15:00 지연이면 16시 건너뜀, 변형 없음.
- 입력 low38 + L/14 조각평균768; StandardScaler + balanced LogisticRegression(C=.5,max_iter=3000), 기존 H.lrfit 동일.
- 양성 im_set_gpt1/2/3/gptr 전부. 음성 h3 real + h3j real + gptr_src 전부. 유효 특징만 학습. 기존 split train/hold를 새 모델에서는 통합(사용자 전체4묶음 지시).
- LOGO 4회: 빠진 GPT 세트는 양성 학습에서 전부 제외. 그 세트의 source/group 및 확인 가능한 원본 계보와 겹치는 음성도 제외. scaler·모델·OOF·문턱 모두 남은 학습에서만 산정.
- theta: 학습 진짜 group5fold OOF + im66의 newreal/realg/realn 게이트안 유효 진짜. 99.75% 분위, rc 진짜는 99.9%. LOGO에서 빠진 원본과 겹치는 보정 진짜도 제외. 문턱 단계 조정 없음. g(s,theta)는 기존 H.g.
- 형식 게이트(나단9 Q92 4:2:0 JPEG) 안에서만 max(p9,g), rc에는 theta_rc. 밖은 원나단9 그대로.
- (a) LOGO 3/4 이상 각 새 가짜탐지>=1, 합>=5. 놓친 가짜는 고정 나단9 p<.5.
- (b) final fit: newreal,realg,realn,sealed_real/brown/bright,jig500 각각 FP증가<=.005. 전문가 합집합 진짜 FP <= N9+.001. 풀 정의 im66 동일: 학습진짜 OOF 게이트안 + newreal/realg/realn 유효 게이트안. N9도 학습진짜에 독립 OOF 적용(기존 im66 보존 OOF가 있으면 사용, 없으면 재계산). 최종 E_gpt OOF와 같은 행으로 비교. gptr_src 추가 진짜는 해당 N9 전문가의 학습계보를 확인해 OOF 평가.
- (c) pub21 21/21, R2~R5 각각 BA>=N9-.002. 모두 필요. 원시 배열·행키·분할 보존.
- 15:00 이전 전부통과일 때만 N9 복사에 모델/config/추론 연결을 추가한 새 N11 ZIP을 만든다. self_check·공개21, 15:40 전달. T4 시간은 실측 없으면 미확인 표시.
- 계획/중간/최종 독립 비평. 결과는 표+요약5줄. 서버 GPU0만; 다른 AI 폴더 기존파일 수정없음. 사용자 업로드만, 자동제출 없음.
