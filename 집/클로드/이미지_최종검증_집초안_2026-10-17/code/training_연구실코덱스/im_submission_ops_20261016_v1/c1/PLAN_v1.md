# C1 사전 실행 등록 v1 — 2026-10-16

## 단일 변경
나단9 E_f만재학습하고E_f θ/θ_rc만H2절차로재보정. 다른3전문가/기본판별기/원본대조/형식게이트불변. 학습negative에h3원본진짜1800의G_path판1개씩만추가; N_path변형없음/실패후수정없음.

## G_path와계보
im_h_raw manifest htrain 원본1800을 h3real source와정확대조. target긴변pool은기존im28의GPT200장치수메타순서그대로(생성사진/평가realg데이터아님), 사전에SHA고정. rng=Random(crc32('C1G|'+source)),긴변LANCZOS→PNG무손실왕복→기존dist_cover(rng보간)→JPEGQ92/420. 원본source/group/tar기록,원본/배포사진SHA보존. 원본이목표보다작으면확대도가능하며항상축소라하지않음.
realg/realn자체및그source는추가음성학습에미사용. h3raw와realg/realn exactsource 비중복assert. 같은촬영묶음은겹칠수있어group독립확증주장안함.

## 특징·학습
기존im43의나단6고정모델로G판emb/low/rc추출(모델변경없음),개인서버폴더에저장. GPU0무료상태확인/nadan/키인증만. 원본은읽기만.
H2그대로 h3+h3j+새G를묶음5fold로분리,원본·파생동일fold검사. E_f계수학습만저수준38+L14emb768,StandardScaler+LR C=.5 balanced max_iter3000. positive=기존filter_jpeg/blur/noise,negative=real(원래+jig+G). partialfake학습미사용. 모든단계예측유한성검사.
θ풀=모든학습진짜OOF+newreal/realg/realn고정새Ef배포점수. θ=99.75%분위,rc진짜θrc=99.9%. h3j게이트밖도원H2절차처럼θ풀에포함. realg/realn은계수학습미사용이나기존H2문턱보정에는사용함(독립평가아님).

## 판정(사용자 그대로)
1. newreal+realg+realn 최종FP합이baseline보다3장이상감소。
2. R2~5 filter_*에서base정답→candidate놓침의손실행수합≤2(순증감으로상쇄하지않는엄격계산,net도참고보고)。
3. R3/R4/R5각BA≥N9。
4. 공개21/21,jig500·sealed real/brown/bright 신규FP모두0。
전부PASS때만나단10세트제작·self_check·Colab요청。하나라도FAIL이면종료/후속변형없음。R2~5노출탐색자료/확증채택판정아님。

## 시간·검증
C1패키지+self_check 11:30미완료면12시슬롯스킵。전단계계획비평/첫fold중간비평/최종비평。원본Ef계산p9와기존고정캐시전행대조,교체외3전문가배열SHA불변/형식게이트밖불변、独立배열집계후통과만package。提出は사용자。
