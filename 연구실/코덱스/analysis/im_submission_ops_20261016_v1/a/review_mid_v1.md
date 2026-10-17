# A2 첫 fit 중간 비평

결론: rest 진행 가능. 후보 성능은 아직 평가하지 않았으며 본 검토는 샘플/첫fit 무결성 확인이다.

- ownlocal samples_v1.npz를 직접 읽었다. X=(244224,768)이며 전부 유한하다. 파일은 정상적으로 끝까지 로드됐다.
- photos4944의 구성은 h3real1800/h3j1800/pstyle501/v7393/h3s450으로 계획과 일치한다. sampling 기록의 총 token244224/양성86016은 개별 사진 수 합산과 일치하고 양성0개인 가짜는 없다.
- fold별 사진수=[994,985,978,994,993]. fold[own]!=0으로 재계산한 첫 학습 token194624=양성68224+음성126400이며 first_fit_v1.json과 일치한다.
- fold0 모델의 mean/scale/coef768차원과 intercept1차원은 모두 유한하고 scale은 전부 양수다.
- alignment 기록은 원래 CLS bitwise동일, token54x196x768 및 합성 mask 행우선 검사를 통과했다. 앞선 CPU preflight source/group 분리/마스크 크기 검산 및 정적 코드 검토와 모순되지 않는다.

추가 모델/표집/문턱 변경 없이 나머지4fold와 finalfit을 진행할 수 있다. 이후 전체token OOF max, 고정 θ풀, baseline 필수대조 및 최종 신규오탐 조건을 확인한다. 첫fit 정상은 성능 통과나 패키지 제작 승인 판정이 아니다.
