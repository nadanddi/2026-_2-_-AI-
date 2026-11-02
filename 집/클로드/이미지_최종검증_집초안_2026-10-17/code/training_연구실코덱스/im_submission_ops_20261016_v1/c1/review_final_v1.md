# C1 최종 독립 비평

결론: 고정 규칙상 C1 불통과이며 패키지/후속 수정 없이 종료하는 판단에 동의한다. R3 BA 비하락 조건을 충족하지 못했다. realg 개선이나 R4 개선으로 이를 상쇄하면 안 된다.

ownlocal result_artifacts_v1.tar를 추출/변경하지 않고 직접 열어17개 eval NPZ를 별도 코드로 재계산했다. 전 평가 candidate 유한성과 gate밖 base동일도 확인했다.

- newreal FP5→5, realg9→5, realn5→5: 합19→15,4장 감소로 첫 조건 통과.
- R2~R5 filter 정답→놓침은 R3에서1장, 나머지0. 허용≤2 조건 통과.
- R3 BA .9617017801731178→.9606402090478523: 감소하므로 필수 조건 실패.
- R4 BA .9467442335170443→.9481526842212696, R5 .9435589234387249 유지. R4진짜 FP9→8이 개선 원인이다.
- 공개21/21 유지. jig500 및 sealed real/brown/bright 신규진짜오탐은 각각0으로 보호조건 통과.
- 참고 area 진짜FP9→10은 별도 기록할 관찰이며, 사후 필수조건으로 추가하지 않는다.

verification_final_v1.json의 [real_fp_reduction=true,filter_losses=true,r345_ba=false,public21=true,real_guards=true] 및 internal_pass=false와 일치한다. 이번 검산은 저장배열에서 판정/산술을 재계산한 것으로 특징추출/학습을 새로 독립 재현한 것은 아니다.

realg/realn은 계수학습에는 없지만 θ보정에 쓰였으므로 해당FP 감소는 독립 일반화 증거가 아니다. 이번 고정 구성의 실패는 모든G_path음성보강 방식의 불가능성을 뜻하지 않는다. 실패후변형 금지와 나단9유지 규칙을 적용한다.
