# BLK 기준선 SG2 적용 범위 추가 비평 — 2026-10-07

critique_BLK_v2.md 후 부모의 추가 요청에 따른 별도 기록이다. 신규fit·채점 없이 관련 소스를 확인했다.

**P1: 'SG2 포함 baseline'이라는 이름만으로 처리 동등성을 정의할 수 없다.** 실제 배포 submission14 sg2post.py:128 및 연구 SG2 correction:127은 day<179에서 보정을 건너뛴다. WT2 소스는 머리 설명에 TM111 전체행에 SG2를 적용한다고 명시하며 별도 correct 구현을 사용한다. 따라서 WT2 저장 sg4 baseline과 배포 SG2의 적용 범위가 다르다. BLK v2의1440query는 모두 원기록pass1이므로 배포 gate를 그대로 적용하면 SG2는 비활성이다.

원본 numeric day를 가짜pass2 숫자로 바꾸는 조치는 권하지 않는다. 일차·계절 mapping·앞뒤 record 선택·참조 후보·블록 길이까지 바뀌며 기초 데이터가 다른 의미를 갖게 된다. SG2 activation만 바꾸려는 실험과 달력 전체를 바꾸는 실험을 섞게 된다.

fit 전에 다음2 baseline 명세를 분리할 것을 권한다.

- **BLK_RAW_PASS:** 원본 row_id/day/pass 유지, 배포 SG2의 day>=179 적용 gate 유지. 이 BLK에서는 SG2 비활성임을 명시한다. 이는 원배포 정책의 구조검증 대조다.
- **BLK_QUERY_ROLE:** 원본 메타데이터·train 참조 store·달력 fit은 유지하고, 평가형 query라는 별도 interface flag로 SG2 적용범위를 정의한 실험 baseline. 그 flag는값/정답/전체query통계로학습하지 않는다. WT2 correct와 같은지, 배포 correct에서gate만달라졌는지 source/단계출력으로 명시한다. 이를 '원배포와동일baseline'이라고부르지않는다.

가능하면 BLK3후보의 주 비교 baseline 하나를 규칙 등록 전에 고정하고, 다른baseline은 적용범위 진단으로 둔다. 두 baseline 모두에서3후보를 평가·선별하면6variant의선택을장부에반영한다. baseline 두개를 잘나온것으로 나중에고르면 안 된다. CPU 구성원/R3/PFN 문맥·shrink·clip·SG2 입력/출력/activation은 각각별도감사하며 gate활성행·실제변경행·참조가능률을 보고한다.

query-role로 gate를 켜더라도 완전pass2 BLK를 확보한 것이 아니다. 원기록pass1의 달력·서명·학습이웃 밀도·출처구조가 남아 있으므로 실제2차 평가 일반화는 기존 P2LOO/EL1 등에서 따로 확인해야 한다. 초기 BLK 실패/성공만으로 과거 사슬 단서 실패원인, 입력 정보부재, 원14변수 한계를 확정하지 않는다.

최우선 다음단계는 **적용범위가명시된BLK baseline와3방법규칙등록→baseline전체처리재현/인과감사→실행**이다. baseline gate와pass2미가용한계를PROGRESS/보고서에 남긴다.
