# EC EXT12 내부 계절 변환의 정의역 보완

최초 prepare.py가 E_EXT12_0 내부 fit에서 한 온실의 후기 기준점을 모두 제외해 mapping의 len(x)>0 조건에 실패했다. 해당 폴드는 아직 모델 적합/점수 생성 전이다. 다른 33개 체크포인트는 보존한다.

prepare_v2.py: 날짜·온실 metadata만으로, 원outertrain에 후기 날짜가 있지만 innerfit에0개인 온실은 그 온실의 후기 날짜를 metaquery에서 제거하고 ±1buffer를 다시 적용한다. 목표값·모델성능으로 분할을 정하지 않는다. 다른 분할은 그대로다. 내부후기 표본이 줄어드는 한계를 명시한다. 계절 변환·후보식·시드·PFN문맥·판정기준 불변. 최종실행 분할 기록은 plan.json/gap_validation.json이며 초기 gap_audit.csv의 E_EXT12 inner는 보완 전 참고 기록이다.
