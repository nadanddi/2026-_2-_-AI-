from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'비교결과_v1.md').read_text(encoding='utf-8')
s=s.replace('독립 최종 비평과 산술 감사는 진행 중이다.','원모델 재현, 지원·경로 감사, 독립 산술 검산을 완료했다. 최종 혹독비평을 함께 읽는다.')
s=s.replace('19,456조합','18,432조합')
s=s.replace('공통모델은자료가적어다른support풀을쓴다. 원모델대비season도달라질수있고46개비계절열동일성만확인했다.','공통 모델은 자료가 적어 다른 학습지원 풀을 쓴다. 이번 대상·대조 8날의 query season은 실제로 원모델과 공통 모델에서 모두 같았다(192행의 row_id별 season 값 한 가지, preparation의 common_season_changes=[]). 46개 비계절 관측 특징도 결측을 포함해 동일하다. 다만 각 모델의 학습 미디언은 다를 수 있어 결측 대체 후 모델 입력까지 모두 동일하다는 뜻은 아니다. 공통 세 시드끼리는 학습행·float32 학습입력·정답·미디언이 정확히 같다.')
s=s.replace('- 끝점지원·분기추가감사와독립산술·혹독비평은진행중이며완료보고서에서연결한다.','''- [끝점 지원·분기 감사](audit_endpoint_v2.json): 6모델·40끝점의 시간별 잎 지원을 별도 cohort 산술로 재계산해 PASS. 새 전체 tree 감사는 반복하지 않고 기존 tree receipt의 SHA를 확인했다.
- [공유prefix·공통자료 감사](audit_prefix_v1.json): 24,000 first-divergence 기록 중 갈라지는 23,670 경로의 모든 공유prefix/양값/선택자식 재현, 공통3seed 학습입력·정답·미디언 동일, 총18,432조합 PASS.
- [독립 산술 감사](critic_arithmetic_v2.json): run/core/helper 직접 호출 없이 20쌍·18,432mask의 지원합·정답가중합·prefix·고EC학습날 정의·Shapley·양문맥 비가산성·날지원 CSV·SHA 재계산 PASS. 최대 rawcurve 차6.44e−15, Shapley 차5.00e−16. tree 재routing은 수행하지 않았다.
- [최종 혹독비평](최종혹독비평_v1.md),[보완 기록](보완기록_v1.md). 초안의 총조합수 오기는 새v2에서 정정했고 모델·결과 배열은 변경하지 않았다.''')
p=H/'비교결과_v2.md';assert not p.exists();p.write_text(s,encoding='utf-8')
