from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'비교결과_v2.md').read_text(encoding='utf-8')
s=s.replace('**실패 네 날에서 높은 EC 학습지원이 늘어나는 경로를 확인했다.','**선정한 성공 대조날과 비교한 고정 ET 안에서, 실패 네 날의 높은 EC 학습지원이 늘어나는 경로를 확인했다.')
s=s.replace('조합별오지원과성공반례','조합별로 query 과대예측에 연결된 지원과 성공 반례')
s=s.replace('[보완 기록](보완기록_v1.md)','[보완 기록](보완기록_v1.md),[최종 보고서 비평 대조](최종보고서_비평대조_v1.md)')
p=H/'비교결과_v3.md';assert not p.exists();p.write_text(s,encoding='utf-8')
