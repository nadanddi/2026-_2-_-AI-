from pathlib import Path

HERE=Path(__file__).resolve().parent
s=(HERE/'전체비교표_v2.html').read_text(encoding='utf-8')
bad='</option value="입력">'
good='</option><option value="입력">'
assert s.count(bad)==1
(HERE/'전체비교표_v3.html').write_text(s.replace(bad,good),encoding='utf-8')
print('HTML v3 생성: 입력 필터의 option 시작 태그 복원, 원 데이터 보존')
