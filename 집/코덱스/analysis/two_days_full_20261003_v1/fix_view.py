from pathlib import Path

HERE=Path(__file__).resolve().parent
s=(HERE/'전체비교표_v1.html').read_text(encoding='utf-8')
bad='document.querySelectorAll("a[href^="#"]")'
good='document.querySelectorAll("a[href^=\'#\']")'
assert s.count(bad)==1
(HERE/'전체비교표_v2.html').write_text(s.replace(bad,good),encoding='utf-8')
print('HTML v2 생성: 링크 selector 문자열 수정, 원 데이터와 v1 보존')
