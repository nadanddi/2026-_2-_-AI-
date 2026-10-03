from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'전체비교표_v1.md').read_text(encoding='utf-8')
clean=source.replace('。','').replace('差(F47−F13)','차이(F47−F13)')
(HERE/'전체비교표_v2.md').write_text(clean,encoding='utf-8')
print('정리한 Markdown v2 생성, v1 보존')
