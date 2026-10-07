from pathlib import Path
out=Path(__file__).parent
source=(out/'supplementary_v1.py').read_text(encoding='utf8').replace("encoding='utf8-sig'","encoding='utf-8-sig'")
exec(compile(source,str(out/'supplementary_v1.py')+' [v2 UTF8 BOM codec correction]','exec'))
