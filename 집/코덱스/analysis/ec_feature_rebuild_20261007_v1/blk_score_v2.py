"""Same predeclared BLK statistics, efficient ID filter; no altered candidate rules."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    code=(HERE/'blk_score_v1.py').read_text(encoding='utf-8')
    code=code.replace("    truth={}\n", "    truth={}\n    allowed_score_ids=set(ids)\n")
    code=code.replace("if row['row_id'] in set(ids):", "if row['row_id'] in allowed_score_ids:")
    exec(compile(code,str(Path(__file__).resolve()),'exec'),globals())
