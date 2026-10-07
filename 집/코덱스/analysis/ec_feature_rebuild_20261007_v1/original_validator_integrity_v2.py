"""Same complete-hour check with one grouped pass per original fold."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    source=(HERE/'original_validator_integrity_v1.py').read_text(encoding='utf-8')
    old="    for f,d in {key(r)[:2] for r in tr|va}:\n        assert {key(r)[2] for r in tr|va if key(r)[:2]==(f,d)}==set(range(24))"
    new="    hours=defaultdict(set)\n    for rid in tr|va:\n        f,d,h=key(rid);hours[f,d].add(h)\n    assert all(v==set(range(24)) for v in hours.values())"
    assert old in source
    source=source.replace(old,new).replace('original_validator_integrity_audit_v1.json','original_validator_integrity_audit_v2.json')
    exec(compile(source,str(Path(__file__).resolve()),'exec'),globals())
