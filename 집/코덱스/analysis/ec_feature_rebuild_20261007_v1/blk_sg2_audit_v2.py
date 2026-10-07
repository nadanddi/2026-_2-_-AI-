"""Re-run source comparison on nonconstant hourly predictions using v2 adapter."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE/'blk_sg2_audit_v1.py').read_text(encoding='utf-8')
source=source.replace('from blk_sg2_refonly_v1 import *','from blk_sg2_refonly_v2 import *')
source=source.replace('baseline=np.full(len(ids),.7)','baseline=np.array([.65+.003*key(r)[2]+.0001*key(r)[1] for r in ids])')
source=source.replace("f'{f}_{d:03d}_{i:02d}':.7", "f'{f}_{d:03d}_{i:02d}':.65+.003*i+.0001*d")
source=source.replace('assert rawpass==.7','assert rawpass==.65+.003*h+.0001*d')
source=source.replace('BLK_SG2_refonly_audit_v1.json','BLK_SG2_refonly_audit_v2.json')
source=source.replace('constant_baseline','nonconstant_baseline')
source=source.replace('constant baseline audit only','nonconstant hourly baseline audit only')
source=source.replace("sha(HERE/'blk_sg2_refonly_v1.py')","{n:sha(HERE/n) for n in ['blk_sg2_refonly_v1.py','blk_sg2_refonly_v2.py']}")
exec(compile(source,str(Path(__file__).resolve()),'exec'),globals())
