import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,shutil,hashlib
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local/image_source_mapping_20261010_v1'
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
dirs=[ROOT/'집/코덱스/local'/n for n in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1','image_source_mapping_20261010_v1']]
before=sum(size(p) for p in dirs);new_before=size(LOCAL)
source=LOCAL/'development_group_contact_v1.jpg';b=source.read_bytes()
derived_before=size(LOCAL/'qa')+sum(p.stat().st_size for p in LOCAL.glob('*contact*.jpg'))
free_before=shutil.disk_usage(ROOT).free
if before+len(b)>12*1024**3 or new_before+len(b)>2*1024**3 or derived_before+len(b)>128*1024**2 or free_before-len(b)<30*1024**3:raise ValueError('common output budget gate')
target=LOCAL/'development_group_contact_v2.jpg'
with target.open('xb') as f:f.write(b)
assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(b).digest()
result={'cache_before_bytes':before,'cache_after_bytes':before+len(b),'cache_cap':12*1024**3,'new_cache_after_bytes':new_before+len(b),'new_cache_cap':2*1024**3,'derived_after_bytes':derived_before+len(b),'derived_cap':128*1024**2,'free_C_after_bytes':shutil.disk_usage(ROOT).free,'free_reserve':30*1024**3,'all_budget_checks_pass':True,'contact':str(target),'correction':'v1 contact omitted aggregate/free/derived writer guard; actual output was within limits; v2 copied through all guards','scope':'four current image task caches incl runtime wheels/tmp; source Drive cache occupancy not measured'}
assert result['free_C_after_bytes']>=result['free_reserve']
with (RUN/'storage_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
