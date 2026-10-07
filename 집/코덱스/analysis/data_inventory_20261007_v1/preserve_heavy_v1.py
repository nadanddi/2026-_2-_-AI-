from pathlib import Path
import json,hashlib,shutil
OUT=Path(__file__).parent;ROOT=OUT.parents[3];LOCAL=ROOT/'집/코덱스/local/data_inventory_20261007_v1'
LOCAL.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
records=[]
for name in ['all_file_checks_v1.json','all_file_checks_v2.json','array_archive_checks_v1.json']:
    src=OUT/name;dst=LOCAL/name
    if dst.exists():raise FileExistsError(dst)
    shutil.copy2(src,dst);left=sha(src);right=sha(dst);assert left==right
    records.append({'file':name,'bytes':src.stat().st_size,'sha256':left,'local_copy':str(dst),'match':True})
p=OUT/'heavy_preservation_v1.json';assert not p.exists();p.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
with (ROOT/'집/코덱스/작업일지/2026-10-07.md').open('a',encoding='utf8') as f:f.write('\n- 무거운 전체프로필3JSON은 own local/data_inventory_20261007_v1에 SHA검증 복제, 원본내용보존/analysis쪽복제본.gitignore 처리. sync_end의 home_codex_local Drive대상. proof heavy_preservation_v1.json.\n')
with (ROOT/'공용/HANDOFF.md').open('a',encoding='utf8') as f:f.write('- 전체프로필3JSON은 집/코덱스/local/data_inventory_20261007_v1에 SHA검증 복제해 Drive동기화대상으로 보존(analysis원본보존·gitignore), heavy_preservation_v1.json.\n')
print('HEAVY_PRESERVED',sum(r['bytes'] for r in records),flush=True)
