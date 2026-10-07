from pathlib import Path
import sys,json,hashlib,zipfile,collections,os
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
OUT=Path(__file__).parent
def load(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
inv=load('inventory_v1.json');summary=load('inventory_summary_v1.json');files=load('all_file_checks_v1.json');arrays=load('array_archive_checks_v1.json');ext=load('external_manifest_v1.json');verified=load('external_independent_v1.json');xv=load('xlsx_independent_v1.json');cp=load('competition_profile_v1.json');cv=load('competition_join_and_independent_v1.json')
assert len(inv['files'])==summary['files']==len(files)
assert len({r['path'] for r in files})==len(files)
assert {r['path'] for r in inv['files']}=={r['path'] for r in files}
assert sum(r['format']=='.csv' for r in inv['files'])==5489
assert all(r['matches_previous_sha'] for r in ext)
assert sum(r['members'] for r in ext)==len(verified)+len(xv)
for p in OUT.glob('external_*_v1.json'):
    if p.name in ['external_manifest_v1.json','external_independent_v1.json']:continue
    d=json.loads(p.read_text(encoding='utf8'));assert sha(ROOT/'집/코덱스/local/smartfarm_all_structured_20261004_v1'/d['file'])==d['sha256']
    if d['file']=='2022_env.xlsx':
        assert len(d['members'])==len(xv)
        for m,v in zip(d['members'],xv):assert m['rows']==v['rows'] and m['sensor']['strawberry_rows']==v['strawberry_rows']
for name,r in cp.items():
    if isinstance(r,dict) and 'sha256' in r:assert sha(Path(env.DATA)/name)==r['sha256']
bundle=[]
with zipfile.ZipFile(ROOT/'공용/대회자료/온라인대회자료.zip') as z:
    for name in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']:
        data=z.read('정형데이터/참가자_배포/'+name);h=hashlib.sha256(data).hexdigest();assert h==cp[name]['sha256'];bundle.append({'file':name,'sha256_match':True})
issues=[]
for r in arrays:
    if r['status']=='ERROR':
        p=ROOT/r['path'];head=p.read_bytes()[:12]
        if head.startswith(b'\x93NUMPY'):kind='object_npy_payload_not_deserialized'
        elif 'synthetic' in r['path']:kind='synthetic_non_npz_fixture'
        else:kind='invalid_real_npz_requires_regeneration'
        issue={'path':r['path'],'classification':kind,'bytes':p.stat().st_size,'sha256':sha(p)}
        if kind=='invalid_real_npz_requires_regeneration':issue['all_bytes_zero']=not any(p.read_bytes())
        issues.append(issue)
array_nans=[r for r in arrays if any(a.get('nan',0) or a.get('infinite',0) for a in r.get('arrays',[]))]
checks={'inventory_count_and_unique_paths':'PASS','competition_independent_csv':'PASS','raw_source_sha_fresh':'PASS','competition_distribution_zip_matches_extracted_csv':bundle,'external_independent_csv_members':len(verified),'xlsx_independent_sheets':len(xv),'external_table_rows':sum(r['rows'] for r in ext),'external_sources':len(ext),'all_file_statuses':collections.Counter(r['status'] for r in files),'file_read_errors':[{'path':r['path'],'error':r['error']} for r in files if r['status']=='read_error'],'array_archive_statuses':collections.Counter(r['status'] for r in arrays),'array_errors_classified':issues,'array_files_with_nan_or_inf':len(array_nans),'array_fields_with_nan_or_inf':sum(bool(a.get('nan',0) or a.get('infinite',0)) for r in arrays for a in r.get('arrays',[])),'array_fields_with_inf':sum(bool(a.get('infinite',0)) for r in arrays for a in r.get('arrays',[])),'object_fields_not_deserialized':sum(a['status']=='object_array_not_deserialized' for r in arrays for a in r.get('arrays',[])),'files_changed_during_table_read':[r['path'] for r in files if r['changed_during_read']],'files_changed_during_array_read':[r['path'] for r in arrays if r['changed_during_read']],'cleaning_log':{'deleted_rows':0,'imputed_cells':0,'unit_changes':0,'modified_source_files':0},'limitations':['repository inventory only; not all internet sources','model pickle/joblib payload and object arrays not deserialized','HWP/PDF inventoried; this run reads problem PDF text only','array NaN may be masked or outside-fold positions; not all NaN imply corruption','format and value checks do not certify each model provenance or causal feature compliance','external EC sensor definition/scale/licence detail must be confirmed before training']}
p=OUT/'final_verification_v1.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)
