from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,datetime,collections
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
DEST=ROOT/'집/코덱스/local/work_history_backup_20261006_v1'
GROUPS=['ab_dynamic_ensemble_review_20261005_v1','ec_high_classifier_20261005_v1','ec_high_classifier_repair_20261005_v1','ec_hardcase_crossfit_20261005_v1','ec_actual_A_cases_20261006_v1','ec_actual_A_nested_oof_20261006_v1']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    DEST.mkdir(parents=True,exist_ok=False);selected=[]
    for group in GROUPS:
        for area in ['analysis','local']:
            base=ROOT/'집/코덱스'/area/group
            if not base.is_dir():continue
            for p in base.rglob('*'):
                if not p.is_file() or '__pycache__' in p.parts or '.staging' in p.parts or '.partial_' in p.name or p.name=='worker.lock':continue
                selected.append((str(p.relative_to(ROOT)).replace('\\','/'),p))
    for p in [ROOT/'집/코덱스/analysis/critic_workflow_20261006_v1.md',ROOT/'집/코덱스/작업일지/2026-10-06.md',ROOT/'공용/HANDOFF.md',ROOT/'공용/데이터_단서_카탈로그.md',H/'저장내용_v1.md',Path(__file__)]:
        selected.append((str(p.relative_to(ROOT)).replace('\\','/'),p))
    doc=Path('C:/Users/aozks/OneDrive/바탕 화면/A_B_모델_동적_앙상블_전략.docx')
    if doc.is_file():selected.append(('references/A_B_모델_동적_앙상블_전략.docx',doc))
    selected.sort();assert len({name for name,p in selected})==len(selected)
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    launch=json.loads((ROOT/'집/코덱스/analysis/ec_actual_A_nested_oof_20261006_v1/launch_v4.json').read_text(encoding='utf-8'))
    manifests=[];components=collections.defaultdict(set);csvs=[]
    zp=DEST/'work_snapshot_v1.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i,(name,p) in enumerate(selected):
            data=p.read_bytes();digest=hashlib.sha256(data).hexdigest();z.writestr(name,data);manifests.append(dict(path=name,bytes=len(data),sha256=digest))
            if 'ec_actual_A_nested_oof_20261006_v1/components/' in name and p.suffix=='.npz':
                v,k,j,kind,s=p.stem.split('_');components[(v,int(k),int(j))].add((kind,int(s)))
            if 'ec_actual_A_nested_oof_20261006_v1/' in name and p.name.startswith('OOF_') and p.suffix=='.csv':csvs.append(p.name)
            if i%50==0:print('ARCHIVING',i+1,len(selected),flush=True)
        full={('r3',s) for s in [7,101,2024]}|{('pfn',s) for s in [1,2,3,4]}
        record=dict(status='WORK_IN_PROGRESS_SNAPSHOT',created_kst=now,git_head=head,files=manifests,total_files=len(manifests),uncompressed_bytes=sum(x['bytes'] for x in manifests),nested_complete_contexts=sum(v==full for v in components.values()),nested_component_npz=sum(len(v) for v in components.values()),nested_oof_csv=csvs,worker=launch,excluded=['unfinished staging/partial files','worker.lock','unrelated other-AI files','Python environment and original TabPFN checkpoint'],limit='Snapshot only; training continues. Logs are captured bytes at read time.')
        payload=json.dumps(record,ensure_ascii=False,indent=2).encode('utf-8');z.writestr('snapshot_manifest_v1.json',payload)
    with (DEST/'snapshot_manifest_v1.json').open('xb') as f:f.write(payload)
    print('VERIFYING_ARCHIVE',len(manifests),flush=True)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and set(z.namelist())=={x['path'] for x in manifests}|{'snapshot_manifest_v1.json'}
        for x in manifests:
            h=hashlib.sha256()
            with z.open(x['path']) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            assert h.hexdigest()==x['sha256'],x['path']
        assert z.read('snapshot_manifest_v1.json')==payload
    proof=dict(status='PASS_LOCAL_SNAPSHOT',zip_path=str(zp.relative_to(ROOT)),zip_sha256=sha(zp),zip_bytes=zp.stat().st_size,manifest_sha256=sha(DEST/'snapshot_manifest_v1.json'),files=len(manifests),nested_complete_contexts=record['nested_complete_contexts'],nested_component_npz=record['nested_component_npz'],nested_oof_csv=len(csvs),created_kst=now)
    with (H/'snapshot_proof_v1.json').open('x',encoding='utf-8') as f:json.dump(proof,f,ensure_ascii=False,indent=2)
    print(json.dumps(proof,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
