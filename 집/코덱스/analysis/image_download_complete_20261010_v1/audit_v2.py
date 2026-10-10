import sys
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import json,csv,hashlib,collections,shutil,argparse,unicodedata
PREV=ROOT/'집'/'코덱스'/'analysis'/'image_ingestion_20261010_v1'
sys.path.insert(0,str(PREV))
from profile_all_labels_v1 import tar_records
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집'/'코덱스'/'local'/'image_download_complete_20261010_v1'
DATA=Path(r'G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터')
FIELDS=['farm_id','kind_type','crops_id','fname','date_captured','image_id','width','height','growth_stage']
FIRST=['VL_1.금실1.tar','VL_2.설향1.tar']

def write(name,obj):
    p=RUN/name
    if p.exists():raise RuntimeError('immutable output')
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def normal_name(name):return unicodedata.normalize('NFC',PurePosixPath(name.replace('\\','/')).name).casefold() if isinstance(name,str) and name.strip() else ''

def local_bytes():
    dirs=[ROOT/'집'/'코덱스'/'local'/n for n in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1']]
    return sum(p.stat().st_size for d in dirs if d.exists() for p in d.rglob('*') if p.is_file())

def inventory():
    paths=[('Training','TS','1.Training','원천데이터_add_20260807'),('Training','TL','1.Training','라벨링데이터_add_20260807'),('Validation','VS','2.Validation','원천데이터'),('Validation','VL','2.Validation','라벨링데이터')]
    rows=[]
    for split,kind,folder,sub in paths:
        for p in sorted((DATA/folder/sub).iterdir()):
            if not p.is_file():continue
            s=p.stat();rows.append({'split':split,'kind':kind,'path':str(p),'name':p.name,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'candidate':p.suffix=='.tar' and s.st_size>0})
    totals=[]
    for split,kind,_,_ in paths:
        r=[x for x in rows if x['split']==split and x['kind']==kind]
        totals.append({'split':split,'kind':kind,'files':len(r),'listed_bytes':sum(x['bytes'] for x in r),'temporary_or_zero':[x['name'] for x in r if not x['candidate']]})
    missing=[]
    for split,s,l in [('Training','TS','TL'),('Validation','VS','VL')]:
        source={r['name'][3:] for r in rows if r['kind']==s and r['candidate']};labels={r['name'][3:] for r in rows if r['kind']==l and r['candidate']}
        missing.append({'split':split,'source_without_label':sorted(source-labels),'label_without_source':sorted(labels-source)})
    result={'user_reported_download_complete':True,'scope':'directory metadata and paired archive names; raw payload not checked','totals':totals,'logical_listed_bytes':sum(r['bytes'] for r in rows),'pairs':missing,'rows':rows,'free_C_bytes':shutil.disk_usage(ROOT).free}
    write('inventory_v1.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))


def profile(stage):
    inv=json.loads((RUN/'inventory_v1.json').read_text(encoding='utf-8'))
    files=[x for x in inv['rows'] if x['kind']=='VL' and x['candidate'] and (stage=='full' or x['name'] in FIRST)]
    assert stage=='full' or len(files)==2
    if (RUN/f'validation_{stage}_v1.json').exists():raise RuntimeError('immutable output')
    LOCAL.mkdir(parents=True,exist_ok=True);rows=[];reports=[];issues=[];requests=0;new_read=0;invalid_identity=[];field_states={k:collections.Counter() for k in ['farm_id','kind_type','fname','image_id']}
    first_requested=json.loads((RUN/'validation_first_v1.json').read_text(encoding='utf-8'))['new_direct_requested_upper_bytes'] if stage=='full' else 0
    requests=first_requested
    for item in files:
        p=Path(item['path']);s=p.stat();target=LOCAL/p.name
        try:
            if s.st_size<=0 or s.st_size>=4*1024**2 or list(p.parent.glob(p.name+'.*')):raise ValueError('readiness gate')
            if shutil.disk_usage(ROOT).free-s.st_size<30*1024**3 or local_bytes()+s.st_size>10*1024**3:raise ValueError('space gate')
            if target.exists():
                previous=json.loads((RUN/'validation_first_v1.json').read_text(encoding='utf-8'))
                prior=next(x for x in previous['archives'] if x['archive']==p.name)
                if (s.st_size,s.st_mtime_ns)!=(prior['source_bytes'],prior['source_mtime_ns']):raise ValueError('changed source since first stage')
                raw=target.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=prior['sha256']:raise ValueError('changed local cache')
            else:
                requests+=s.st_size+1
                if requests>128*1024**2:raise ValueError('direct request budget')
                with p.open('rb') as f:raw=f.read(s.st_size+1)
                new_read+=len(raw);after=p.stat()
                if len(raw)!=s.st_size or (s.st_size,s.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('changed snapshot')
                target.write_bytes(raw)
            parsed=list(tar_records(raw));staged=[]
            for member,sha,obj in parsed:
                im=obj['images'];r={k:im.get(k) for k in FIELDS}
                invalid=[]
                for k in ['farm_id','kind_type','fname','image_id']:
                    value=im.get(k);state='absent' if k not in im else 'null' if value is None else 'blank' if isinstance(value,str) and not value.strip() else 'type_invalid' if (type(value) not in [str,int] if k=='image_id' else not isinstance(value,str)) else 'present'
                    field_states[k][state]+=1
                    if state!='present':invalid.append(k)
                if invalid:invalid_identity.append({'archive':p.name,'member':member,'invalid_fields':invalid})
                r['invalid_identity_fields']=json.dumps(invalid,ensure_ascii=False)
                r.update(archive=p.name,member=member,json_sha256=sha,prefix_candidate=str(im.get('fname','')).rsplit('_',1)[0],normalized_fname=normal_name(im.get('fname','')),prefix_meaning='unverified')
                staged.append(r)
            rows.extend(staged);reports.append({'archive':p.name,'source_bytes':s.st_size,'source_mtime_ns':s.st_mtime_ns,'sha256':hashlib.sha256(raw).hexdigest(),'JSON_rows':len(staged),'scope':'stable readable local label snapshot'})
        except Exception as e:issues.append({'archive':p.name,'error':str(e)})
    cross=collections.Counter((str(r['farm_id']),str(r['kind_type'])) for r in rows)
    result={'stage':stage,'archives':reports,'rows':len(rows),'issues':issues,'new_source_read_bytes':new_read,'new_direct_requested_upper_bytes':requests-first_requested,'cumulative_requested_upper_bytes':requests,'invalid_identity':invalid_identity,'identity_field_states':field_states,'farm_cultivar':[{'farm':k[0],'cultivar':k[1],'rows':n} for k,n in sorted(cross.items())],'scope':'VL labels only; VS raw images not opened','images_viewed':0}
    if stage=='full':
        training=list(csv.DictReader((PREV/'metadata_all_v1.csv').open(encoding='utf-8-sig')))
        train_names={normal_name(r['fname']) for r in training if normal_name(r['fname'])};train_ids={str(r['image_id']) for r in training if str(r['image_id']).strip()};train_composite={(str(r['farm_id']),str(r['kind_type']),str(r['image_id'])) for r in training if all(str(r[k]).strip() for k in ['farm_id','kind_type','image_id'])}
        train_prefix={(str(r['farm_id']),str(r['kind_type']),r['prefix_candidate']) for r in training}
        val_names=collections.Counter(r['normalized_fname'] for r in rows if 'fname' not in json.loads(r['invalid_identity_fields']));val_ids=collections.Counter(str(r['image_id']) for r in rows if 'image_id' not in json.loads(r['invalid_identity_fields']))
        val_composite=collections.Counter((str(r['farm_id']),str(r['kind_type']),str(r['image_id'])) for r in rows if not any(k in json.loads(r['invalid_identity_fields']) for k in ['farm_id','kind_type','image_id']))
        vp={(str(r['farm_id']),str(r['kind_type']),r['prefix_candidate']) for r in rows}
        result.update({'within_val_normalized_fname_duplicate_excess':sum(n-1 for n in val_names.values() if n>1),'within_val_global_id_duplicate_excess':sum(n-1 for n in val_ids.values() if n>1),'within_val_composite_id_duplicate_excess':sum(n-1 for n in val_composite.values() if n>1),'cross_shared_normalized_fnames':len(train_names&set(val_names)),'cross_shared_global_ids':len(train_ids&set(val_ids)),'cross_shared_composite_ids':len(train_composite&set(val_composite)),'cross_shared_prefix_candidate_groups':len(train_prefix&vp),'val_prefix_candidate_groups':len(vp),'new_val_farms':sorted({str(r['farm_id']) for r in rows}-{r['farm_id'] for r in training}),'prefix_shared_means':'possible group dependence; not proof of exact image duplicates','image_id_semantics_verified':False,'gold_farms_full_training_validation':sorted({str(r['farm_id']) for r in rows+training if r['kind_type']=='금실'})})
        fields=FIELDS+['archive','member','json_sha256','prefix_candidate','normalized_fname','prefix_meaning','invalid_identity_fields']
        with (RUN/'validation_metadata_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        split=json.loads((PREV/'split_reservation_v1.json').read_text(encoding='utf-8'))
        roles=[]
        for r in rows:
            farm=str(r['farm_id']);role='reserved_evaluation' if farm in split['reserved_evaluation_farms'] else 'public_dev_diagnostic' if farm==split['public_development_diagnostic_farm'] else 'development_CV' if farm in split['development_farm_folds'] else 'new_farm_unassigned'
            roles.append({'fname':r['fname'],'farm_id':farm,'kind_type':r['kind_type'],'source_split':'Validation','inherited_farm_role':role,'development_fold':split['development_farm_folds'].get(farm),'prefix_candidate':r['prefix_candidate']})
        result['inherited_role_counts']=dict(collections.Counter(r['inherited_farm_role'] for r in roles));result['independent_holdout_assumed']=False
        write('validation_farm_role_membership_v1.json',roles)
    write(f'validation_{stage}_v1.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='archives'},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','first','full']);a=p.parse_args();inventory() if a.stage=='inventory' else profile(a.stage)

