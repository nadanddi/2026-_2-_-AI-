import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import json, hashlib, tarfile, io, shutil, collections, argparse, os
RUN=Path(__file__).resolve().parent
LOCAL=ROOT/'집'/'코덱스'/'local'/'image_ingestion_20261010_v1'
DATA=Path(r'G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터\1.Training')
MAX_TL=4*1024**2

def save(name,obj):
    path=RUN/name
    if path.exists(): raise RuntimeError('immutable output exists: '+str(path))
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def inventory():
    rows=[]
    for dirname in ['라벨링데이터_add_20260807','원천데이터_add_20260807']:
        for p in sorted((DATA/dirname).iterdir()):
            if not p.is_file(): continue
            st=p.stat()
            rows.append(dict(path=str(p),name=p.name,kind='TL' if p.name.startswith('TL') else 'TS',cultivar='금실' if '금실' in p.name else '설향' if '설향' in p.name else 'unknown',bytes=st.st_size,mtime_ns=st.st_mtime_ns,attributes=getattr(st,'st_file_attributes',None),candidate=p.suffix=='.tar' and st.st_size>0,provider_completion='unknown',offline_availability='unknown'))
    totals=[]
    for key in sorted({(r['kind'],r['cultivar']) for r in rows}):
        chosen=[r for r in rows if (r['kind'],r['cultivar'])==key]
        totals.append(dict(kind=key[0],cultivar=key[1],directory_entries=len(chosen),positive_tar_candidates=sum(r['candidate'] for r in chosen),listed_bytes=sum(r['bytes'] for r in chosen),noncandidate_names=[r['name'] for r in chosen if not r['candidate']]))
    result={'scope':'directory metadata only; listed bytes not verified download bytes','rows':rows,'totals':totals,'c_free_bytes':shutil.disk_usage(ROOT).free}
    save('inventory_v1.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))

def flatten(obj,prefix=''):
    if isinstance(obj,dict):
        for k,v in obj.items(): yield from flatten(v,prefix+'.'+k if prefix else k)
    elif isinstance(obj,list):
        yield prefix+'[]','array_len='+str(len(obj))
    else: yield prefix,obj

def labels():
    LOCAL.mkdir(parents=True,exist_ok=True)
    all_records=[];reports=[]
    for name in ['TL_1.금실1.tar','TL_2.설향1.tar']:
        p=DATA/'라벨링데이터_add_20260807'/name
        before=p.stat()
        if before.st_size<=0 or before.st_size>=MAX_TL: raise RuntimeError('size gate')
        if list(p.parent.glob(name+'.*')): raise RuntimeError('temporary companion exists')
        if shutil.disk_usage(ROOT).free<30*1024**3: raise RuntimeError('space gate')
        with p.open('rb') as f:
            raw=f.read(before.st_size+1)
        after=p.stat()
        if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise RuntimeError('snapshot changed')
        if len(raw)%512 or raw[-1024:]!=bytes(1024): raise RuntimeError('TAR missing end blocks')
        target=LOCAL/name
        if target.exists(): raise RuntimeError('cache file exists')
        target.write_bytes(raw)
        seen=set(); values=collections.defaultdict(collections.Counter); files=0;names=[]; max_end=0
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:
            for m in tf:
                parts=Path(m.name.replace('\\','/')).parts
                if m.name.startswith(('/', '\\')) or '..' in parts or ':' in m.name or m.issym() or m.islnk(): raise RuntimeError('unsafe TAR member')
                if m.name in seen: raise RuntimeError('duplicate TAR member')
                seen.add(m.name)
                if len(seen)>5000: raise RuntimeError('member budget')
                end=m.offset_data+((m.size+511)//512)*512
                if end>len(raw)-1024: raise RuntimeError('truncated member')
                max_end=max(max_end,end)
                if m.sparse: raise RuntimeError("sparse TAR rejected")
                if m.isdir(): continue
                if not m.isfile() or not m.name.lower().endswith('.json') or m.size>1024**2: raise RuntimeError('unexpected member')
                data=tf.extractfile(m).read(m.size)
                obj=json.loads(data.decode('utf-8-sig'))
                all_records.append({'archive':name,'member':m.name,'content_sha256':hashlib.sha256(data).hexdigest(),'json':obj})
                for key,v in flatten(obj):
                    text=json.dumps(v,ensure_ascii=False)
                    if len(text)<200 and len(values[key])<1000: values[key][text]+=1
                files+=1;names.append(m.name)
        if any(raw[max_end:]): raise RuntimeError('nonzero bytes after member records')
        reports.append({'archive':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'json_files':files,'member_names_first':names[:4],'source_snapshot_stable':True,'tar_end_valid':True,'provider_completion':'unknown','verified_scope':'bounded readable local snapshot only; not entire dataset completion','fields':{k:{'distinct_values_up_to_cap':len(v),'examples':v.most_common(8)} for k,v in values.items()}})
    save('label_records_v1.json',all_records)
    save('label_profile_v1.json',reports)
    print(json.dumps(reports,ensure_ascii=False)[:14500])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['inventory','labels']);args=parser.parse_args()
    inventory() if args.mode=='inventory' else labels()

