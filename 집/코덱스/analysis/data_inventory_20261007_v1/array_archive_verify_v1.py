from pathlib import Path
import sys,json,zipfile,collections
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
OUT=Path(__file__).parent
records=json.loads((OUT/'inventory_v1.json').read_text(encoding='utf8'))['files']
results=[]
for r in records:
    p=ROOT/r['path'];e=p.suffix.lower()
    if e not in {'.zip','.npz','.npy'}:continue
    before=p.stat();q={'path':r['path'],'category':r['category'],'status':'PASS'}
    try:
        if e=='.zip':
            with zipfile.ZipFile(p) as z:
                bad=z.testzip();q.update(bad_member=bad,members=[{'name':i.filename,'bytes':i.file_size,'crc':i.CRC} for i in z.infolist()])
                if bad:q['status']='CRC_ERROR'
        else:
            q['arrays']=[]
            a=np.load(p,allow_pickle=False)
            pairs=[(n,lambda n=n:a[n]) for n in a.files] if e=='.npz' else [(p.name,lambda:a)]
            for name,get in pairs:
                item={'name':name}
                try:
                    arr=get();item.update(shape=arr.shape,dtype=str(arr.dtype),cells=arr.size)
                    if np.issubdtype(arr.dtype,np.number):item.update(nan=int(np.isnan(arr).sum()),infinite=int(np.isinf(arr).sum()))
                    item['status']='payload_read_ok'
                except ValueError as exc:
                    if 'Object arrays' in str(exc):item.update(status='object_array_not_deserialized',reason=str(exc))
                    else:raise
                q['arrays'].append(item)
            if hasattr(a,'close'):a.close()
    except Exception as exc:q.update(status='ERROR',error=repr(exc))
    after=p.stat();q['changed_during_read']=before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns;results.append(q)
    if len(results)%100==0:print('ARRAY_ARCHIVE',len(results),flush=True)
p=OUT/'array_archive_checks_v1.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
print('ARRAY_ARCHIVE_DONE',len(results),collections.Counter(r['status'] for r in results),flush=True)
