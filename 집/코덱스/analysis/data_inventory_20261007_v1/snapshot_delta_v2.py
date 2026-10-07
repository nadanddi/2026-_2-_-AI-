from pathlib import Path
import sys,json,collections,zipfile,ast
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd
import numpy as np
import audit_v1 as old
OUT=Path(__file__).parent
def save(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf8')
old.save=lambda n,v:save(n.replace('_v1','_v2'),v)
old.inventory()
one=json.loads((OUT/'inventory_v1.json').read_text(encoding='utf8'));two=json.loads((OUT/'inventory_v2.json').read_text(encoding='utf8'));first={r['path']:r for r in one['files']};second={r['path']:r for r in two['files']}
added=[r for p,r in second.items() if p not in first];changed=[r for p,r in second.items() if p in first and (r['bytes'],r['mtime_ns'])!=(first[p]['bytes'],first[p]['mtime_ns'])];removed=[p for p in first if p not in second]
tree=ast.parse((OUT/'repair_checks_v2.py').read_text(encoding='utf8'));fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='profile');exec(compile(ast.Module(body=[fun],type_ignores=[]),'<trusted-profile-v2>','exec'))
results=[]
for r in added+changed:
    p=ROOT/r['path'];q={**r,'status':'metadata_only'}
    try:
        if p.suffix=='.csv':
            df,enc=old.read_csv(p);q.update(status='full_table_profile',**profile(df))
        elif p.suffix=='.json':
            obj=json.loads(p.read_text(encoding='utf8-sig'));q.update(status='json_parse_ok',root_type=type(obj).__name__)
        elif p.suffix in ['.npz','.npy']:
            a=np.load(p,allow_pickle=False);q['arrays']=[]
            pairs=[(n,lambda n=n:a[n]) for n in a.files] if p.suffix=='.npz' else [(p.name,lambda:a)]
            for n,get in pairs:
                try:
                    arr=get();d={'name':n,'shape':arr.shape,'dtype':str(arr.dtype)}
                    if np.issubdtype(arr.dtype,np.number):d.update(nan=int(np.isnan(arr).sum()),infinite=int(np.isinf(arr).sum()))
                    q['arrays'].append(d)
                except ValueError as exc:q['arrays'].append({'name':n,'error':str(exc)})
            if hasattr(a,'close'):a.close()
            q['status']='array_payload_checked'
        elif p.suffix=='.zip':
            with zipfile.ZipFile(p) as z:q.update(status='archive_crc_checked',bad_member=z.testzip())
    except Exception as exc:q.update(status='read_error',error=repr(exc))
    results.append(q)
save('snapshot_delta_v2.json',{'initial_files':len(first),'final_snapshot_files':len(second),'added':added,'changed':changed,'removed':removed,'checks':results,'source_changes':[r['path'] for r in added+changed if r['category'] in ['competition_raw','public_external_raw']]})
print('DELTA_DONE',len(added),len(changed),len(removed),flush=True)
