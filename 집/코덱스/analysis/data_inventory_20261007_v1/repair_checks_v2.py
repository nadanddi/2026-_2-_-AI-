from pathlib import Path
import sys,json,collections
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd
import numpy as np
import audit_v1 as original
OUT=Path(__file__).parent
def profile(df):
    r={'rows':len(df),'columns':list(map(str,df.columns)),'dtypes':{str(c):str(t) for c,t in df.dtypes.items()},'missing':{str(c):int(v) for c,v in df.isna().sum().items()},'duplicate_rows':int(df.duplicated().sum()),'fields':[]}
    for c in df:
        x=df[c];p={'column':str(c),'unique':int(x.nunique(dropna=True)),'missing':int(x.isna().sum()),'top':{str(k):int(v) for k,v in x.value_counts(dropna=False).head(3).items()}}
        if pd.api.types.is_bool_dtype(x):
            p.update(true=int(x.fillna(False).sum()),false=int((x==False).sum()),kind='boolean_category_not_numeric_quantile')
        elif pd.api.types.is_numeric_dtype(x):
            v=x.dropna();finite=v[np.isfinite(v)];p.update(nonfinite=int((~np.isfinite(v)).sum()),zero=int(v.eq(0).sum()),negative=int(v.lt(0).sum()))
            if len(finite):p.update(min=float(finite.min()),p01=float(finite.quantile(.01)),median=float(finite.median()),p99=float(finite.quantile(.99)),max=float(finite.max()))
        r['fields'].append(p)
    for c in ['row_id','ID','id']:
        if c in df:r['key_'+c]={'missing':int(df[c].isna().sum()),'duplicates':int(df[c].duplicated().sum())}
    if {'farm','day','hour'}.issubset(df):r['coordinate_duplicates']=int(df.duplicated(['farm','day','hour']).sum())
    return r
checks=json.loads((OUT/'all_file_checks_v1.json').read_text(encoding='utf8'));arrays={r['path']:r for r in json.loads((OUT/'array_archive_checks_v1.json').read_text(encoding='utf8'))};repairs=[]
for r in checks:
    if r['status']!='read_error':continue
    p=ROOT/r['path'];e=p.suffix.lower()
    try:
        if e in ['.csv','.tsv'] and 'numpy boolean subtract' in r['error']:
            df,enc=original.read_csv(p,'\t' if e=='.tsv' else ',');r.update(**profile(df),encoding=enc,status='full_table_profile');r.pop('error');repairs.append({'path':r['path'],'fix':'boolean categorical counts instead of invalid quantile'})
        elif e in ['.npz','.npy'] and r['path'] in arrays:
            a=arrays[r['path']]
            if a['status']=='PASS':r.update(status='array_full_payload_checked',arrays=a.get('arrays',[]));r.pop('error');repairs.append({'path':r['path'],'fix':'public np.load payload check replaces removed private numpy API'})
            elif e=='.npy' and a['error'].startswith("ValueError('Object arrays"):
                with p.open('rb') as f:
                    ver=np.lib.format.read_magic(f)
                    reader=np.lib.format.read_array_header_1_0 if ver==(1,0) else np.lib.format.read_array_header_2_0
                    shape,order,dtype=reader(f)
                r.update(status='object_array_metadata_only',shape=shape,dtype=str(dtype));r.pop('error')
            else:r['error']=a['error']
    except Exception as exc:r['repair_error']=repr(exc)
    if len(repairs)%200==0:print('REPAIR',len(repairs),flush=True)
for n,d in [('all_file_checks_v2.json',checks),('repair_log_v2.json',repairs),('file_check_summary_v2.json',{'files':len(checks),'statuses':collections.Counter(r['status'] for r in checks),'errors':[{'path':r['path'],'error':r['error']} for r in checks if r['status']=='read_error']})]:
    p=OUT/n
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
print('REPAIR_DONE',len(repairs),collections.Counter(r['status'] for r in checks),flush=True)
