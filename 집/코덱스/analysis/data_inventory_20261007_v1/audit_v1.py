from pathlib import Path
import os, sys, json, csv, io, zipfile, hashlib, collections, time
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'집/클로드/research'))
import env
import pandas as pd
import numpy as np
OUT = Path(__file__).parent
SKIP = {'.git','.analysis-tools','.venv','venv','node_modules','__pycache__','tabdpt130_cpu_v1'}
EXT = {'.csv','.tsv','.xlsx','.xls','.parquet','.feather','.npz','.npy','.pkl','.pickle','.joblib','.zip','.json','.jsonl','.hwp','.pdf'}
def save(name, data):
    p=OUT/name
    if p.exists(): raise FileExistsError(p)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda v: v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def category(p):
    s=p.relative_to(ROOT).as_posix()
    if s.startswith('공용/대회자료/정형데이터/참가자_배포/'):return 'competition_raw'
    if 'smartfarm_all_structured_20261004_v1' in s and '/local/' in s:return 'public_external_raw'
    if s.startswith('공용/대회자료/'):return 'competition_other'
    if s.startswith('제출/'):return 'submission_artifact'
    if 'backup' in s or 'work_history' in s:return 'backup'
    if '/local/' in s or '/analysis/local/' in s:return 'derived_cache'
    return 'derived_or_document'
def inventory():
    records=[]; excluded=[]
    for base, dirs, files in os.walk(ROOT,followlinks=False):
        kept=[]
        for d in dirs:
            p=Path(base)/d
            if d in SKIP or p.is_junction() or p==OUT:
                excluded.append({'path':str(p.relative_to(ROOT)),'reason':'runtime/git/junction/audit output'})
            else:kept.append(d)
        dirs[:]=kept
        for name in files:
            p=Path(base)/name
            if p.suffix.lower() not in EXT:continue
            st=p.stat();records.append({'path':p.relative_to(ROOT).as_posix(),'format':p.suffix.lower(),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'category':category(p)})
    save('inventory_v1.json',{'files':records,'excluded_directories':excluded,'private_prior_competition':'not opened or copied','scope':'repository files; no whole-PC inventory'})
    counts=collections.Counter(r['category'] for r in records)
    formats={e:{'files':sum(r['format']==e for r in records),'bytes':sum(r['bytes'] for r in records if r['format']==e)} for e in sorted(EXT)}
    save('inventory_summary_v1.json',{'files':len(records),'bytes':sum(r['bytes'] for r in records),'categories':counts,'formats':formats})
    print(json.dumps({'files':len(records),'categories':counts,'formats':formats},ensure_ascii=False),flush=True)
def read_csv(source, sep=','):
    for encoding in ['utf-8-sig','cp949','utf-16']:
        try:
            if isinstance(source,bytes):return pd.read_csv(io.BytesIO(source),encoding=encoding,sep=sep,low_memory=False),encoding
            return pd.read_csv(source,encoding=encoding,sep=sep,low_memory=False),encoding
        except UnicodeError:continue
    raise ValueError('unknown encoding')
def frame_profile(df, detail=True):
    r={'rows':len(df),'columns':list(map(str,df.columns)),'dtypes':{str(c):str(t) for c,t in df.dtypes.items()},'missing':{str(c):int(v) for c,v in df.isna().sum().items()},'duplicate_rows':int(df.duplicated().sum()),'fields':[]}
    for c in df:
        x=df[c];p={'column':str(c),'unique':int(x.nunique(dropna=True)),'missing':int(x.isna().sum()),'top':{str(k):int(v) for k,v in x.value_counts(dropna=False).head(3).items()}}
        if pd.api.types.is_numeric_dtype(x):
            v=x.dropna();finite=v[np.isfinite(v)]
            p.update(nonfinite=int((~np.isfinite(v)).sum()),zero=int(v.eq(0).sum()),negative=int(v.lt(0).sum()))
            if len(finite):p.update(min=float(finite.min()),p01=float(finite.quantile(.01)),median=float(finite.median()),p99=float(finite.quantile(.99)),max=float(finite.max()))
        r['fields'].append(p)
    for c in ['row_id','ID','id']:
        if c in df:r['key_'+c]={'missing':int(df[c].isna().sum()),'duplicates':int(df[c].duplicated().sum())}
    if {'farm','day','hour'}.issubset(df):
        r['coordinate_duplicates']=int(df.duplicated(['farm','day','hour']).sum())
        r['farms']=df['farm'].value_counts().to_dict();r['days']=int(df[['farm','day']].drop_duplicates().shape[0])
    return r
def sensor_profile(df):
    crop=next((c for c in ['품목명','작물명','품목'] if c in df),None)
    item=next((c for c in ['장비명','항목명','측정항목'] if c in df),None)
    val=next((c for c in ['측정값','측정데이터','값'] if c in df),None)
    farm=next((c for c in ['농가id','농가ID','온실id','온실ID','시설ID'] if c in df),None)
    date=next((c for c in ['수집일자','측정일시','측정일자','수집일시','일시'] if c in df),None)
    sf=df
    if crop:
        ss=df[crop].astype(str);sf=df[ss.str.contains('딸기',na=False)|ss.isin(['80400','080400'])]
    r={'crop_column':crop,'item_column':item,'value_column':val,'farm_column':farm,'time_column':date,'strawberry_rows':len(sf) if crop else None,'farms':int(sf[farm].nunique()) if farm else None,'groups':[],'farm_item_times':[]}
    if date:
        dt=pd.to_datetime(sf[date],errors='coerce');r.update(start=str(dt.min()),end=str(dt.max()),invalid_time=int(dt.isna().sum()))
    if item and val:
        for name,g in sf.groupby(item,dropna=False):
            x=pd.to_numeric(g[val],errors='coerce');v=x.dropna();v=v[np.isfinite(v)]
            p={'item':str(name),'n':len(g),'missing_or_nonnumeric':int(x.isna().sum()),'nonfinite':int((x.notna()&~np.isfinite(x)).sum()),'zero':int(v.eq(0).sum()),'negative':int(v.lt(0).sum()),'unique':int(v.nunique()),'fractional':int((v-v.round()).abs().gt(1e-9).sum()),'units':list(map(str,g['단위'].drop_duplicates())) if '단위' in g else []}
            if len(v):p.update(min=float(v.min()),p01=float(v.quantile(.01)),median=float(v.median()),p99=float(v.quantile(.99)),max=float(v.max()))
            r['groups'].append(p)
        if farm and date:
            for (f,i),g in sf.groupby([farm,item],dropna=False):
                dt=pd.to_datetime(g[date],errors='coerce');valid=dt.dropna().sort_values();diff=valid.drop_duplicates().diff().dt.total_seconds().dropna()
                r['farm_item_times'].append({'farm':str(f),'item':str(i),'n':len(g),'invalid':int(dt.isna().sum()),'duplicate_times':int(valid.duplicated().sum()),'start':str(valid.min()),'end':str(valid.max()),'median_interval_seconds':float(diff.median()) if len(diff) else None,'gaps_over_1h':int(diff.gt(3600).sum()),'unique_values':int(g[val].nunique())})
    return r
def primary():
    data=Path(env.DATA);frames={};report={}
    for p in sorted(data.glob('*.csv')):
        df,enc=read_csv(p);frames[p.name]=df
        report[p.name]={'sha256':sha(p),'encoding':enc,**frame_profile(df)}
    x,y,t,s=(frames[k] for k in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv'])
    key=next(c for c in ['row_id','ID','id'] if c in x and c in y)
    report['alignment']={'key':key,'train_order_equal':x[key].equals(y[key]),'train_unmatched':len(set(x[key])-set(y[key])),'y_unmatched':len(set(y[key])-set(x[key])),'train_test_id_overlap':len(set(x[key])&set(t[key])),'test_submission_order_equal':t[key].equals(s[key]) if key in s else None,'train_test_column_equal':list(x)==list(t)}
    report['mask_pattern']={'test_all_missing_columns':[c for c in t if t[c].isna().all()],'test_part_missing_columns':[c for c in t if t[c].isna().any() and not t[c].isna().all()]}
    coord=x[key].astype(str).str.extract(r'^(.*)_(\d+)_(\d+)$');coord.columns=['farm','day','hour']
    if coord.notna().all().all():
        coord[['day','hour']]=coord[['day','hour']].astype(int)
        report['train_coordinates']={'farms':coord['farm'].value_counts().to_dict(),'farm_days':len(coord[['farm','day']].drop_duplicates()),'duplicates':int(coord.duplicated().sum()),'hour_counts':coord['hour'].value_counts().sort_index().to_dict(),'days_per_farm':coord.groupby('farm')['day'].nunique().to_dict(),'bad_day_hour_coverage':int(coord.groupby(['farm','day'])['hour'].nunique().ne(24).sum())}
    save('competition_profile_v1.json',report);print('PRIMARY_DONE '+json.dumps(report['alignment'],ensure_ascii=False),flush=True)
def external():
    raw=ROOT/'집/코덱스/local/smartfarm_all_structured_20261004_v1';expected=json.loads((ROOT/'집/코덱스/analysis/smartfarm_all_structured_20261004_v1/raw_manifest_v1.json').read_text(encoding='utf-8'))
    manifest=[]
    for p in sorted(raw.iterdir()):
        if not p.is_file():continue
        h=sha(p);e=next((q for q in expected if q['file']==p.name),None)
        result={'file':p.name,'sha256':h,'matches_previous_sha':bool(e and e['sha256']==h),'bytes':p.stat().st_size,'members':[]}
        if p.suffix=='.zip':
            with zipfile.ZipFile(p) as z:
                result['zip_bad_member']=z.testzip();result['member_manifest']=[{'name':i.filename,'bytes':i.file_size} for i in z.infolist()]
                for i in z.infolist():
                    if i.filename.lower().endswith('.csv'):
                        df,enc=read_csv(z.read(i));result['members'].append({'member':i.filename,'encoding':enc,**frame_profile(df),'sensor':sensor_profile(df)})
        elif p.suffix=='.xlsx':
            book=pd.ExcelFile(p)
            for sheet in book.sheet_names:
                df=pd.read_excel(book,sheet_name=sheet);result['members'].append({'member':sheet,**frame_profile(df),'sensor':sensor_profile(df)})
        else:result['status']='metadata_only_document'
        save('external_'+p.stem.replace(' ','_')+'_v1.json',result)
        manifest.append({'file':p.name,'sha256':h,'matches_previous_sha':result['matches_previous_sha'],'members':len(result['members']),'rows':sum(m['rows'] for m in result['members'])})
        print('EXTERNAL_DONE '+json.dumps(manifest[-1],ensure_ascii=False),flush=True)
    save('external_manifest_v1.json',manifest)
def derived():
    records=json.loads((OUT/'inventory_v1.json').read_text(encoding='utf-8'))['files'];results=[]
    for k,r in enumerate(records):
        p=ROOT/r['path'];e=p.suffix.lower();q={**r,'status':'metadata_only'}
        before=p.stat()
        try:
            if e in {'.csv','.tsv'}:
                df,encoding=read_csv(p, '\t' if e=='.tsv' else ',');q.update(status='full_table_profile',encoding=encoding,**frame_profile(df))
            elif e=='.npz':
                with zipfile.ZipFile(p) as z:
                    arrays=[]
                    for i in z.infolist():
                        with z.open(i) as f:
                            ver=np.lib.format.read_magic(f)
                            shape,fortran,dtype=np.lib.format._read_array_header(f,ver)
                            arrays.append({'name':i.filename,'shape':shape,'dtype':str(dtype),'object':dtype.hasobject,'bytes':i.file_size})
                    q.update(status='array_headers_only_no_pickle',arrays=arrays)
            elif e=='.npy':
                with p.open('rb') as f:
                    ver=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,ver)
                    q.update(status='array_header_only_no_pickle',shape=shape,dtype=str(dtype))
            elif e=='.json':
                obj=json.loads(p.read_text(encoding='utf-8-sig'));q.update(status='json_parse_ok',root_type=type(obj).__name__,length=len(obj) if hasattr(obj,'__len__') else None)
            elif e=='.zip':
                with zipfile.ZipFile(p) as z:q.update(status='zip_directory_read_ok',members=len(z.infolist()),uncompressed_bytes=sum(i.file_size for i in z.infolist()),table_members=[i.filename for i in z.infolist() if i.filename.lower().endswith(('.csv','.xlsx','.parquet'))])
            elif e in {'.pkl','.pickle','.joblib'}:q['status']='serialized_model_not_deserialized'
            elif e in {'.parquet','.feather','.xlsx','.xls'} and category(p)!='public_external_raw':
                if e=='.parquet':df=pd.read_parquet(p)
                elif e=='.feather':df=pd.read_feather(p)
                else:
                    book=pd.ExcelFile(p);q['sheets']=[]
                    for sheet in book.sheet_names:q['sheets'].append({'sheet':sheet,**frame_profile(pd.read_excel(book,sheet_name=sheet))})
                    df=None
                if df is not None:q.update(**frame_profile(df))
                q['status']='full_table_profile'
        except Exception as exc:q.update(status='read_error',error=repr(exc))
        after=p.stat();q['changed_during_read']=before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns
        results.append(q)
        if k%100==0:print('INVENTORY_CHECK',k,'/',len(records),flush=True)
    save('all_file_checks_v1.json',results)
    save('file_check_summary_v1.json',{'files':len(results),'statuses':collections.Counter(r['status'] for r in results),'changed':[r['path'] for r in results if r['changed_during_read']],'errors':[{'path':r['path'],'error':r['error']} for r in results if r['status']=='read_error']})
    print('ALL_FILE_CHECKS_DONE',flush=True)
if __name__=='__main__':
    globals()[sys.argv[1]]()
