from pathlib import Path
import sys, io, zipfile, json, csv, collections, hashlib, shutil
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd
import numpy as np
OUT=Path(__file__).parent
RAW=ROOT/'집/코덱스/local/smartfarm_all_structured_20261004_v1'
RAW.mkdir(parents=True,exist_ok=True)
results=[]
for filename in ['2024_data (1).zip','2022_11_ds.zip','2022_env.xlsx']:
    src=Path('C:/Users/aozks/Downloads')/filename
    if not src.exists():continue
    dest=RAW/filename
    if not dest.exists():shutil.copy2(src,dest)
    assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest()
    if src.suffix=='.zip':
        with zipfile.ZipFile(src) as z:
            members=[(i.filename,z.read(i)) for i in z.infolist() if i.filename.endswith('.csv')]
    else:
        book=pd.ExcelFile(src)
        members=[(sheet,pd.read_excel(book,sheet_name=sheet)) for sheet in book.sheet_names]
    for name,data in members:
        if isinstance(data,bytes):
            for enc in ['utf-8-sig','cp949']:
                try:txt=data.decode(enc);break
                except UnicodeError:pass
            frame=pd.read_csv(io.StringIO(txt),low_memory=False)
            independent=list(csv.DictReader(io.StringIO(txt)))
            assert len(independent)==len(frame)
        else:frame=data;independent=None
        itemcol=next((c for c in ['장비명','항목명','측정항목','항목'] if c in frame),None)
        valuecol=next((c for c in ['측정값','측정데이터','값'] if c in frame),None)
        item={'file':filename,'member':name,'rows':len(frame),'columns':list(frame),'groups':[]}
        for c in ['품목명','장비명','항목명','단위']:
            if c in frame:item[c]=frame[c].value_counts(dropna=False).to_dict()
        if itemcol and valuecol:
            for key,g in frame.groupby(itemcol,dropna=False):
                x=pd.to_numeric(g[valuecol],errors='coerce');v=x.dropna()
                p={'item':str(key),'rows':len(g),'numeric':len(v),'missing':int(x.isna().sum()),'zero':int(v.eq(0).sum()),'fractional':int((v-v.round()).abs().gt(1e-9).sum()),'unique':v.nunique(),'min':v.min(),'max':v.max(),'median':v.median(),'units':g['단위'].unique().tolist() if '단위' in g else []}
                if independent is not None:
                    assert sum(r[itemcol]==str(key) for r in independent)==len(g)
                item['groups'].append(p)
        if '품목명' in frame and '농가id' in frame:
            sf=frame[frame['품목명'].astype(str).str.contains('딸기')]
            item['strawberry_rows']=len(sf);item['strawberry_farms']=sf['농가id'].nunique()
            item['strawberry_farm_profiles']=[]
            for farm,g in sf.groupby('농가id'):
                p={'farm':farm,'rows':len(g),'items':g[itemcol].unique().tolist(),'fields':{}}
                for key,gg in g.groupby(itemcol):
                    x=pd.to_numeric(gg[valuecol],errors='coerce')
                    dt=pd.to_datetime(gg['수집일자'],errors='coerce').sort_values()
                    p['fields'][key]={'n':len(gg),'zero':int(x.eq(0).sum()),'missing':int(x.isna().sum()),'min':x.min(),'max':x.max(),'unique':x.nunique(),'fractional':int((x-x.round()).abs().gt(1e-9).sum()),'start':str(dt.min()),'end':str(dt.max()),'gap_hours_mode':dt.diff().dt.total_seconds().div(3600).mode().tolist(),'duplicate_times':int(dt.duplicated().sum())}
                item['strawberry_farm_profiles'].append(p)
        results.append(item)
(OUT/'item_profiles_v1.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
for item in results:
    print(json.dumps({k:v for k,v in item.items() if k!='strawberry_farm_profiles'},ensure_ascii=False,default=str))
    for farm in item.get('strawberry_farm_profiles',[]):print(json.dumps(farm,ensure_ascii=False,default=str))
