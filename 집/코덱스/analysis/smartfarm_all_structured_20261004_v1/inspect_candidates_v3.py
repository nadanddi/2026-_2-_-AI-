from pathlib import Path
import sys,zipfile,io,json,csv,collections,hashlib,shutil
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd
import numpy as np
OUT=Path(__file__).parent
RAW=ROOT/'집/코덱스/local/smartfarm_all_structured_20261004_v1'
RAW.mkdir(parents=True,exist_ok=True)
for filename in sys.argv[1:]:
    src=Path('C:/Users/aozks/Downloads')/filename
    dest=RAW/filename
    if not dest.exists():shutil.copy2(src,dest)
    assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest()
    result={'source':filename,'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'members':[]}
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            if not info.filename.lower().endswith('.csv'):continue
            data=z.read(info)
            for enc in ['utf-8-sig','cp949']:
                try:txt=data.decode(enc);break
                except UnicodeError:continue
            df=pd.read_csv(io.StringIO(txt),low_memory=False)
            name=info.filename
            if not info.flag_bits&0x800:
                try:name=name.encode('cp437').decode('cp949')
                except UnicodeError:pass
            m={'member':name,'rows':len(df),'columns':list(df),'duplicates':int(df.duplicated().sum()),'missing':df.isna().sum().to_dict()}
            if '품목명' not in df or '장비명' not in df or '측정값' not in df:
                result['members'].append(m);continue
            m['crop_counts']=df['품목명'].value_counts().to_dict();m['all_items']=df['장비명'].value_counts().to_dict()
            crop=df['품목명'].astype(str)
            strawberry_mask=crop.str.contains('딸기')|crop.isin(['80400','080400'])
            sf=df[strawberry_mask]
            m['crop_selection']='literal 딸기 or official strawberry code 080400/80400'
            m['strawberry_rows']=len(sf);m['strawberry_farms']=sf['농가id'].nunique();m['groups']=[];m['farms']=[]
            manual=collections.Counter();manualzero=collections.Counter();farmset=set();nmanual=0
            for row in csv.DictReader(io.StringIO(txt)):
                if '딸기' in row['품목명'] or row['품목명'] in ['80400','080400']:
                    nmanual+=1;manual[row['장비명']]+=1;farmset.add(row['농가id'])
                    try:manualzero[row['장비명']]+=float(row['측정값'])==0
                    except ValueError:pass
            assert nmanual==len(sf) and len(farmset)==m['strawberry_farms']
            for key,g in sf.groupby('장비명'):
                x=pd.to_numeric(g['측정값'],errors='coerce');v=x.dropna()
                assert manual[key]==len(g) and manualzero[key]==int(v.eq(0).sum())
                m['groups'].append({'item':key,'n':len(g),'missing':int(x.isna().sum()),'zero':int(v.eq(0).sum()),'fractional':int((v-v.round()).abs().gt(1e-9).sum()),'unique':v.nunique(),'min':v.min(),'p01':v.quantile(.01),'median':v.median(),'p99':v.quantile(.99),'max':v.max(),'units':g['단위'].unique().tolist()})
            for farm,g in sf.groupby('농가id'):
                p={'farm':farm,'fields':{}}
                for key,gg in g.groupby('장비명'):
                    x=pd.to_numeric(gg['측정값'],errors='coerce');dt=pd.to_datetime(gg['수집일자'],errors='coerce').sort_values()
                    p['fields'][key]={'n':len(gg),'zero':int(x.eq(0).sum()),'missing':int(x.isna().sum()),'unique':x.nunique(),'min':x.min(),'max':x.max(),'start':str(dt.min()),'end':str(dt.max()),'duplicate_times':int(dt.duplicated().sum()),'gaps_over_hour':int(dt.diff().dt.total_seconds().gt(3600).sum())}
                m['farms'].append(p)
            result['members'].append(m)
    output=OUT/(src.stem.replace(' ','_')+'_items_v3.json')
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    print(json.dumps({**result,'members':[{k:v for k,v in m.items() if k!='farms'} for m in result['members']]},ensure_ascii=False,default=str))
