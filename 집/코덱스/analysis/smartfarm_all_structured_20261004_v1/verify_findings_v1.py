from pathlib import Path
import sys,json,zipfile,io,csv,collections,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd
OUT=Path(__file__).parent
RAW=ROOT/'집/코덱스/local/smartfarm_all_structured_20261004_v1'
result={'cleaning':'None: raw data preserved; crop filtering only','files':[]}
for filename in ['2022_11_ds.zip','2022_9_ds.zip','2022_1_ds.zip','2023_17_ds.zip','2024_19_ds.zip','2024_21_ds.zip','2024_23_ds.zip','2024_20_ds.zip','2023_13_ds.zip']:
    src=RAW/filename
    if not src.exists():continue
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            if not info.filename.endswith('.csv'):continue
            data=z.read(info)
            for enc in ['utf-8-sig','cp949']:
                try:txt=data.decode(enc);break
                except UnicodeError:continue
            df=pd.read_csv(io.StringIO(txt),low_memory=False)
            if not all(c in df for c in ['품목명','장비명','측정값','농가id']):continue
            crop=df['품목명'].astype(str);sf=df[crop.str.contains('딸기')|crop.isin(['80400','080400'])]
            p={'file':filename,'rows':len(sf),'farms':sf['농가id'].nunique(),'groups':[]}
            for key,g in sf.groupby('장비명'):
                if not any(k in key.lower() for k in ['ec','온도','무게','습','함수','창','커튼','방','공급량','배액량']):continue
                x=pd.to_numeric(g['측정값'],errors='coerce');dt=pd.to_datetime(g['수집일자'],errors='coerce')
                q={'item':key,'n':len(g),'zero':int(x.eq(0).sum()),'negative':int(x.lt(0).sum()),'fractional':int((x-x.round()).abs().gt(1e-9).sum()),'nunique':x.nunique(),'min':x.min(),'max':x.max(),'start':str(dt.min()),'end':str(dt.max()),'farms':g['농가id'].nunique()}
                if '보간데이터여부' in g:
                    q['interpolation_flags']=g['보간데이터여부'].value_counts(dropna=False).to_dict()
                    q['flag_groups']={str(flag):{'n':len(gg),'fractional':int((pd.to_numeric(gg['측정값'])-pd.to_numeric(gg['측정값']).round()).abs().gt(1e-9).sum()),'zero':int(pd.to_numeric(gg['측정값']).eq(0).sum())} for flag,gg in g.groupby('보간데이터여부',dropna=False)}
                if 'ec' in key.lower():
                    q['farm_details']=[]
                    for farm,gg in g.groupby('농가id'):
                        xx=pd.to_numeric(gg['측정값']);q['farm_details'].append({'farm':farm,'n':len(gg),'unique':xx.nunique(),'min':xx.min(),'max':xx.max(),'zero':int(xx.eq(0).sum()),'fractional':int((xx-xx.round()).abs().gt(1e-9).sum())})
                    counter=collections.Counter();zeros=0;frac=0;manualfarms=set()
                    for row in csv.DictReader(io.StringIO(txt)):
                        if ('딸기' in row['품목명'] or row['품목명'] in ['80400','080400']) and row['장비명']==key:
                            v=float(row['측정값']);counter[v]+=1;zeros+=v==0;frac+=abs(v-round(v))>1e-9;manualfarms.add(row['농가id'])
                    assert sum(counter.values())==len(g) and zeros==q['zero'] and frac==q['fractional'] and len(counter)==q['nunique'] and len(manualfarms)==q['farms']
                p['groups'].append(q)
            result['files'].append(p)
rda=pd.read_excel(RAW/'2022_env.xlsx');sf=rda[rda['품목'].astype(str).str.contains('딸기')]
result['rda2022']={'all_rows':len(rda),'crop_counts':rda['품목'].value_counts().to_dict(),'strawberry_rows':len(sf),'farms':sf['농가명'].nunique(),'fields':{str(c):int(sf[c].notna().sum()) for c in sf},'columns':list(sf)}
(OUT/'verified_findings_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
for p in result['files']:
    print(json.dumps({**p,'groups':[{k:v for k,v in q.items() if k!='farm_details'} for q in p['groups']]},ensure_ascii=False,default=str))
print(json.dumps(result['rda2022'],ensure_ascii=False,default=str))
