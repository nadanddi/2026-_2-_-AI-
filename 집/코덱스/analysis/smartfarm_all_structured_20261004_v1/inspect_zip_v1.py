from pathlib import Path
import sys, io, zipfile, json, hashlib
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'집/클로드/research'))
import env
import pandas as pd
import numpy as np

def profile(path):
    result={'source_file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'members':[]}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if not info.filename.lower().endswith(('.csv','.xlsx','.xls')): continue
            name=info.filename
            if not (info.flag_bits&0x800):
                try: name=name.encode('cp437').decode('cp949')
                except UnicodeError: pass
            data=z.read(info)
            if name.lower().endswith('.csv'):
                for enc in ['utf-8-sig','cp949','utf-16']:
                    try: df=pd.read_csv(io.BytesIO(data),encoding=enc,low_memory=False);break
                    except UnicodeError: continue
                else: raise ValueError(name)
            else: df=pd.read_excel(io.BytesIO(data))
            m={'name':name,'bytes':len(data),'rows':len(df),'columns':list(df.columns),'duplicates':int(df.duplicated().sum()),'fields':[]}
            for c in df.columns:
                x=df[c];v=pd.to_numeric(x,errors='coerce'); valid=v.dropna()
                p={'name':str(c),'missing':int(x.isna().sum()),'unique':int(x.nunique()),'numeric':int(v.notna().sum())}
                if len(valid): p.update(min=float(valid.min()),max=float(valid.max()),median=float(valid.median()),zero=int((valid==0).sum()),fractional=int((valid.sub(valid.round()).abs()>1e-9).sum()))
                if any(k in str(c).lower() for k in ['일자','일시','시간','시각','date','time']):
                    dt=pd.to_datetime(x,errors='coerce'); p.update(time_valid=int(dt.notna().sum()),start=str(dt.min()),end=str(dt.max()))
                m['fields'].append(p)
            result['members'].append(m)
    return result

if __name__=='__main__':
    result=profile(Path(sys.argv[1]));dest=Path(sys.argv[2]);dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
