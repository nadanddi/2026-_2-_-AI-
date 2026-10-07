from pathlib import Path
import sys,json,collections,math
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import openpyxl
OUT=Path(__file__).parent
book=openpyxl.load_workbook(ROOT/'집/코덱스/local/smartfarm_all_structured_20261004_v1/2022_env.xlsx',read_only=True,data_only=True)
results=[]
for sheet in book:
    rows=sheet.iter_rows(values_only=True);cols=next(rows);counts=collections.Counter();farms=set();stats={c:{'valid':0,'missing':0,'zero':0,'negative':0,'min':None,'max':None} for c in cols[6:]};start=None;end=None
    for values in rows:
        if all(v is None for v in values):continue
        r=dict(zip(cols,values));counts['all']+=1
        if '딸기' not in str(r['품목']):continue
        counts['strawberry']+=1;farms.add(r['농가명']);dt=r['측정시간']
        if dt is not None:
            dt=str(dt);start=min(start,dt) if start else dt;end=max(end,dt) if end else dt
        for c,st in stats.items():
            v=r[c]
            if v is None:st['missing']+=1;continue
            try:v=float(v)
            except (ValueError,TypeError):st['missing']+=1;continue
            if not math.isfinite(v):st['missing']+=1;continue
            st['valid']+=1;st['zero']+=v==0;st['negative']+=v<0;st['min']=min(st['min'],v) if st['min'] is not None else v;st['max']=max(st['max'],v) if st['max'] is not None else v
    results.append({'sheet':sheet.title,'columns':cols,'rows':counts['all'],'strawberry_rows':counts['strawberry'],'strawberry_farms':len(farms),'start':start,'end':end,'strawberry_numeric':stats})
    print('XLSX_INDEPENDENT_DONE',counts,len(farms),flush=True)
book.close();p=OUT/'xlsx_independent_v1.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
