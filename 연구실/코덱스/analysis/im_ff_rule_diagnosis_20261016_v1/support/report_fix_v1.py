from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'aggregate_v2.py').read_text(encoding='utf-8')
old="    return dict(고친진짜=int(((y==0)&bb&~cc).sum()),"
new="    return dict(진짜분모=int((y==0).sum()),기존진짜오판=int(((y==0)&bb).sum()),반사실진짜오판=int(((y==0)&cc).sum()),기존오탐률=float(((y==0)&bb).sum()/(y==0).sum()) if (y==0).any() else None,반사실오탐률=float(((y==0)&cc).sum()/(y==0).sum()) if (y==0).any() else None,고친진짜=int(((y==0)&bb&~cc).sum()),"
assert old in s;s=s.replace(old,new)
s=s.replace("진짜분모=int(real.sum()),진짜경보율=", "진짜경보율=")
old="cols=[]"
new="""            if sn in core:
                target=(z['y']==1)&(z['p9']<.5)&np.isin(z['kind'],('partial_style','partial_gen_v7','partial_gen_v7s'))
                for kd in sorted(set(z['kind'][target])):
                    sel=target&(z['kind']==kd)
                    row('E3 자료유형별부분편집',sn,scope,name,int((a&sel).sum()),int(sel.sum()),threshold,kind=str(kd))
cols=[]"""
assert old in s;s=s.replace(old,new)
with (P/'aggregate_v3.py').open('x',encoding='utf-8') as f:f.write(s)
print('REPORTING_ONLY_V3')
