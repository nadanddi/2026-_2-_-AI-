from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'analyze.py').read_text(encoding='utf-8')
s=s.replace("assert len(paths)==66,('incomplete',len(paths))","assert len(paths)==1,('incomplete',len(paths))")
a=s.index('maxdiff=0\n');b=s.index('boot={};',a)
new='''maxdiff=0
for (v,k,s,f,d),g in o.groupby(['validator','fold','seed','farm','day']):
    lo,hi=bounds[(v,k)]
    for r in g.itertuples():
        expected=min(hi,max(lo,r.baseline+r.correction))
        delta=abs(expected-r.candidate);assert delta<1e-12;maxdiff=max(maxdiff,delta);checks+=1
    base=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==s)].set_index('row_id').season_v2.reindex(g.row_id).to_numpy()
    assert np.array_equal(base,g.baseline.to_numpy());checks+=1
'''
s=s[:a]+new+s[b:]
p=H/'analyze_calibration.py';assert not p.exists();compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
print('saved',p)
