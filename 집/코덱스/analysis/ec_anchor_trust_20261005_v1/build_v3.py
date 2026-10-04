from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'run_v2.py').read_text(encoding='utf-8-sig')
s=s.replace('assert e<1e-10','assert e<=1e-12').replace('preparation_v2.json','preparation_v3.json')
s=s.replace("old=load(PREV);assert old", "assert sha(PREV)=='df4ff2c3b7617676c61e4cee4b8e9690f9312a7cca448c0caec189cb5051b860'; old=load(PREV);assert old")
s=s.replace("sig=signatures(raw);idx=lab.set_index('row_id');records=[];refs={}","""sig=signatures(raw); causal=[]
 for farm in ['F13','F47']:
  day=int(raw.loc[raw.farm==farm,'day'].min());g=raw[(raw.farm==farm)&(raw.day==day)].copy()
  for h in [0,6,12]:
   cut=g[g.hour<=h];v=signatures(cut).to_numpy();full=sig.reindex(cut.row_id).to_numpy();assert np.array_equal(np.isnan(v),np.isnan(full));close(v[np.isfinite(v)],full[np.isfinite(full)])
   altered=g.copy();altered.loc[altered.hour>h,CC]=12345.;v2=signatures(altered).reindex(cut.row_id).to_numpy();assert np.array_equal(np.isnan(v2),np.isnan(full));close(v2[np.isfinite(v2)],full[np.isfinite(full)]);causal.append(dict(farm=farm,day=day,hour=h,error=0.))
 idx=lab.set_index('row_id');records=[];refs={} """)
s=s.replace("prep=dict(status=", "prep=dict(causal_prefix=causal,status=")
s=s.replace("assert set(a.row_id)|set(b.row_id)<=set(tr.row_id)","assert a.row_id.is_unique and b.row_id.is_unique; assert set(a.row_id)|set(b.row_id)<=set(tr.row_id)")
s=s.replace("refs,prep=preparation();assert load(H/'preparation_v3.json')==prep", "refs,prep=preparation(); reg=load(H/'registration_v1.json'); assert reg['run_sha']==sha(Path(__file__)) and reg['prep_sha']==sha(H/'preparation_v3.json') and reg['prereg_sha']==sha(H/'preregistration_v1.md'); assert load(H/'preparation_v3.json')==prep")
s=s.replace("close(p,p2);close(d,d2);close(z,z2);assert fit==fit2", """close(p,p2);close(d,d2);close(z,z2);assert fit==fit2
    x,ok,dq=design(ref['q'],ref['baseline'][seed],ref['NQ'],ref['QQ']);replayed=replay(mode,x,fit)
    close(z,replayed);close(replayed,replay(mode,x[::-1],fit)[::-1]);close(replayed,np.array([replay(mode,x[i:i+1],fit)[0] for i in range(len(x))]));changed=x[:8].copy();changed[1:]+=10000.;close(replayed[:1],replay(mode,changed,fit)[:1])
    scalar=np.array([min(ref['bounds'][1],max(ref['bounds'][0],float(a)+float(b))) for a,b in zip(ref['baseline'][seed],d)]);close(p,scalar) """)
s=s.replace("def actual(mode):", """def replay(mode,x,fit):
 xx=(x-np.asarray(fit['mean']))/fit['scale']
 if fit['model_none']:return np.full(len(x),float(fit['positive']==fit['n']))
 lin=xx@np.asarray(fit['coef']).reshape(-1)+float(np.asarray(fit['intercept']).reshape(-1)[0])
 if mode=='RIDGE':return lin
 z=np.empty_like(lin);pos=lin>=0;z[pos]=1/(1+np.exp(-lin[pos]));e=np.exp(lin[~pos]);z[~pos]=e/(1+e);return z
def actual(mode):""")
with (H/'run_v3.py').open('x',encoding='utf-8') as f:f.write(s)
t=(H/'verify_v1.py').read_text(encoding='utf-8').replace('run_v2.py','run_v3.py').replace('preparation_v2.json','preparation_v3.json')
with (H/'verify_v2.py').open('x',encoding='utf-8') as f:f.write(t)
print('new v3 runner / v2 verifier saved; v1/v2 failures preserved')
