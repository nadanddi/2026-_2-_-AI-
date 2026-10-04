from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'run_v5.py').read_text(encoding='utf-8-sig').replace('preparation_v5.json','preparation_v6.json')
s=s.replace("fit=dict(n=int(eligible.sum())", "fit=dict(eligible_ids=ids(b.loc[eligible,'row_id']),features_sha=ar(xb[eligible]),cost_sha=ar(cost[eligible]),mode=mode,seed=seed,n=int(eligible.sum())")
s=s.replace("save(H/f'first_{mode}_v1.json',dict(status='PASS',repeat_error=close(p,p2),fit=fit,source=sha(Path(__file__))))", "save(H/f'first_{mode}_v1.json',dict(status='PASS',repeat_error=close(p,p2),serialized_error=close(z,replayed),reverse_error=close(replayed,replay(mode,x[::-1],fit)[::-1]),single_error=close(replayed,np.array([replay(mode,x[i:i+1],fit)[0] for i in range(len(x))])),other_query_error=close(replayed[:1],replay(mode,changed,fit)[:1]),scalar_error=close(p,scalar),causal_prefix=prep['causal_prefix'],fit=fit,source=sha(Path(__file__))))")
with (H/'run_v6.py').open('x',encoding='utf-8') as f:f.write(s)
v=(H/'verify_v4.py').read_text(encoding='utf-8-sig').replace('run_v5.py','run_v6.py').replace('preparation_v5.json','preparation_v6.json')
v=v.replace("first['repeat_error']<=1e-12;", "all(0<=first[z]<=1e-12 for z in ['repeat_error','serialized_error','reverse_error','single_error','other_query_error','scalar_error']) and first['causal_prefix']==prep['causal_prefix'];")
with (H/'verify_v5.py').open('x',encoding='utf-8') as f:f.write(v)
l=(H/'verify_learning_v3.py').read_text(encoding='utf-8-sig').replace('run_v5.py','run_v6.py').replace('preparation_v5.json','preparation_v6.json')
l=l.replace("m=R.load(R.OUT/mode/f'{v}_{k}_{s}_fit.json');", "m=R.load(R.OUT/mode/f'{v}_{k}_{s}_fit.json');assert m['eligible_ids']==R.ids(b.loc[sel,'row_id']) and m['features_sha']==R.ar(x[sel]) and m['cost_sha']==R.ar(cost[sel]) and m['mode']==mode and m['seed']==s;")
with (H/'verify_learning_v4.py').open('x',encoding='utf-8') as f:f.write(l)
print('v6 first audit all errors + training fingerprint; verifier whole and learning pinned')
