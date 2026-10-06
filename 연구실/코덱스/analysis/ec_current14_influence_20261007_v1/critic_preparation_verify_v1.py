"""Read-only structural/source-signature audit of fit-zero preparation."""
from pathlib import Path
import json, hashlib, collections
H=Path(__file__).resolve().parent
p=json.loads((H/'preparation_v4.json').read_text(encoding='utf-8'))
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
root=H.parents[3]
for key,path in [('script_sha',H/'runner_v4.py'),('adapter_sha',H/'sg2_ref_v2.py'),('plan_sha',H/'PLAN_v2.md'),('package_model_sha',root/'집/클로드/submission14_ec_sg2/model.py'),('package_sg2_sha',root/'집/클로드/submission14_ec_sg2/sg2post.py'),('imported_season_sha',Path(p['imported_season_path']))]:
    assert p[key]==sha(path),(key,path)
query=[]; coverage=[]
for r in p['records']:
    ti,qi=r['train_ids'],r['query_ids']
    assert len(ti)==r['train_rows']==len(set(ti))
    assert len(qi)==r['query_rows']==len(set(qi))
    assert not set(ti)&set(qi)
    td=collections.defaultdict(set)
    for i in ti:td[i[:3]].add(int(i[4:7]))
    for i in qi:assert min(abs(int(i[4:7])-d) for d in td[i[:3]])>=2
    counts=collections.Counter(i[:7] for i in qi)
    assert all(v==24 for v in counts.values())
    assert len(counts)==len(r['coverage'])
    assert len(r['PFN_sources'])==4
    query.extend(qi);coverage.extend(r['coverage'])
assert len(query)==len(set(query))==8640
assert len(coverage)==360 and p['fit']==0
assert [len(p['features'][x]) for x in ['FULL_R3','BASE_R3','FULL_PFN']]==[47,23,38]
out=dict(status='PASS_SOURCE_AND_PREPARATION_STRUCTURE',fit=0,query_rows=len(query),days=len(coverage),high_days=sum(x['high'] for x in coverage),ordinary_days=sum(not x['high'] for x in coverage),pass2_days=sum(x['pass2'] for x in coverage),limitation='No independent FULL-frame hash/RNG cache reexecution in this structural audit; runner asserts are code-reviewed only.')
with (H/'critic_preparation_verify_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
