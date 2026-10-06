from pathlib import Path
import json,hashlib,collections
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=load(H/'preparation_v1.json');old=load(H.parent/'ec_actual_A_cases_20261006_v1/result_v1.json')
dep=old['audit_manifest'][0]['dependencies'];checks=[]
for path,kind in [('집/코덱스/analysis/statistical_experiments_20261003_v1/support.py','support'),('집/코덱스/analysis/codex_independent/rl_ec_v1/run.py','core'),('집/클로드/research/env.py','env'),('집/클로드/research/env_extra.py','env_extra'),('집/코덱스/analysis/ec_submission10_season_20261002_v1/season.py','season')]:
    assert p['dependencies'][str(Path(path))]==sha(ROOT/path)==dep[kind],kind;checks.append(kind)
prior=load(H.parent/'ec_hardcase_crossfit_20261005_v1/preparation_v1.json');prior={(r['v'],r['k']):r['split'] for r in prior['records']}
group=collections.defaultdict(list)
for e in p['records']:group[e['v'],e['k']].append(e)
for key,g in group.items():
    outer=g[0]['outer_train_ids'];days=collections.defaultdict(set)
    for rid in outer:days[rid[:3]].add(int(rid[4:7]))
    assignment={(f,d):(i//5)%4 for f,ds in days.items() for i,d in enumerate(sorted(ds))}
    for e in g:
        j=e['j'];q=[rid for rid in outer if assignment[rid[:3],int(rid[4:7])]==j]
        ban={(rid[:3],int(rid[4:7])+o) for rid in q for o in [-1,0,1]};tr=[rid for rid in outer if (rid[:3],int(rid[4:7])) not in ban]
        assert e['query_ids']==q and e['train_ids']==tr
        assert e['ti']==prior[key][j]['ti'] and e['vi']==prior[key][j]['vi']
assert len(group)==20 and len(p['records'])==80
result=dict(status='PASS_INDEPENDENT_PRELAUNCH_FIT0',original_A_source_parity=checks,split_same_as_previous_proxy_experiment=True,outer_count=len(group),contexts=len(p['records']),train_rows_min=min(len(e['train_ids']) for e in p['records']),train_rows_max=max(len(e['train_ids']) for e in p['records']),query_rows_min=min(len(e['query_ids']) for e in p['records']),query_rows_max=max(len(e['query_ids']) for e in p['records']),prepared_sha=sha(H/'preparation_v1.json'))
with (H/'prelaunch_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))

