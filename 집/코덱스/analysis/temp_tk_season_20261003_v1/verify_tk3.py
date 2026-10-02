from world import *
import math
from analyze import boot
j=json.loads((HERE/'TK3_reproduction.json').read_text(encoding='utf-8'));o=pd.read_csv(OUT/'TK3_original_mix.csv');checks=[]
for r in j['scores']:
    if r['segment']!='all':continue
    d=o[(o.validator==r['validator'])&(o.seed==r['seed'])&(o.context==r['context'])]
    a=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(d.base,d.sub_temp))/len(d));b=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(d.candidate,d.sub_temp))/len(d));assert max(abs(a-r['baseline_rmse']),abs(b-r['candidate_rmse']))<1e-12
    if r['validator']=='DIAG10':assert boot(d,d.base.to_numpy(),d.candidate.to_numpy())['p_worse']==r['p_worse']
    checks.append(dict(validator=r['validator'],seed=r['seed'],context=r['context'],n=len(d),baseline=a,candidate=b,delta_pct=100*(b/a-1)))
assert max(r['maxdiff'] for r in j['comparisons'])<1e-7
(HERE/'TK3_verification.json').write_text(json.dumps(dict(status='PASS',fsum_scores=checks,reproduction_maxdiff=max(r['maxdiff'] for r in j['comparisons']),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'집/클로드/research/temp_TC1_season_members_v1.py',ROOT/'집/클로드/research/temp_TC2_w30g_season_v1.py',ROOT/'집/코덱스/analysis/ec_submission10_season_20261002_v1/season.py',HERE/'tk3_reproduce.py']}),ensure_ascii=False,indent=2),encoding='utf-8');print('TK3 12 mixed score cells reverified, original TC1/TC2 reproduced')
