"""Run isolated v2 helpers on synthetic fixtures. No original-script imports/fit."""
from pathlib import Path
import ast,json,math,hashlib
from decimal import Decimal,localcontext
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'집/코덱스/analysis/ec_nested_log_blend_20261004_v1/run_v2.py'
tree=ast.parse(SRC.read_text(encoding='utf-8'))
names={'finite','maxdiff','vector_id_sha','features_sha','first_artifact_guard','weight'}
nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
ns={'np':np,'pd':pd,'json':json,'math':math,'LAMBDA':.01}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC),'exec'),ns)

weight_checks=[]
for name,e,d in [('interior',[-.12,0.,.02,0.],[.2,0.,.1,0.]),('zero',[.1,.2],[.2,.1]),
                 ('upper',[-2.,-2.],[.2,.1]),('all_zero',[.2,-.3],[0.,0.])]:
    e,d=np.asarray(e),np.asarray(d)
    w,stats=ns['weight'](e,d)
    with localcontext() as ctx:
        ctx.prec=40
        ee=[Decimal(str(x)) for x in e];dd=[Decimal(str(x)) for x in d]
        num=sum((x*y for x,y in zip(ee,dd)),Decimal(0))/Decimal(len(e))
        den=sum((x*x for x in dd),Decimal(0))/Decimal(len(e))+Decimal('.01')
        expected=max(Decimal(0),min(Decimal(1),-num/den))
        assert abs(float(expected)-w)<1e-12
        assert abs(float(num)-stats['numerator'])<1e-12 and abs(float(den)-stats['denominator'])<1e-12
    weight_checks.append(dict(case=name,weight=w,gradient=stats['gradient']))

x=pd.DataFrame({'x':[.1,np.nan,.3],'season':[1.,2.,3.]})
sha=ns['features_sha'](x,['x','season'])
assert ns['features_sha'](x.copy(),['x','season'])==sha
changed=x.copy();changed.loc[0,'x']+=1
assert ns['features_sha'](changed,['x','season'])!=sha
changed=x.copy();changed.loc[0,'season']+=1
assert ns['features_sha'](changed,['x','season'])!=sha
assert ns['features_sha'](x.iloc[::-1],['x','season'])!=sha
assert ns['features_sha'](x,['season','x'])!=sha
payload=x.copy();payload.loc[1,'x']=np.array([0x7ff8000000000001],dtype=np.uint64).view(np.float64)[0]
assert ns['features_sha'](payload,['x','season'])==sha

fixtures=OUT/'nested_log_v2_synthetic_first_artifacts_v1'
assert not fixtures.exists()
fixtures.mkdir()
checks=[];signature={'synthetic':'same'}
for pattern in range(16):
    case=fixtures/f'case_{pattern:02d}';case.mkdir();ns['H']=case
    paths=[case/'cell.csv',case/'cell.json',case/'cell.npz',case/'first_fold_verification_v2.json']
    for i,p in enumerate(paths):
        if pattern&(1<<i):p.write_text(json.dumps({'status':'PASS','signature':signature}),encoding='utf-8')
    raised=False
    try:ns['first_artifact_guard'](*paths[:3],signature)
    except AssertionError:raised=True
    assert raised==(pattern not in [0,15])
    checks.append(dict(present_pattern=pattern,stopped_before_fit=raised))
ns['H']=fixtures/'case_15'
bad_signature_stopped=False
try:ns['first_artifact_guard'](*(fixtures/'case_15'/p for p in ['cell.csv','cell.json','cell.npz']),{'synthetic':'changed'})
except AssertionError:bad_signature_stopped=True
assert bad_signature_stopped

main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
calls=[(n.lineno,n.func.id) for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
assert min(l for l,n in calls if n=='first_artifact_guard')<min(l for l,n in calls if n=='fit_direction')
record=dict(status='PASS',scope='Actual v2 helper extraction on synthetic data only; no import/model/prediction/real scoring',
            source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),weight_checks=weight_checks,
            feature_hash_detects_value_season_order_column_changes=True,feature_hash_canonicalizes_nan_payload=True,
            first_artifact_combinations=checks,changed_first_signature_stopped=True,
            first_guard_precedes_fit_direction=True)
(OUT/'nested_log_v2_code_check_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='first_artifact_combinations'},ensure_ascii=False,indent=2))
