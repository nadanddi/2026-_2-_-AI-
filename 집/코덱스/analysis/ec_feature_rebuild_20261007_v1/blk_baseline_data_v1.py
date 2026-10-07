"""CPU baseline feature/calendar preparation, using BLKContext boundaries."""
from pathlib import Path
import ast
import hashlib
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from blk_context_v1 import BLKContext, key

DEPLOY=ROOT/'집/클로드/submission14_ec_sg2/model.py'
DC4=ROOT/'집/클로드/research/ec2_DC4_exact_twin_anchor_v1.py'
CORE=Path(env.CODEX)/'rl_ec_v1/run.py'
RAW=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
ACTS=RAW[3:]
W=['out_temp','out_hum','out_rad','out_wspd']
DAY_BASE=RAW+['day','hr_sin','hr_cos','midnight']
DAY_FULL=DAY_BASE+[v+'_h0' for v in ACTS+RAW[:3]]+[n for v in ACTS for n in (v+'_tdm',v+'_tdz')]
BASE=[c for c in DAY_BASE if c!='day']+['season']
FULL=[c for c in DAY_FULL if c!='day']+['season']
OPS=['seal_run','vent_open_hours','first_open_hour','thermal_switches','shade_switches','since_curtain_change','heat_run','co2_hours','vent_max']
FULL_R3=FULL+OPS
BASE_R3=BASE+OPS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(path,names,ns):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
ns=dict(globals())
extract(CORE,['identify'],ns)
extract(DEPLOY,['run_len','ops_day','features','shrink'],ns)
extract(DC4,['weather_vectors','season_index'],ns)

def dataframe(observations):
    return pd.DataFrame([{'row_id':rid,**obs} for rid,obs in sorted(observations.items())])

def prepare_reference(ctx):
    x=dataframe(ctx.reference_inputs)
    tr=ns['features'](x)
    tr['sub_ec']=[ctx.reference_labels[r]['sub_ec'] for r in tr.row_id]
    td=tr[['farm','day']].drop_duplicates()
    qd=pd.DataFrame(sorted({key(r)[:2] for r in ctx.query_ids}),columns=['farm','day'])
    season,qs=ns['season_index'](td,qd,ns['weather_vectors'](x))
    tr['season']=[season[f,int(d)] for f,d in zip(tr.farm,tr.day)]
    table=dict(zip(qd.itertuples(index=False,name=None),qs))
    return x,tr,table

def prepare_query(ctx,rids,table):
    # Only the prefixes required for requested rows are handed to the generator.
    observations={}
    for rid in rids:
        observations.update(ctx.query_prefix(rid))
    q=ns['features'](dataframe(observations)).set_index('row_id').loc[list(rids)].reset_index()
    q['season']=[table[f,int(d)] for f,d in zip(q.farm,q.day)]
    return q

def audit(ctx,q,table):
    checks=0
    original=q.set_index('row_id')
    for f in ['F13','F47']:
        ids=sorted(r for r in ctx.query_ids if key(r)[0]==f)
        for i in [0,len(ids)//2,len(ids)-1]:
            rid=ids[i]
            single=prepare_query(ctx,[rid],table).set_index('row_id')
            pd.testing.assert_frame_equal(original.loc[[rid],FULL_R3],single[FULL_R3])
            checks+=1
    shuffled=prepare_query(ctx,list(reversed(q.row_id.tolist())),table).set_index('row_id')
    pd.testing.assert_frame_equal(original[FULL_R3],shuffled.loc[original.index,FULL_R3])
    assert len(FULL)==38 and len(FULL_R3)==47 and len(BASE_R3)==23
    return {'status':'PASS','single_prefix_and_order_checks':checks+1,
            'calendar_fit_reference_only':True,'query_values_used_for_fit':False}
