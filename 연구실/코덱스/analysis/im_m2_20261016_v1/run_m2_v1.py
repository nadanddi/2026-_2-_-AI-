"""Read-only tracing of the frozen candidate; control R5 entry before any score read."""
import os,sys,json,hashlib,runpy
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
HERE=Path(__file__).resolve().parent
MODE=sys.argv[1]; assert MODE in ('preflight','open')
STAGE=HERE/MODE
STAGE.mkdir(exist_ok=False)
os.environ['M2_STAGE_DIR']=str(STAGE)
CODE=HERE/'im66_m2_v1.py'
DEADLINE=datetime(2026,10,16,11,0,tzinfo=timezone(timedelta(hours=9)))
now=lambda:datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))
assert now()<DEADLINE,'deadline expired'
if MODE=='open':
    prior=json.loads((HERE/'preflight/state_v1.json').read_text())
    assert all(prior['gates'].values())
    proof=json.loads((HERE/'r5_integrity_v1.json').read_text())
    assert proof['complete'] and proof['features_ok'] and proof['source_disjoint'] and proof['group_disjoint']
    assert not (HERE/'R5_OPEN_ONCE.json').exists()
entered=False
def hashes(p):
    return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.glob('m_v2*'))}
def snapshot(loc):
    st=dict(mode=MODE,finished_at=now().isoformat(),deadline=DEADLINE.isoformat(),
        gates={k:bool(loc[k]) for k in ('g1','g2','g3','g5') if k in loc},r5_entered=entered)
    if 'O9' in loc:
        st['budget']=dict(n=len(loc['O9']),o9_count=int(loc['O9'].sum()),o9_rate=float(loc['O9'].mean()),
            limit=float(loc['O9'].mean())+.001,union_count=int(loc['un'].sum()),union_rate=float(loc['un'].mean()),
            theta=float(loc['TH']),theta_rc=float(loc['THrc']),fl=float(loc['fl']),flrc=float(loc['flrc']))
        np.savez(STAGE/'calibration_v1.npz',P=loc['P'],PRC=loc['PRC'],O9=loc['O9'],un=loc['un'])
    if 's9' in loc and isinstance(loc['s9'],np.ndarray):
        np.savez(STAGE/'g5_v1.npz',s9=loc['s9'],sM=loc['sM'])
    for key,C in loc.get('C',{}).items():
        n=len(C['y'])
        np.savez(STAGE/f'eval_{key}.npz',y=C['y'],p9=C['p9'],pm=C['pm'],
                 kind=np.array(['unknown']*n) if C['k'] is None else C['k'],
                 group=np.array(['unknown']*n) if C['g'] is None else C['g'])
    if 'pw' in loc:
        st['r5_p_worse']=float(loc['pw']); st['r5_conditions']=[bool(v) for v in loc['c']]
        st['on_time']=now()<DEADLINE
        st['adoption_candidate']=all(st['r5_conditions']) and st['on_time']
    with (STAGE/'state_v1.json').open('x') as f: json.dump(st,f,indent=2)
def trace(frame,event,arg):
    global entered
    if Path(frame.f_code.co_filename).resolve()!=CODE.resolve(): return None
    # At rep(im_set_r5) call, locals have the name; no dataset loaded yet.
    if event=='call' and frame.f_code.co_name=='rep' and frame.f_locals.get('sn')=='im_set_r5':
        assert MODE=='open' and not entered and now()<DEADLINE
        assert hashes(STAGE)==hashes(HERE/'preflight'),'retrained files differ before R5 read'
        with (HERE/'R5_OPEN_ONCE.json').open('x') as f: json.dump(dict(opened_at=now().isoformat(),identical_artifacts=hashes(STAGE)),f,indent=2)
        entered=True
    if event=='return' and frame.f_code.co_name=='main': snapshot(frame.f_locals)
    return trace
sys.argv=[str(CODE)]+(['open_r5'] if MODE=='open' else [])
sys.settrace(trace)
try: runpy.run_path(str(CODE),run_name='__main__')
finally: sys.settrace(None)
