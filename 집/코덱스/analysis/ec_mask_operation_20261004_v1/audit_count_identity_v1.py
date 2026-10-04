"""Check count-feature identities; no target, model training or scores."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import ast, hashlib, json, importlib.util
import numpy as np
import pandas as pd
spec = importlib.util.spec_from_file_location('input_audit', HERE/'audit_inputs_v1.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
CORE = ROOT/'집/코덱스/analysis/codex_independent/rl_ec_v1/run.py'

def main():
    output = HERE/'count_identity_v1.json'
    assert not output.exists(), output
    tree = ast.parse(CORE.read_text(encoding='utf-8-sig'))
    acts = ['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
    indoor = ['in_temp','in_hum','in_co2']
    base = indoor+acts+['day','hr_sin','hr_cos','midnight']
    full = base+[c+'_h0' for c in acts+indoor]+[name for c in acts for name in (c+'_tdm',c+'_tdz')]
    ns = dict(np=np,pd=pd,ACTS=acts,INDOOR=indoor,FULL=full)
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'identify','features'}]
    assert len(nodes) == 2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CORE),'exec'),ns)
    raw = pd.read_csv(A.INPUT,usecols=['row_id']+indoor+acts,float_precision='round_trip')
    raw = raw[raw.row_id.str[:3].isin(['F13','F47'])]
    d = ns['features'](raw)
    op = pd.concat([A.extracted()(g) for _,g in d.groupby(['farm','day'])]).reindex(d.index)
    recovered_hour = np.rint(np.arctan2(d.hr_sin,d.hr_cos)*24/(2*np.pi)).astype(int)%24
    assert np.array_equal(recovered_hour.to_numpy(),d.hour.to_numpy())
    checks = []
    for new, original in [('vent_open_hours','act_vent_tdz'),('co2_hours','act_co2_tdz')]:
        derived = (recovered_hour+1)*(1-d[original])
        error = float(np.max(np.abs(derived-op[new])))
        assert error < 1e-12
        checks.append(dict(feature=new, existing_feature=original, rows=len(d), max_abs_error=error,
                           ET_FULL_contains_inputs=all(c in full for c in [original,'hr_sin','hr_cos']),
                           LGB_MLP_BASE_contains_history=original in base))
    result = dict(status='PASS_ALGEBRA_ONLY', rows=len(d), days=len(d[['farm','day']].drop_duplicates()),
                  core_sha256=A.sha(CORE), dp1_source_sha256=A.sha(A.SOURCE), input_sha256=A.sha(A.INPUT),
                  checks=checks, comparisons=2*len(d), target_use=0, fit=0,predict=0,scores=0,
                  limitations=['Identities rely on observed finite nonnegative controls and complete hour0..h rows.',
                               'Deterministic transforms can improve tree representation despite containing no new information.',
                               'Counts are reconstructible in ET FULL, but history is absent from LGB/MLP BASE.',
                               'No performance attribution, original training-frame replay or feature-removal decision.'])
    with output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))

if __name__ == '__main__':
    main()
