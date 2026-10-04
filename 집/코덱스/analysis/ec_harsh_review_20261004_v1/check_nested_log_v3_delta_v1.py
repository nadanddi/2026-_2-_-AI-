"""Static v2/v3 code comparison and public metadata only, no experiment import."""
from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'집/코덱스/analysis/ec_nested_log_blend_20261004_v1'
OLD=ast.parse((SRC/'run_v2.py').read_text(encoding='utf-8'))
NEW=ast.parse((SRC/'run_v3.py').read_text(encoding='utf-8'))
old_functions={n.name:n for n in OLD.body if isinstance(n,ast.FunctionDef)}
new_functions={n.name:n for n in NEW.body if isinstance(n,ast.FunctionDef)}
assert set(old_functions)==set(new_functions)
unchanged=[]
for name in old_functions:
    if name=='main':continue
    assert ast.dump(old_functions[name],include_attributes=False)==ast.dump(new_functions[name],include_attributes=False),name
    unchanged.append(name)
def assignment(tree,name):
    return next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
assert ast.dump(assignment(OLD,'LAMBDA'))==ast.dump(assignment(NEW,'LAMBDA'))
assert ast.dump(assignment(OLD,'SEEDS'))==ast.dump(assignment(NEW,'SEEDS'))
prepared=json.loads((SRC/'preparation_v3.json').read_text(encoding='utf-8'))
expected=dict(python='3.12.14',numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0')
assert prepared['status']=='PASS' and prepared['runtime']==expected
assert prepared['outer_log_cells_audited']==66
assert prepared['run_source_sha256']==hashlib.sha256((SRC/'run_v3.py').read_bytes()).hexdigest()
ORIGINAL=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
cells=[]
for (validator,fold) in [(c['validator'],c['fold']) for c in prepared['checked']]:
    for seed in [7,101,2024]:
        meta=json.loads((ORIGINAL/f'{validator}_{fold}_r3_{seed}.json').read_text(encoding='utf-8'))
        assert all(meta['provenance']['environment'][k]==v for k,v in expected.items())
        assert meta['provenance']['shared']['input_sha256']['train_X.csv']==prepared['train_input_sha256']
        assert meta['provenance']['shared']['core_sha256']==prepared['dependency_sha256']['core']
        cells.append((validator,fold,seed))
assert len(cells)==66 and len(set(cells))==66
record=dict(status='PASS',scope='Static source and existing public metadata audit only; no model fit/raw targets/experiment import',
            unchanged_helpers=unchanged,lambda_seeds_unchanged=True,
            runtime=expected,public_r3_metadata_cells=len(cells),runtime_input_core_metadata_match=True,
            source_sha256=prepared['run_source_sha256'])
(OUT/'nested_log_v3_delta_check_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
