"""Loaded native parameter schema plus adversarial model gate, no fitting."""
from pathlib import Path
import ast,json,copy
H=Path(__file__).resolve().parent
tree=ast.parse((H/'verify_full_v3.py').read_text(encoding='utf-8-sig'));nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='validate_model'];assert len(nodes)==1
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'validate_model','exec'),ns)
params=json.loads((H/'loaded_params_probe_v1.json').read_text())['params']
original=json.loads((H/'preparation_v3.json').read_text())['params']['7']
bs=json.loads((H/'preparation_v3.json').read_text())['manifest'][0]['old_lgb_features']
class Fixture:
 def __init__(self):self.params=copy.deepcopy(params);self.features=list(bs);self.trees=1;self.objective='tweedie_exact_leaf rho:1.5 lambda_l2:1'
 def feature_name(self):return self.features
 def num_feature(self):return len(self.features)
 def num_trees(self):return self.trees
 def dump_model(self):return {'objective':self.objective}
validate=ns['validate_model'];assert validate(Fixture(),bs,7,original)==1
rejected=[]
for key in ['features','trees0','trees801','objective','num_iterations','learning_rate','bagging_fraction','bagging_freq','feature_fraction','lambda_l2','lambda_l1','tweedie_variance_power','num_leaves','min_data_in_leaf','seed']:
 bad=Fixture()
 if key=='features':bad.features=bad.features[::-1]
 elif key=='trees0':bad.trees=0
 elif key=='trees801':bad.trees=801
 elif key=='objective':bad.objective='tweedie'
 else:bad.params[key]+=1
 try:validate(bad,bs,7,original)
 except AssertionError:rejected.append(key)
 else:raise AssertionError(('accepted',key))
with (H/'synthetic_model_gate_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_SYNTHETIC_ONLY',corruption_rejected=rejected,loaded_schema_source='loaded_params_probe_v1.json',fit=0,real_predict=0,real_score=0),f,indent=2)
print('MODEL_GATE_SYNTHETIC_PASS',len(rejected))
