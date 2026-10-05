from pathlib import Path
import ast,json,hashlib,uuid,os,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env;import env_extra
import numpy as np,pandas as pd
OUT=ROOT/'집/코덱스/local/ec_actual_A_nested_oof_synthetic_20261006_v1';OUT.mkdir(parents=True,exist_ok=False)
ns=dict(Path=Path,np=np,pd=pd,OUT=OUT,json=json,hashlib=hashlib,uuid=uuid,os=os)
names={'sha','load','save','cp','savecsv','cache','writecache'}
nodes=[n for n in ast.parse((H/'run_v3.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_v3_helpers','exec'),ns)
tests=[]
p=OUT/'json_v1.json';ns['save'](p,{'x':1});digest=ns['sha'](p);ns['save'](p,{'x':1});assert ns['sha'](p)==digest
try:ns['save'](p,{'x':2})
except AssertionError:pass
else:raise AssertionError('changed JSON must fail')
assert ns['sha'](p)==digest;tests.append('JSON same-value resume; changed-value rejection and preservation')
df=pd.DataFrame({'row_id':['x','y'],'value':[.1,3.3]});p=OUT/'csv_v1.csv';ns['savecsv'](p,df);digest=ns['sha'](p);ns['savecsv'](p,df)
changed=df.copy();changed.loc[0,'value']+=1e-10
try:ns['savecsv'](p,changed)
except AssertionError:pass
else:raise AssertionError('exact comparison must reject even tiny change')
assert ns['sha'](p)==digest;tests.append('CSV exact-match resume; 1e-10 change rejection and preservation')
stage=OUT/'.staging'/'interrupted_example';stage.mkdir(parents=True);(stage/'unfinished.npz').write_bytes(b'partial artifact preserved')
sig={'synthetic':True};d={'row_id':np.array(['x','y']),'raw':np.array([.1,.2])}
assert ns['cache'](OUT/'component_v1.npz',sig,d['row_id']) is None
_,meta=ns['writecache'](OUT/'component_v1.npz',d,sig,{'test':'not a model fit'})
cached,m=ns['cache'](OUT/'component_v1.npz',sig,d['row_id']);assert np.array_equal(cached['raw'],d['raw']) and m==meta
assert stage.exists() and not list((OUT/'components'/'component_v1').glob('*.partial*'));tests.append('interrupted staging ignored; complete NPZ+JSON directory published and reusable')
try:ns['cache'](OUT/'component_v1.npz',{'synthetic':False},d['row_id'])
except AssertionError:pass
else:raise AssertionError('wrong training signature must fail')
tests.append('changed component signature rejected')
proof=dict(status='PASS_SYNTHETIC_RECOVERY_ONLY',tests=tests,real_model_fit=0,source_sha=ns['sha'](H/'run_v3.py'))
with (H/'synthetic_recovery_v1.json').open('x',encoding='utf-8') as f:json.dump(proof,f,ensure_ascii=False,indent=2)
print(json.dumps(proof,ensure_ascii=False))
