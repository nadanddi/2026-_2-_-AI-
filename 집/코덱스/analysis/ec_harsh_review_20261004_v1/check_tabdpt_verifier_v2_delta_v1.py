"""Family20 verifier v2 delta and synthetic original-R3 guard audit.

One existing original R3 JSON is read only for field/type structure.
No existing NPZ, TabDPT outputs, public data, scores, model imports/fit/predict.
"""
import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT=Path('C:/work/farmai')
HERE=ROOT/'집/코덱스/analysis/ec_harsh_review_20261004_v1'
EXP=ROOT/'집/코덱스/analysis/ec_tabdpt_20261004_v1'
RESULT=HERE/'tabdpt_verifier_v2_delta_check_v1.json'
assert not RESULT.exists() and __debug__
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n):return ast.dump(n,include_attributes=False)
def assigned(n,name):return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
old=ast.parse((EXP/'verify_full_v1.py').read_text(encoding='utf-8-sig'))
new=ast.parse((EXP/'verify_full_v2.py').read_text(encoding='utf-8-sig'))
of={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
nf={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
assert set(nf)-set(of)=={'guard_original_r3'} and not set(of)-set(nf)
unchanged=[name for name in of if name not in ['verify_complete','synthetic_check']]
assert all(dump(of[name])==dump(nf[name]) for name in unchanged)
assert all(dump(a)==dump(b) for a,b in zip(
    [n for n in old.body if not isinstance(n,ast.FunctionDef)],
    [n for n in new.body if not isinstance(n,ast.FunctionDef)]))

normal=copy.deepcopy(new)
normal.body=[n for n in normal.body if not (isinstance(n,ast.FunctionDef) and n.name=='guard_original_r3')]
vf=next(n for n in normal.body if isinstance(n,ast.FunctionDef) and n.name=='verify_complete')
original_vf=of['verify_complete']
# Normalize only each of the reviewed new audit statements.
new_initial=next(n for n in vf.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
                 and [t.id for t in n.targets[0].elts]==['frames','cells','split_audit','r3_cache_audit'])
old_initial=next(n for n in original_vf.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
                 and [t.id for t in n.targets[0].elts]==['frames','cells','split_audit'])
vf.body[vf.body.index(new_initial)]=copy.deepcopy(old_initial)
new_loop=next(n for n in ast.walk(vf) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
              and n.target.id=='seed' and any(assigned(x,'original_path') for x in n.body))
old_loop=next(n for n in ast.walk(original_vf) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
              and n.target.id=='seed' and any(assigned(x,'original_r3') for x in n.body))
start=next(i for i,n in enumerate(new_loop.body) if assigned(n,'original_path'))
end=next(i for i,n in enumerate(new_loop.body) if isinstance(n,ast.Expr)
         and ast.unparse(n)=="compare(old[f'r3_{seed}'], original_r3['raw_r3'])")
guard_statements=new_loop.body[start:end]
assert len(guard_statements)==8
assert ast.unparse(guard_statements[0])=="original_path = OLD / f'{v}_{k}_r3_{seed}.npz'"
assert ast.unparse(guard_statements[1])=='original_digest = sha(original_path)'
assert ast.unparse(guard_statements[2])=="assert original_meta['prediction_sha256'] == original_digest"
assert ast.unparse(guard_statements[3])=='original_r3 = dict(np.load(original_path, allow_pickle=False))'
assert ast.unparse(guard_statements[4])=='r3_guard = guard_original_r3(original_r3, original_meta, original_digest, tr, q, seed, v, k, label_map)'
assert ast.unparse(guard_statements[5])=="r3_guard['metadata_sha256'] = sha(original_path.with_suffix('.json'))"
assert ast.unparse(guard_statements[6])=='r3_cache_audit.append(r3_guard)'
assert ast.unparse(guard_statements[7])=="compare(original_meta['provenance']['train_target_bounds'], [old['lo'], old['hi']])"
oi=next(i for i,n in enumerate(old_loop.body) if assigned(n,'original_r3'))
new_loop.body[start:end]=copy.deepcopy(old_loop.body[oi:oi+2])
new_count=next(n for n in vf.body if isinstance(n,ast.Assert) and ast.unparse(n.test)=='len(r3_cache_audit) == 66')
vf.body.remove(new_count)
result=next(n for n in vf.body if assigned(n,'result'))
assert len([kw for kw in result.value.keywords if kw.arg=='r3_cache_audit'])==1
result.value.keywords=[kw for kw in result.value.keywords if kw.arg!='r3_cache_audit']
last_print=vf.body[-1]
assert ast.unparse(last_print)=="print(json.dumps({k: v for k, v in result.items() if k not in ('runtime', 'split_audit', 'r3_cache_audit')}, ensure_ascii=False, indent=2))"
vf.body[-1]=copy.deepcopy(original_vf.body[-1])
assert dump(vf)==dump(original_vf)
old_syn=of['synthetic_check'];new_syn=nf['synthetic_check']
old_syn_result=next(i for i,n in enumerate(old_syn.body) if assigned(n,'result'))
assert all(dump(a)==dump(b) for a,b in zip(old_syn.body[:old_syn_result],new_syn.body[:old_syn_result]))
normal.body[normal.body.index(next(n for n in normal.body if isinstance(n,ast.FunctionDef) and n.name=='synthetic_check'))]=copy.deepcopy(old_syn)
assert dump(normal)==dump(old)

# Extract actual v2 helper, plus existing hash/compare helpers only.
ns=dict(np=np,pd=pd,math=math,json=json,hashlib=hashlib,CHECKS=0)
nodes=[copy.deepcopy(nf[name]) for name in ['array_sha','ids_sha','compare','guard_original_r3']]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<actual-r3-guard>', 'exec'),ns)
engine_path=ROOT/'집/코덱스/analysis/ec_stage2_tabpfn_20261002_v2/run_stage2_tabpfn.py'
engine=ast.parse(engine_path.read_text(encoding='utf-8-sig'))
canon=next(n for n in engine.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_hash')
exec(compile(ast.Module(body=[canon],type_ignores=[]),'<actual-original-key-function>','exec'),ns)
tr=pd.DataFrame(dict(row_id=['F13_010_00','F13_010_01','F47_020_00'],
                     farm=['F13','F13','F47'],day=[10,10,20],hour=[0,1,0],sub_ec=[.1,.7,2.]))
q=pd.DataFrame(dict(row_id=['F13_012_00','F47_022_00'],farm=['F13','F47'],
                    day=[12,22],hour=[0,0],sub_ec=[.3,.6]))
labels=pd.concat([tr,q]).set_index('row_id').sub_ec
arrays=dict(row_id=q.row_id.to_numpy(str),train_row_id=tr.row_id.to_numpy(str),
            sub_ec=q.sub_ec.to_numpy(float),raw_et=np.array([.4,.8]),
            raw_lgb=np.array([.5,.9]),raw_mlp=np.array([.6,1.]),train_raw_r3=np.array([.15,.6,1.9]))
arrays['raw_r3']=.6*arrays['raw_et']+.3*arrays['raw_lgb']+.1*arrays['raw_mlp']
prov=dict(train_rows=3,validation_rows=2,train_days=2,validation_days=2,train_target_bounds=[.1,2.],
          synthetic_unicode='합성 구조 검산')
digest='a'*64


def metadata(p=None):
    p=copy.deepcopy(prov) if p is None else p
    payload=p|dict(kind='R3 original .6ET+.3LGB+.1MLP',seed=7)
    return dict(prediction_sha256=digest,kind='r3',seed=7,fold=['DIAG10',0],provenance=p,
                key=ns['canonical_hash'](payload))


def call(a=None,m=None,d=None,t=None,v=None,l=None):
    return ns['guard_original_r3'](arrays if a is None else a,metadata() if m is None else m,
                                  digest if d is None else d,tr if t is None else t,
                                  q if v is None else v,7,'DIAG10',0,labels if l is None else l)


records=[]
valid=call();assert valid['raw_member_arithmetic']=='EXACT_PASS' and valid['train_rows']==3
records.append(dict(case='valid_with_original_canonical_unicode_key',status='PASS'))


def check_bad(name,a=None,m=None,d=None,t=None,v=None,l=None):
    try:call(a,m,d,t,v,l)
    except (AssertionError,KeyError):records.append(dict(case=name,status='REJECTED_AS_EXPECTED'))
    else:raise AssertionError('Contamination accepted: '+name)


a=copy.deepcopy(arrays);a['train_row_id']=a['train_row_id'][::-1];check_bad('train_order',a=a)
a=copy.deepcopy(arrays);a['sub_ec'][0]+=.1;check_bad('stored_target',a=a)
a=copy.deepcopy(arrays);a['raw_r3'][0]+=.1;check_bad('member_arithmetic',a=a)
check_bad('file_sha',d='b'*64)
p=copy.deepcopy(prov);p['train_target_bounds']=[.1,1.9];check_bad('train_target_bounds',m=metadata(p))
for name,value in [('seed',101),('fold',['B',0]),('kind','pfn'),('key','b'*64)]:
    m=metadata();m[name]=value;check_bad('metadata_'+name,m=m)
for name,value in [('train_rows',4),('validation_rows',3),('train_days',3),('validation_days',1)]:
    p=copy.deepcopy(prov);p[name]=value;check_bad('provenance_'+name,m=metadata(p))
a=copy.deepcopy(arrays);a['row_id']=a['row_id'][::-1];check_bad('query_order',a=a)
a=copy.deepcopy(arrays);a['raw_et'][0]=np.nan;check_bad('query_member_nan',a=a)
a=copy.deepcopy(arrays);a['train_raw_r3'][0]=np.inf;check_bad('train_prediction_inf',a=a)
a=copy.deepcopy(arrays);a.pop('raw_mlp');check_bad('required_member_missing',a=a)
a=copy.deepcopy(arrays);a['raw_r3']=a['raw_r3'][:,None];check_bad('wrong_array_shape',a=a)
a=copy.deepcopy(arrays);a['raw_r3'][0]+=1e-15;check_bad('tiny_member_identity_change',a=a)
wrong_labels=labels.copy();wrong_labels.loc[tr.row_id.iloc[1]]+=.1;check_bad('public_train_target',l=wrong_labels)
wrong_labels=labels.copy();wrong_labels.loc[q.row_id.iloc[0]]+=.1;check_bad('public_query_target',l=wrong_labels)
a=copy.deepcopy(arrays);t=tr.copy();t.loc[0,'row_id']=q.row_id.iloc[0];a['train_row_id']=t.row_id.to_numpy(str)
check_bad('train_query_overlap',a=a,t=t)
a=copy.deepcopy(arrays);a['train_row_id'][1]=a['train_row_id'][0];check_bad('duplicate_train_id',a=a)

# Intentionally document an accepted input outside the requested arithmetic guard scope.
a=copy.deepcopy(arrays);a['train_raw_r3']+=100.
call(a=a)
records.append(dict(case='finite_train_raw_not_recomputed_scope_limit',status='ACCEPTED_SCOPE_LIMIT'))

# Allowed single original metadata read: output key/type schema only, no values.
meta_path=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/DIAG10_0_r3_7.json'
existing=json.loads(meta_path.read_text(encoding='utf-8'))
assert {'kind','seed','fold','key','prediction_sha256','provenance'}<=set(existing)
assert {'train_rows','validation_rows','train_days','validation_days','train_target_bounds','shared','environment'}<=set(existing['provenance'])
schema=dict(top={k:type(v).__name__ for k,v in existing.items()},
            provenance={k:type(v).__name__ for k,v in existing['provenance'].items()},
            fold_length=len(existing['fold']),train_target_bounds_length=len(existing['provenance']['train_target_bounds']))
assert schema['fold_length']==2 and schema['train_target_bounds_length']==2
result=dict(status='PASS_DELTA_AND_SYNTHETIC_ONLY',verifier_v1_sha256=sha(EXP/'verify_full_v1.py'),
            verifier_v2_sha256=sha(EXP/'verify_full_v2.py'),normalized_full_ast_equal=True,
            unchanged_functions=unchanged,new_guard='guard_original_r3',
            source_constants_and_main_ast_unchanged=True,
            cases=records,case_count=len(records),reject_count=sum(r['status']=='REJECTED_AS_EXPECTED' for r in records),
            comparisons=ns['CHECKS'],one_original_metadata_schema=schema,
            actual_tabdpt_output_reads=0,actual_data_or_score_reads=0,original_npz_reads=0,
            original_metadata_structure_reads=1,actual_fit=0,actual_predict=0,full_verify_runs=0,
            limits=['guard compares the supplied digest; whole-verifier path computes actual NPZ SHA before load, tested by source inspection only',
                    'metadata key is a consistency digest, not an independent original fit replay',
                    'finite train_raw_r3 is shape/finite checked; not decomposed or refitted and not used in candidate query predictions',
                    'same season/core and stored first audit scope limits from v18 remain unchanged'])
with RESULT.open('x',encoding='utf-8') as h:json.dump(result,h,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
