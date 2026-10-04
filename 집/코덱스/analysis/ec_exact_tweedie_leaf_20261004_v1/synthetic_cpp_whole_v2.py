"""Synthetic trace / fake checkpoint adversarial check: fit0 real-data0."""
from pathlib import Path
import sys,json,math,copy
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
from whole_cpp_recheck_v2 import independent_cpp
tr=pd.DataFrame({'sub_ec':np.array([.2,.4,.6,.8,1.,1.2,1.4,1.6],np.float32),'x':np.arange(8)})
target=tr.sub_ec.to_numpy().astype(np.float32).astype(float);bias=math.log(math.fsum(target)/len(target))
member=np.array([0,0,0,0,1,1,1,1]);bag=[0,1,3,4,6,7];records=[dict(kind='bag',iteration=0,ids=bag)];values={}
for li in [0,1]:
 ids=[i for i in bag if member[i]==li];A=math.fsum(target[i]*math.exp(-bias/2) for i in ids);B=len(ids)*math.exp(bias/2)
 lo,hi=-64.,64.
 for _ in range(120):
  m=(lo+hi)/2
  if -A*math.exp(-m/2)+B*math.exp(m/2)+m>0:hi=m
  else:lo=m
 root=(lo+hi)/2;values[li]=bias+.03*root
 records.append(dict(kind='leaf',ids=ids,y=target[ids].tolist(),F=[bias]*len(ids),w=[1.]*len(ids),A=A,B=B,root=root,newton=(A-B)/(.5*(A+B)+1)))
class Fake:
 def dump_model(self):return {'tree_info':[dict(num_leaves=2,tree_structure=dict(left_child=dict(leaf_index=0,leaf_value=values[0]),right_child=dict(leaf_index=1,leaf_value=values[1])))]}
 def predict(self,X,pred_leaf=False,raw_score=False):return member[:,None] if pred_leaf else np.asarray([values[int(i)] for i in member])
path=H/'synthetic_cpp_whole_trace_v2.ndjson'
with path.open('x',encoding='utf-8') as f:f.write('\n'.join(json.dumps(r) for r in records))
result=independent_cpp(path,Fake(),tr,['x']);rejected=[]
for kind in ['missingbag','duplicate','root','score','label','weight','sum','iteration']:
 bad=copy.deepcopy(records)
 if kind=='missingbag':bad[0]['ids']=bad[0]['ids'][1:]
 elif kind=='duplicate':bad[1]['ids'][1]=bad[1]['ids'][0]
 elif kind=='root':bad[1]['root']+=.01
 elif kind=='score':bad[1]['F'][0]+=.01
 elif kind=='label':bad[1]['y'][0]+=.01
 elif kind=='weight':bad[1]['w'][0]=2.
 elif kind=='sum':bad[1]['A']+=.01
 else:bad[0]['iteration']=1
 p=H/f'synthetic_cpp_corrupt_{kind}_v2.ndjson'
 with p.open('x',encoding='utf-8') as f:f.write('\n'.join(json.dumps(r) for r in bad))
 try:independent_cpp(p,Fake(),tr,['x'])
 except AssertionError:rejected.append(kind)
 else:raise AssertionError(('accepted',kind))
with (H/'synthetic_cpp_whole_v2.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_SYNTHETIC_ONLY',result=result,rejected=rejected,fit=0,real_predict=0,real_score=0),f,indent=2)
print('CPP_WHOLE_SYNTHETIC_PASS',len(rejected))
