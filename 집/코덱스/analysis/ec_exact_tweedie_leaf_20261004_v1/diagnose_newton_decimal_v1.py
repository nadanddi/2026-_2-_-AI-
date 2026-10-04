"""First trace root diagnosis only; no prediction/RMSE/model loading."""
from pathlib import Path
import math,json,hashlib
from decimal import Decimal,localcontext
H=Path(__file__).resolve().parent;OUT=H.parents[3]/'집/코덱스/local'/H.name
def old(A,B):
 x=0.;lo=-64.;hi=64.;trail=[]
 for i in range(100):
  g=-A*math.exp(-x/2)+B*math.exp(x/2)+x
  if g>0:hi=x
  else:lo=x
  h=.5*(A*math.exp(-x/2)+B*math.exp(x/2))+1;proposed=x-g/h
  nextx=proposed if lo<proposed<hi else (lo+hi)/2
  trail.append(dict(iteration=i,x=x,g=g,h=h,step=g/h,proposed=proposed,lo=lo,hi=hi,nextx=nextx,bisect=not lo<proposed<hi));x=nextx
 return x,trail
def repaired(A,B):
 x=0.;lo=-64.;hi=64.
 for i in range(100):
  g=-A*math.exp(-x/2)+B*math.exp(x/2)+x;h=.5*(A*math.exp(-x/2)+B*math.exp(x/2))+1;proposed=x-g/h
  if abs(g/h)<=1e-16 or proposed==x:return x,i+1
  if g>0:hi=x
  else:lo=x
  x=proposed if lo<proposed<hi else (lo+hi)/2
 return x,100
def decimalroot(A,B):
 with localcontext() as c:
  c.prec=80;A=Decimal(str(A));B=Decimal(str(B));lo=Decimal(-64);hi=Decimal(64)
  for i in range(270):
   x=(lo+hi)/2;g=-A*(-x/2).exp()+B*(x/2).exp()+x
   if g>0:hi=x
   else:lo=x
  x=(lo+hi)/2
  return str(x),str(-A*(-x/2).exp()+B*(x/2).exp()+x)
trace=OUT/'first_exact_cpp_trace_v1.ndjson';bad=None;maxgap=0.;n=0;iteration=-1
for line in trace.open(encoding='utf-8'):
 leaf=json.loads(line)
 if leaf['kind']=='bag':iteration=leaf['iteration'];continue
 n+=1;A=math.fsum(float(y)*math.exp(-float(f)/2) for y,f in zip(leaf['y'],leaf['F']));B=math.fsum(math.exp(float(f)/2) for f in leaf['F']);z=leaf['root'];new,steps=repaired(A,B);gap=abs(new-z);maxgap=max(maxgap,gap)
 assert gap<=1e-12
 if bad is None:
  previous,trail=old(A,B)
  if abs(previous-z)>1e-12:
   root,residual=decimalroot(A,B);native_root,native_residual=decimalroot(leaf['A'],leaf['B'])
   with localcontext() as c:
    c.prec=80;highA=sum((Decimal(str(y))*(-Decimal(str(f))/2).exp() for y,f in zip(leaf['y'],leaf['F'])),Decimal(0));highB=sum(((Decimal(str(f))/2).exp() for f in leaf['F']),Decimal(0))
   highroot,highres=decimalroot(highA,highB)
   bad=dict(tree_iteration=iteration,leaf_ordinal=n,rows=len(leaf['ids']),A=A,B=B,native_A=leaf['A'],native_B=leaf['B'],z=z,old100=previous,old_gap=abs(previous-z),new=new,new_gap=gap,new_iterations=steps,decimal80_fsum_root=root,decimal80_fsum_residual=residual,decimal80_native_AB_root=native_root,decimal80_native_AB_residual=native_residual,decimal80_exp_AB_root=highroot,decimal80_exp_AB_residual=highres,decimal_root_gap=abs(float(root)-z),old_trail=trail)
assert bad is not None
for key in ['decimal80_fsum_root','decimal80_native_AB_root','decimal80_exp_AB_root']:assert abs(float(bad[key])-bad['z'])<=1e-12
r=dict(status='PROVED_INDEPENDENT_NEWTON_TERMINATION_BUG',first_failed=bad,leaves_checked=n,new_all_leaf_max_gap=maxgap,original_root_atol=1e-12,trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_fit=0,native_predict=0,outer_scores=0,raw_ec_reads=0,test_reads=0,EL1_reads=0)
with (H/'diagnose_newton_decimal_result_v1.json').open('x',encoding='utf-8') as out:json.dump(r,out,indent=2)
print(r['status'],n,maxgap,bad['decimal80_fsum_root'],bad['decimal80_exp_AB_root'])
