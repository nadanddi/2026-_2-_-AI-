"""Readonly failure diagnosis; no candidate outer scores/model prediction."""
from pathlib import Path
import math,json,hashlib
H=Path(__file__).resolve().parent
trace=H.parents[3]/'집/코덱스/local'/H.name/'first_exact_cpp_trace_v1.ndjson'
bad=None
for line in trace.open(encoding='utf-8'):
    leaf=json.loads(line)
    if leaf['kind']!='leaf':continue
    A=math.fsum(y*math.exp(-f/2) for y,f in zip(leaf['y'],leaf['F']))
    B=math.fsum(math.exp(f/2) for f in leaf['F']);z=leaf['root']
    x=0.;lo=-64.;hi=64.;history=[]
    for i in range(100):
        g=-A*math.exp(-x/2)+B*math.exp(x/2)+x
        if g>0:hi=x
        else:lo=x
        h=.5*(A*math.exp(-x/2)+B*math.exp(x/2))+1
        proposed=x-g/h
        xx=proposed if lo<proposed<hi else (lo+hi)/2
        history.append(dict(i=i,x=x,g=g,h=h,proposal=proposed,next=xx,lo=lo,hi=hi))
        x=xx
    if abs(x-z)>1e-12:
        q=0.;ql=-64.;qh=64.
        for i in range(100):
            g=-A*math.exp(-q/2)+B*math.exp(q/2)+q
            h=.5*(A*math.exp(-q/2)+B*math.exp(q/2))+1
            if abs(g/h)<=1e-16:break
            if g>0:qh=q
            else:ql=q
            p=q-g/h;q=p if ql<p<qh else (ql+qh)/2
        bad=dict(status='DIAGNOSED_FIRST_FAILURE_ONLY',A=A,B=B,cpp_root=z,old_newton=x,old_gap=abs(x-z),terminated_newton=q,new_gap=abs(q-z),new_iters=i,old_last15=history[-15:],trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),actual_score=0,fit=0,predict=0)
        break
assert bad is not None
with (H/'diagnose_newton_root_result_v1.json').open('x',encoding='utf-8') as f:json.dump(bad,f,indent=2)
print(json.dumps({k:v for k,v in bad.items() if k!='old_last15'}))
