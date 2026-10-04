"""Algebra on synthetic endpoint fixtures only: no fit or real prediction scoring."""
from pathlib import Path
import json, math
from decimal import Decimal, localcontext

OUT=Path(__file__).resolve().parent
lo,hi=.06,3.2
base=[lo,hi,.8,1.5,.9]
full=[hi,lo,1.8,.2,hi]
clip=lambda x:max(lo,min(hi,x))
convex=[]
for w in [0.,.125,.37,1.]:
    p=[a+w*(b-a) for a,b in zip(base,full)]
    clipped=[clip(x) for x in p]
    assert all(lo-1e-15<=x<=hi+1e-15 for x in p)
    assert max(abs(a-b) for a,b in zip(p,clipped))<1e-15
    with localcontext() as ctx:
        ctx.prec=40
        q=[Decimal(str(a))+Decimal(str(w))*(Decimal(str(b))-Decimal(str(a))) for a,b in zip(base,full)]
        assert max(abs(a-float(b)) for a,b in zip(p,q))<1e-15
    convex.append(dict(weight=w,clip_maxdiff=max(abs(a-b) for a,b in zip(p,clipped))))

checks=[]
for name,e,d in [('interior',[-.12,0.,.02,0.],[.2,0.,.1,0.]),
                 ('zero',[.1,.2],[.2,.1]),('upper',[-2.,-2.],[.2,.1]),
                 ('zero_direction',[.2,-.3],[0.,0.])]:
    n=len(e);num=math.fsum(a*b for a,b in zip(e,d))/n
    den=math.fsum(x*x for x in d)/n+.01
    w=max(0.,min(1.,-num/den));grad=2*(num+den*w)
    with localcontext() as ctx:
        ctx.prec=40
        ee=list(map(lambda x:Decimal(str(x)),e));dd=list(map(lambda x:Decimal(str(x)),d))
        nn=sum((a*b for a,b in zip(ee,dd)),Decimal(0))/Decimal(n)
        vv=sum((a*a for a in dd),Decimal(0))/Decimal(n)+Decimal('.01')
        dw=max(Decimal(0),min(Decimal(1),-nn/vv))
        assert abs(w-float(dw))<1e-12
    assert (w==0 and grad>=-1e-12) or (w==1 and grad<=1e-12) or abs(grad)<1e-12
    mse0=math.fsum(x*x for x in e)/n
    msew=math.fsum((a+w*b)**2 for a,b in zip(e,d))/n
    assert msew+.01*w*w<=mse0+1e-12
    checks.append(dict(case=name,weight=w,gradient=grad,synthetic_mse0=mse0,synthetic_msew=msew))

# Scaling an unclipped correction before clip differs from endpoint mixing.
clamp=lambda x:max(0.,min(1.,x))
bb=.9;proposal=1.5;ww=.5
endpoint=clamp(bb+ww*(clamp(proposal)-bb))
raw_delta_first=clamp(bb+ww*(proposal-bb))
assert abs(endpoint-.95)<1e-12 and abs(raw_delta_first-1.)<1e-12
with localcontext() as ctx:
    ctx.prec=40
    assert Decimal('.9')+Decimal('.5')*(Decimal('1')-Decimal('.9'))==Decimal('.95')

record=dict(status='PASS',scope='Synthetic algebra only; no data/model import, fit, real prediction or new validation score',
            convex_cases=convex,weight_cases=checks,
            clipping_order_example=dict(endpoint_mixture=endpoint,raw_delta_then_clip=raw_delta_first))
(OUT/'log_nested_design_check_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
