"""Independent root review: source and scalar vs vector exact penalized roots.
Synthetic only; no EC/model/DLL fitting or original environment modification.
"""
from pathlib import Path
import sys, json, math, hashlib, random
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def root(a,b):
    def g(t):return -a*math.exp(-t/2)+b*math.exp(t/2)+t
    lo,hi=-1.,1.
    while g(lo)>0:lo*=2
    while g(hi)<0:hi*=2
    for _ in range(120):
        t=(lo+hi)/2
        if g(t)>0:hi=t
        else:lo=t
    return (lo+hi)/2

def main():
    src=(H/'exact_leaf_class_v1.hpp').read_text(encoding='utf-8')
    required=['config.tweedie_variance_power==1.5','config.lambda_l2==1.0',
      'config.lambda_l1==0.0','config.device_type=="cpu"',
      'weights_==nullptr?1.0:static_cast<double>(weights_[j])',
      'bagging_mapper[index_mapper[k]]','F=y-getter(label_,j)',
      'A+=w*y*std::exp(-0.5*F);B+=w*std::exp(0.5*F)',
      '-A*std::exp(-0.5*t)+B*std::exp(0.5*t)+t','return root;']
    assert all(s in src for s in required)
    rng=random.Random(20261004);cases=[]
    for k in range(200):
        y=[0. if k==0 else rng.random()*4 for _ in range(43)]
        f=[rng.uniform(-5,3) for _ in y];w=[rng.uniform(.1,2) for _ in y]
        a=math.fsum(wi*yi*math.exp(-fi/2) for yi,fi,wi in zip(y,f,w))
        b=math.fsum(wi*math.exp(fi/2) for fi,wi in zip(f,w))
        t=root(a,b);newton=(a-b)/(.5*(a+b)+1)
        def loss(t):
            return math.fsum(2*wi*(yi*math.exp(-(fi+t)/2)+math.exp((fi+t)/2)) for yi,fi,wi in zip(y,f,w))+t*t/2
        g=math.fsum(wi*(-yi*math.exp(-(fi+t)/2)+math.exp((fi+t)/2)) for yi,fi,wi in zip(y,f,w))+t
        assert abs(g)<=1e-10*max(1.,a+b)
        assert loss(t)<=loss(newton)+1e-10*max(1.,loss(t))
        assert loss(t)<=loss(0)+1e-10*max(1.,loss(t))
        # Independent damped Newton solve from 0, monotone derivative and line search.
        q=0.
        for _ in range(100):
            n=a*math.exp(-q/2);p=b*math.exp(q/2)
            grad=-n+p+q;hess=(n+p)/2+1
            step=grad/hess;scale=1.
            while loss(q-scale*step)>loss(q)+1e-12:scale/=2
            q-=scale*step
            if abs(grad)<=1e-13*max(1.,n+p+abs(q)):break
        assert abs(q-t)<1e-10,(k,q,t)
        cases.append(dict(k=k,root=t,newton=newton,residual=g,independent_root=q,
          initial_loss=loss(0),exact_loss=loss(t),newton_loss=loss(newton)))
    ref=json.loads((H/'overlay_manifest_v1.json').read_text())
    assert ref['class_sha256']==sha(H/'exact_leaf_class_v1.hpp')
    out=dict(status='PASS_SYNTHETIC_SOURCE_AND_MATH',cases=cases,source_sha256=sha(H/'exact_leaf_class_v1.hpp'),
       reviewer_sha256=sha(Path(__file__)),fit=0,EC_reads=0,limitations=['No native DLL build equivalence or actual inbag score trajectory verified here',
       'Tree structure still follows original Newton gradient/Hessian split criterion; this modifies leaf values only',
       'Regularized optimum is before unchanged .03 shrinkage, not an exact global boosted-ensemble optimizer'])
    with (H/'math_review_result_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print('PASS_SYNTHETIC_SOURCE_AND_MATH',len(cases))
if __name__=='__main__':main()
