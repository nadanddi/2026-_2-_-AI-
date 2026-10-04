"""최종 clipped prefix squared loss의 미분·leaf coupling 합성 검증만.
실제 데이터/모델/package LightGBM import/fit/predict/score 없음.
"""
from pathlib import Path
import numpy as np,math,json,hashlib
H=Path(__file__).resolve().parent;C=.24
def smooth(n):return .5*np.eye(n)+.5*np.tril(np.ones((n,n)))/np.arange(1,n+1)[:,None]
def block_s(lengths):
    n=sum(lengths);a=np.zeros((n,n));i=0
    for k in lengths:a[i:i+k,i:i+k]=smooth(k);i+=k
    return a
def loss_scalar(f,b,y,lengths,lo,hi):
    i=0;terms=[]
    for n in lengths:
        hist=[]
        for j in range(i,i+n):
            raw=float(b[j])+C*math.exp(float(f[j]));hist.append(raw)
            pred=min(hi,max(lo,.5*raw+.5*math.fsum(hist)/len(hist)))
            terms.append(.5*(pred-float(y[j]))**2)
        i+=n
    return math.fsum(terms)
def derivatives(f,b,y,s,lo,hi):
    d=C*np.exp(f);a=s@(b+d);z=np.clip(a,lo,hi);r=z-y
    active=(a>lo)&(a<hi);g=d*(s.T@(active*r));j=s*d[None,:]
    gn=j.T@(active[:,None]*j);h=gn+np.diag(g)
    return .5*float(r@r),g,h,gn,active,a,z
def fd_gradient(fun,f,eps=1e-6):
    out=[]
    for i in range(len(f)):
        d=np.zeros(len(f));d[i]=eps;out.append((fun(f+d)-fun(f-d))/(2*eps))
    return np.array(out)
def fd_hessian_loss(fun,f,eps=1e-4):
    n=len(f);h=np.zeros((n,n));base=fun(f)
    for i in range(n):
        di=np.eye(n)[i]*eps;h[i,i]=(fun(f+di)-2*base+fun(f-di))/eps**2
        for k in range(i):
            dk=np.eye(n)[k]*eps
            h[i,k]=h[k,i]=(fun(f+di+dk)-fun(f+di-dk)-fun(f-di+dk)+fun(f-di-dk))/(4*eps**2)
    return h
def near(a,b,tol):
    gap=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert gap<=tol,(gap,tol);return gap
def main():
    target=H/'synthetic_derivatives_result_v1.json';assert not target.exists()
    rng=np.random.default_rng(20261004);lengths=[6,6];s=block_s(lengths)
    f=rng.uniform(-.4,.3,12);b=rng.uniform(.1,.3,12);y=rng.uniform(.03,.18,12)
    cases=[]
    for name,ff,bb,yy,lens,lo,hi in [('interior',f,b,y,lengths,0.,2.),('clipped',np.zeros(6),np.array([2.,0.,0.,-2.,0.,2.]),np.full(6,.2),[3,3],0.,1.)]:
        ss=block_s(lens);loss,g,h,gn,active,a,z=derivatives(ff,bb,yy,ss,lo,hi)
        fun=lambda x:loss_scalar(x,bb,yy,lens,lo,hi)
        near(loss,fun(ff),1e-12);assert np.min(np.minimum(abs(a-lo),abs(a-hi)))>.01
        eg=near(g,fd_gradient(fun,ff),2e-8);eh=near(h,fd_hessian_loss(fun,ff),2e-6)
        assert np.max(abs(h-np.diag(np.diag(h))))>1e-6
        assert np.max(abs(h-gn))>1e-6
        cases.append(dict(name=name,n=len(ff),gradient_maxdiff=eg,hessian_maxdiff=eh,offdiag_max=float(np.max(abs(h-np.diag(np.diag(h))))),chain_term_max=float(np.max(abs(h-gn))),active_count=int(active.sum())))
    # A clipped OUTPUT row does not imply its raw score has zero gradient:
    # raw row0 still changes later active outputs through prefix smoothing.
    _,gg,hh,_,act,_,_=derivatives(np.zeros(3),np.array([2.,0.,0.]),np.full(3,.2),smooth(3),0.,1.)
    assert not act[0] and gg[0]!=0
    # Negative exact log-coordinate Hessian despite nonnegative least-square loss.
    _,ng,nh,ngn,_,_,_=derivatives(np.zeros(1),np.array([.1]),np.array([2.]),np.eye(1),0.,3.)
    assert nh[0,0]<0 and ngn[0,0]>0
    # At upper clipping boundary, unequal one-sided derivatives: no classical Hessian.
    kink=lambda x:loss_scalar(np.array([x]),np.zeros(1),np.array([.1]),[1],0.,.24)
    eps=1e-7;left=(kink(0.)-kink(-eps))/eps;right=(kink(eps)-kink(0.))/eps
    assert abs(left-right)>.01 and right==0.
    # Fixed tree with global leaf parameters. Leaves occur across times and days.
    membership=np.eye(3)[np.array([0,1,0,2,1,2,0,2,1,0,1,2])]
    loss,g,h,gn,_,_,_=derivatives(f,b,y,s,0.,2.)
    leafg=membership.T@g;leafh=membership.T@h@membership
    leaf_fun=lambda theta:loss_scalar(f+membership@theta,b,y,lengths,0.,2.)
    near(leafg,fd_gradient(leaf_fun,np.zeros(3)),2e-8);near(leafh,fd_hessian_loss(leaf_fun,np.zeros(3)),2e-6)
    lgb_diagonal_leaf_h=membership.T@np.diag(h)
    assert np.max(abs(np.diag(leafh)-lgb_diagonal_leaf_h))>1e-6
    assert np.max(abs(leafh-np.diag(np.diag(leafh))))>1e-6
    # Illustrative fixed regularization/damping policy; not a chosen EC hyperparameter.
    lam=.01;mu=max(0.,-float(np.linalg.eigvalsh(leafh).min())+1e-4)
    delta=-np.linalg.solve(leafh+(lam+mu)*np.eye(3),leafg)
    assert leafg@delta<0
    accepted=None
    for k in range(20):
        scale=.5**k
        if leaf_fun(scale*delta)<leaf_fun(np.zeros(3)):
            accepted=dict(scale=scale,old_loss=loss,new_loss=leaf_fun(scale*delta));break
    assert accepted is not None
    # Correct reverse prefix backpropagation differs from using forward S on residual.
    d=C*np.exp(f);a=s@(b+d);r=a-y
    assert np.max(abs(d*(s.T@r)-d*(s@r)))>1e-4
    # Inference causal prefix and interday independence are distinct from training gradient coupling.
    baseline=s@(b+d);changed=b.copy();changed[4:6]+=10
    near((s@(changed+d))[:4],baseline[:4],1e-12);near((s@(changed+d))[6:],baseline[6:],1e-12)
    yy=y.copy();yy[5]+=1
    _,changed_g,_,_,_,_,_=derivatives(f,b,yy,s,0.,2.)
    assert changed_g[0]!=g[0] # future TRAIN labels may affect gradient, not inference inputs.
    result=dict(status='PASS_SYNTHETIC_DERIVATIVES_AND_COUNTEREXAMPLES',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),loss='.5 sum squared final clipped residual',cases=cases,clipped_output0_raw_gradient=float(gg[0]),negative_exact_hessian=float(nh[0,0]),positive_gauss_newton=float(ngn[0,0]),kink_left_derivative=left,kink_right_derivative=right,leaf_hessian=leafh.tolist(),diagonal_surrogate_leaf_hessian=lgb_diagonal_leaf_h.tolist(),joint_step=accepted,fit=0,predict=0,real_score=0,raw_ec_reads=0,test_reads=0,EL1_rescore=0)
    with target.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2)
    print('PASS_SYNTHETIC_DERIVATIVES_AND_COUNTEREXAMPLES',cases)
if __name__=='__main__':main()
