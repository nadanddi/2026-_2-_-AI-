"""Audit MLP optimization precisely; never relax the original convergence criterion."""
import numpy as np,math
from scipy.special import expit
from scipy.optimize import minimize
def pack(m):return np.r_[np.asarray(m['coefs'][0]).ravel(),np.asarray(m['coefs'][1]).ravel(),np.asarray(m['intercepts'][0]),np.asarray(m['intercepts'][1])]
def unpack(t):return t[:184].reshape(23,8),t[184:192].reshape(8,1),t[192:200],t[200:201]
def fg(theta,z,target,weight,alpha=10):
    w0,w1,b0,b1=unpack(theta);h=np.tanh(z@w0+b0);logit=(h@w1+b1).reshape(-1);p=expit(logit);n=weight.sum();value=np.sum(weight*(np.logaddexp(0,logit)-target*logit))/n+alpha*(np.sum(w0*w0)+np.sum(w1*w1))/(2*n);error=(p-target)*weight/n;dh=error[:,None]*w1.T*(1-h*h);grad=np.r_[(z.T@dh+alpha*w0/n).ravel(),(h.T@error[:,None]+alpha*w1/n).ravel(),dh.sum(0),error.sum()];return float(value),grad
def toy():
    rng=np.random.default_rng(813);z=rng.normal(size=(27,23));target=rng.integers(2,size=27);weight=rng.uniform(.1,2,size=27);t=rng.normal(0,.1,201);f,g=fg(t,z,target,weight);errors=[]
    for j in range(201):
        e=np.zeros(201);e[j]=1e-6;fd=(fg(t+e,z,target,weight)[0]-fg(t-e,z,target,weight)[0])/2e-6;errors.append(abs(fd-g[j]))
    assert max(errors)<1e-8;return max(errors)
def data(D,ref,s,m):
    q,a,b,x,ok=D.B.pair(ref,s,True);y=q.sub_ec.to_numpy();cost=(a-y)**2-(b-y)**2;eligible=ok&(abs(cost)>D.M.CFG['tie_epsilon']);assert D.R.ids(q.row_id[eligible])==m['eligible_ids'];assert D.R.ar(cost)==m['cost_sha'];assert eligible.sum()==m['n']
    if not m['n']:assert m['constant']==0;return None
    t=(cost[eligible]>0).astype(int);w=abs(cost[eligible]);w/=w.mean();assert D.R.ar(t)==m['label_sha'] and D.R.ar(w)==m['weight_sha'];xx=x[eligible];means=np.array([math.fsum(map(float,col))/len(col) for col in xx.T]);sd=np.array([math.sqrt(math.fsum((float(v)-float(mu))**2 for v in col)/len(col)) for mu,col in zip(means,xx.T)]);sd[sd==0]=1;assert np.max(abs(means-m['mean']))<1e-8 and np.max(abs(sd-m['scale']))<1e-8;z=(xx-np.asarray(m['mean']))/m['scale'];g=D.M.forward('MLP',xx,m);D.close([np.mean((a[eligible]+g*(b[eligible]-a[eligible])-y[eligible])**2)],[m['train_mse']]);D.close([np.mean((a[eligible]-y[eligible])**2)],[m['train_a_mse']]);cl=np.clip(g,1e-15,1-1e-15);ll=math.fsum(float(ww)*(-float(tt)*math.log(float(gg))-(1-float(tt))*math.log(1-float(gg))) for ww,tt,gg in zip(w,t,cl))/math.fsum(map(float,w));assert abs(ll-m['train_logloss'])<1e-10
    if m['constant'] is not None:return None
    value,grad=fg(pack(m),z,t,w);assert abs(value-m['loss'])<1e-10;return z,t,w,pack(m),value,grad
def check(D,ref,s,m):
    items=data(D,ref,s,m);return float(np.max(abs(items[-1]))) if items is not None else 0.
def refine(D,ref,s,m):
    items=data(D,ref,s,m)
    if items is None:return m.copy(),dict(optimized=False,old_gradient=0.,new_gradient=0.,objective_improvement=0.)
    z,t,w,theta,old_value,old_grad=items;opt=minimize(fg,theta,args=(z,t,w),jac=True,method='L-BFGS-B',options=dict(maxiter=5000,ftol=1e-14,gtol=1e-8,maxls=50));value,grad=fg(opt.x,z,t,w);norm=float(np.max(abs(grad)));assert opt.success and norm<=1e-6,(opt.message,norm);assert value<=old_value+1e-12
    w0,w1,b0,b1=unpack(opt.x);new=m.copy();new.update(coefs=[w0.tolist(),w1.tolist()],intercepts=[b0.tolist(),b1.tolist()]);return new,dict(optimized=True,old_gradient=float(np.max(abs(old_grad))),new_gradient=norm,old_objective=old_value,new_objective=value,objective_improvement=old_value-value,iterations=int(opt.nit),message=str(opt.message))
