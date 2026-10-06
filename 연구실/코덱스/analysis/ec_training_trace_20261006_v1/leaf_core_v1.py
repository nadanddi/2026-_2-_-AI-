import math
import numpy as np

def leaf_weights(train_leaf,query_leaf):
    t=np.asarray(train_leaf);q=np.asarray(query_leaf)
    if t.ndim!=2 or q.ndim!=2 or t.shape[1]!=q.shape[1] or not t.shape[0] or not t.shape[1]:raise ValueError('Invalid leaf matrix')
    w=np.zeros((len(q),len(t)),float)
    for tree in range(t.shape[1]):
        order=np.argsort(t[:,tree]);values=t[order,tree]
        lo=np.searchsorted(values,q[:,tree],side='left');hi=np.searchsorted(values,q[:,tree],side='right')
        if np.any(lo==hi):raise ValueError('Query leaf has no training support')
        for row,(a,b) in enumerate(zip(lo,hi)):w[row,order[a:b]]+=1/(t.shape[1]*(b-a))
    assert np.all(w>=0) and np.max(abs(w.sum(axis=1)-1))<1e-12
    return w

def weight_shapley(coalition_weights,n):
    w=np.asarray(coalition_weights,float)
    if w.ndim!=2 or len(w)!=(1<<n) or n<1:raise ValueError('Invalid coalition matrix')
    out=np.zeros((n,w.shape[1]),float)
    for i in range(n):
        for mask in range(1<<n):
            if mask&(1<<i):continue
            k=mask.bit_count();coef=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
            out[i]+=coef*(w[mask|(1<<i)]-w[mask])
    assert np.max(abs(out.sum(axis=0)-(w[-1]-w[0])))<1e-12
    assert np.max(abs(out.sum(axis=1)))<1e-12
    return out
