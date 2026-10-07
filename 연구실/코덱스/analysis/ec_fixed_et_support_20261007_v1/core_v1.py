import math
import numpy as np

GROUPS={
 'season':['season'], 'temperature':['in_temp','in_temp_h0'],
 'humidity':['in_hum','in_hum_h0'], 'co2_sensor':['in_co2','in_co2_h0'],
 'ventilation':['act_vent','act_vent_h0','act_vent_tdm','act_vent_tdz','seal_run','vent_open_hours','first_open_hour','vent_max'],
 'heating':['act_heating','act_heating_h0','act_heating_tdm','act_heating_tdz','heat_run'],
 'curtains':[v+s for v in ['act_shade','act_thermal'] for s in ['', '_h0','_tdm','_tdz']]+['thermal_switches','shade_switches','since_curtain_change'],
 'co2_action':['act_co2','act_co2_h0','act_co2_tdm','act_co2_tdz','co2_hours'],
 'fan':['act_circfan','act_circfan_h0','act_circfan_tdm','act_circfan_tdz'],
 'fog':['act_fog','act_fog_h0','act_fog_tdm','act_fog_tdz'],
 'clock':['hr_sin','hr_cos','midnight']}

def alpha(smooth):
    if not smooth:return np.ones(24)/24
    return np.array([.5/24+.5/24*sum(1/(h+1) for h in range(i,24)) for i in range(24)])

def coalitions(control,target,cols):
    flat=sum(GROUPS.values(),[]);assert len(flat)==len(set(flat))==len(cols)==47 and set(flat)==set(cols)
    active=[g for g,cc in GROUPS.items() if not np.array_equal(control[:,[cols.index(c) for c in cc]],target[:,[cols.index(c) for c in cc]],equal_nan=True)]
    assert 'clock' not in active and 1<=len(active)<=10
    out=np.tile(control,(1<<len(active),1,1))
    for mask in range(len(out)):
        for i,g in enumerate(active):
            if mask&(1<<i):
                ix=[cols.index(c) for c in GROUPS[g]];out[mask][:,ix]=target[:,ix]
    assert np.array_equal(out[0],control,equal_nan=True) and np.array_equal(out[-1],target,equal_nan=True)
    return active,out

def aggregate(train_leaf,query_leaf,smooth):
    assert len(query_leaf)%24==0
    w=np.zeros((len(query_leaf)//24,len(train_leaf)))
    a=alpha(smooth);nt=train_leaf.shape[1];qi=np.arange(len(query_leaf))
    for j in range(nt):
        order=np.argsort(train_leaf[:,j]);v=train_leaf[order,j]
        lo=np.searchsorted(v,query_leaf[:,j],'left');hi=np.searchsorted(v,query_leaf[:,j],'right');n=hi-lo
        assert np.min(n)>0
        ix=np.repeat(qi,n);offset=np.arange(n.sum())-np.repeat(np.cumsum(n)-n,n)
        ti=order[np.repeat(lo,n)+offset]
        np.add.at(w,(ix//24,ti),a[ix%24]/(nt*n[ix]))
    assert np.max(abs(w.sum(axis=1)-1))<1e-12
    return w

def shapley(v,n):
    v=np.asarray(v);out=np.zeros(n)
    for i in range(n):
        for mask in range(1<<n):
            if mask&(1<<i):continue
            k=mask.bit_count();out[i]+=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)*(v[mask|(1<<i)]-v[mask])
    assert abs(out.sum()-(v[-1]-v[0]))<1e-10
    return out

class Forest:
    def __init__(self,z):self.z=z
    def apply(self,X):
        X=np.asarray(X,np.float32);z=self.z;out=np.empty((len(X),len(z['offsets'])-1),np.int32)
        for j,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
            node=np.zeros(len(X),np.int32)
            while True:
                live=z['left'][a+node]!=-1
                if not live.any():break
                rows=np.flatnonzero(live);loc=a+node[rows];f=z['feature'][loc]
                node[rows]=np.where(X[rows,f]<=z['threshold'][loc],z['left'][loc],z['right'][loc])
            out[:,j]=node
        return out
    def predict(self,X):
        leaf=self.apply(X);return self.z['value'][leaf+self.z['offsets'][:-1]].mean(axis=1)

def pack(model,X,y,ids,median):
    tt=[e.tree_ for e in model.estimators_];offsets=np.r_[0,np.cumsum([t.node_count for t in tt])]
    return dict(offsets=offsets,left=np.concatenate([t.children_left for t in tt]),right=np.concatenate([t.children_right for t in tt]),feature=np.concatenate([t.feature for t in tt]),threshold=np.concatenate([t.threshold for t in tt]),value=np.concatenate([t.value[:,0,0] for t in tt]),n_samples=np.concatenate([t.n_node_samples for t in tt]),weighted_n_samples=np.concatenate([t.weighted_n_node_samples for t in tt]),train_X=X,train_y=y,train_row_id=ids,imputer_median=median,train_leaf=model.apply(X))
