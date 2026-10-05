import run_v1 as T
import math
from sklearn.metrics import roc_auc_score
np,pd=T.np,T.pd;H=T.H
def scalar(x,m):
    if m['constant'] is not None:return np.full(len(x),m['constant'])
    values=[]
    for row in x:
        z=math.fsum((float(v)-mu)/sd*c for v,mu,sd,c in zip(row,m['mean'],m['scale'],m['coef']))+m['intercept'];values.append(1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z)))
    return np.array(values)
def auc(y,p):
    if not 0<int(sum(y))<len(y):return None
    order=sorted(range(len(y)),key=lambda i:float(p[i]));positive_rank=0.;i=0
    while i<len(order):
        j=i+1
        while j<len(order) and p[order[j]]==p[order[i]]:j+=1
        rank=(i+1+j)/2;positive_rank+=sum(int(y[order[k]]) for k in range(i,j))*rank;i=j
    n=int(sum(y));a=(positive_rank-n*(n+1)/2)/(n*(len(y)-n));assert abs(a-roc_auc_score(y,p))<1e-12;return a
def metrics(y,p,selected):
    y=np.asarray(y,int);chosen=np.asarray(selected,bool);tp=int(sum((y==1)&chosen));fn=int(sum((y==1)&~chosen));fp=int(sum((y==0)&chosen));tn=int(sum((y==0)&~chosen));assert tp+fn+fp+tn==len(y)
    return dict(n=len(y),high=tp+fn,tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None,auc=auc(y,p),brier=math.fsum((float(v)-int(t))**2 for v,t in zip(p,y))/len(y))
def learning(q,x,m):
    ix=np.array(m['indices']);y=T.target(q)[ix];assert m['ids']==T.R.ids(q.row_id.iloc[ix]) and m['labels_sha']==T.R.ar(y) and len(y)==m['n']
    if m['constant'] is not None:assert len(set(y))==1 and m['constant']==float(y[0]);return 0.
    xx=x[ix];mean=np.array([math.fsum(float(v) for v in xx[:,j])/len(xx) for j in range(xx.shape[1])]);sd=np.sqrt(np.array([math.fsum((float(v)-float(mu))**2 for v in xx[:,j])/len(xx) for j,mu in enumerate(mean)]));sd[sd==0]=1;T.close(mean,m['mean']);T.close(sd,m['scale']);z=(xx-mean)/sd;p=scalar(xx,m);n=len(y);w=np.where(y==1,n/(2*sum(y)),n/(2*(n-sum(y))));g=((w*(p-y))@z+np.array(m['coef']))/sum(w);gb=math.fsum(float(ww)*(float(pp)-int(yy)) for ww,pp,yy in zip(w,p,y))/sum(w);norm=max(float(np.max(abs(g))),abs(gb));assert norm<1e-5,norm;return norm
def main():
    T.frozen();refs,bank,pr=T.prep();assert pr==T.load(H/'preparation_v1.json');rc=T.load(H/'receipt_v1.json');frames=[];maximum=0.;grad=0.;checked=set();repeats=0
    for rec in rc['manifest']:
        mode,k,s,context=[rec[n] for n in ['mode','k','s','context']];fp=T.OUT/rec['path'];mp=T.OUT/rec['model'];assert T.sha(fp)==rec['sha'] and T.sha(mp)==rec['model_sha'];f=pd.read_csv(fp,float_precision='round_trip');bundle=T.load(mp);q,a,x,sources=bank[k,s,mode,context=='meta'];iq,ia,ix,inner_sources=bank[k,s,mode,True];assert np.array_equal(q.row_id,f.row_id) and np.array_equal(T.target(q),f.high);T.close(a,f.A);T.close(T.R.prefix(q,a),f.prefix_A)
        if context=='outer':p=scalar(x,bundle['full'])
        else:
            p=np.full(len(q),np.nan)
            for j,m in enumerate(bundle['meta']):
                ti,vi=T.D.meta_split(refs['DIAG10',k],j);assert m['indices']==ti.tolist() and not set(ti)&set(vi);forbidden={(ff,int(dd)+o) for ff,dd in q.iloc[vi][['farm','day']].itertuples(index=False,name=None) for o in [-1,0,1]};assert not set(q.iloc[ti][['farm','day']].itertuples(index=False,name=None))&forbidden;p[vi]=scalar(x[vi],m);grad=max(grad,learning(q,x,m))
            assert np.isfinite(p).all();T.close([T.threshold(q,p)],[bundle['threshold']])
        maximum=max(maximum,T.close(p,f.p));assert (f.threshold==bundle['threshold']).all()
        if (mode,k,s) not in checked:
            grad=max(grad,learning(iq,ix,bundle['full']));assert bundle['full']['indices']==list(range(len(iq)));checked.add((mode,k,s))
            rr=refs['DIAG10',k]['a' if context=='meta' else 'tr'];refkeys=set(rr[['farm','day']].itertuples(index=False,name=None));qkeys=set(q[['farm','day']].itertuples(index=False,name=None))
            for z in sources:
                farm=z['row_id'][:3];day=int(z['row_id'][4:7]);assert all((farm,d) in refkeys and (farm,d) not in qkeys for d in z['days']);assert mode=='ALL' or all(d<day for d in z['days'])
            if (k,s)==(0,7):
                with T.M.threadpool_limits(limits=2):fresh=T.fit(iq,ix,np.arange(len(iq)),s)
                assert fresh==bundle['full'];repeats+=1
        frames.append(f);print('CLASS_VERIFY',mode,k,s,context,flush=True)
    allrows=pd.concat(frames,ignore_index=True);results=[]
    for (mode,s,context),g in allrows.groupby(['mode','s','context']):
        if context=='outer':assert g.row_id.is_unique and len(g)==8640
        for hour in [0,6,12,23]:
            q=g[g.hour==hour];y=q.high.to_numpy();p=q.p.to_numpy();cut=q.threshold.to_numpy();aa=q.prefix_A.to_numpy();d=metrics(y,p,p>=cut);base=metrics(y,aa,aa>=.9);results.append(dict(mode=mode,seed=int(s),context=context,hour=hour,model=d,baseline=base))
    primary=[r for r in results if r['context']=='outer' and r['hour']==23];a={r['seed']:r for r in primary if r['mode']=='ALL'};b={r['seed']:r for r in primary if r['mode']=='PAST'};passes=[]
    for s in T.SEEDS:
        aa,bb,base=a[s]['model'],b[s]['model'],a[s]['baseline'];passes.append(bool(aa['recall']>=bb['recall'] and aa['fp']<bb['fp'] and aa['recall']>=base['recall'] and aa['fp']<base['fp']))
    T.save(H/'verification_v1.json',dict(status='PASS_SCALAR_LEARNING_SPLIT_METRICS',models=240,files=120,max_replay_error=maximum,max_gradient=grad,repeats=repeats,results=results,primary_pass_count=sum(passes),primary_total=3,decision='PUBLIC_CLUE_REQUIRES_FRESH_CONFIRMATION' if all(passes) else 'NO_PROVEN_CLASSIFIER_GAIN',adoption=False,limitations=['inner meta y selects threshold: meta threshold metrics are calibration not final test','same public days reused','not a specialist blend RMSE evaluation','official extra PDF not locally available'],receipt_sha=T.sha(H/'receipt_v1.json')));print('CLASS_VERIFY_PASS',sum(passes),flush=True)
if __name__=='__main__':main()
