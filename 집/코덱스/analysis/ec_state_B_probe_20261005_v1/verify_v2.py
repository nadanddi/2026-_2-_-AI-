"""Independent scalar predictions, loss/segments and training exclusion checks."""
import run_v2 as T
from pathlib import Path
import math,json
np,pd=T.np,T.pd;H=T.H
def manual(x,m):
    if m['mode']=='MEAN':return np.full(len(x),m['mean'])
    return np.array([math.fsum(T.M.tree_value(t['tree_structure'],row) for t in m['tree']['tree_info']) for row in x])
def main():
    T.frozen();refs,p=T.prepare();assert p==T.load(H/'preparation_v1.json');rc=T.load(H/'receipt_v1.json');assert rc['status']=='COMPLETE_B_ONLY_GATE0';models={(r['scope'],r['mode'],r['v'],r['k'],r['s'],r.get('j')):r for r in rc['models']};frames=[];maximum=0.;repeat=0
    for rec in rc['files']:
        fp=T.OUT/rec['path'];assert T.sha(fp)==rec['sha'];f=pd.read_csv(fp,float_precision='round_trip');scope,mode,v,k,s=[rec[n] for n in ['scope','mode','v','k','s']];ref=refs[v,k];q,a,x=T.pair(ref,s,scope!='outer');assert np.array_equal(q.row_id,f.row_id) and np.array_equal(q.sub_ec,f.y);T.close(a,f.A);prediction=np.full(len(q),np.nan)
        iq,ia,ix=T.pair(ref,s,True);parts=[(None,np.arange(len(iq)),np.arange(len(q)))] if scope=='outer' else [(j,*T.D.meta_split(ref,j)) for j in range(3)]
        for j,ti,vi in parts:
            idx=ti if scope!='included_equal_n' else T.equal_train(iq,ti,vi);mr=models[scope,mode,v,k,s,j];assert T.sha(T.OUT/mr['path'])==mr['sha'];m=T.load(T.OUT/mr['path']);assert m['indices']==idx.tolist() and m['ids']==T.R.ids(iq.row_id.iloc[idx]);t=iq.sub_ec.to_numpy()[idx]-ia[idx];assert T.R.ar(t)==m['target_sha'];T.close([math.fsum(map(float,t))/len(t)],[m['mean']])
            if scope=='excluded':
                assert not set(iq.row_id.iloc[idx])&set(q.row_id.iloc[vi]);banned={(ff,int(dd)+delta) for ff,dd in q.iloc[vi][['farm','day']].itertuples(index=False,name=None) for delta in [-1,0,1]};assert not set(iq.iloc[idx][['farm','day']].itertuples(index=False,name=None))&banned
            if scope=='included_equal_n':assert set(vi)<=set(idx) and len(idx)==len(ti)
            bounds=ref['bounds'] if scope=='outer' else ref['innerbounds'];raw=manual(x[vi],m);prediction[vi]=np.array([min(max(float(aa)+min(max(float(rr),-.3),.3),bounds[0]),bounds[1]) for aa,rr in zip(a[vi],raw)]);T.close(raw,T.forward(x[vi],m))
            if (v,k,s,j)==('DIAG10',0,7,None):
                with T.M.threadpool_limits(limits=2):fresh=T.fit(mode,iq,ia,ix,idx,s)
                assert fresh==m;repeat+=1;z=x[:8].copy();z[1:]+=1e4;T.close(T.forward(x[:1],m),T.forward(z,m)[:1])
        maximum=max(maximum,T.close(prediction,f.P));frames.append(f);print('VERIFY_B',scope,mode,v,k,s,flush=True)
    allrows=pd.concat(frames,ignore_index=True);allrows['high']=allrows.groupby(['mode','v','k','s','scope','farm','day']).y.transform('mean')>=1;rows=[]
    for (mode,v,s,scope),g in allrows.groupby(['mode','v','s','scope']):
        masks=[('all',np.ones(len(g),bool)),('high',g.high.to_numpy()),('ordinary',~g.high.to_numpy()),('pass2',(g.day>=179).to_numpy())]+[(f+'_pass'+str(p),((g.farm==f)&((g.day>=179)==(p==2))).to_numpy()) for f in ['F13','F47'] for p in [1,2]]
        for segment,mask in masks:
            z=g[mask]
            if not len(z):continue
            a,y,b=z.A.to_numpy(),z.y.to_numpy(),z.P.to_numpy();sa=math.fsum((float(aa)-float(yy))**2 for aa,yy in zip(a,y));sb=math.fsum((float(bb)-float(yy))**2 for bb,yy in zip(b,y));assert abs(sa-np.sum((a-y)**2))<1e-9 and abs(sb-np.sum((b-y)**2))<1e-9
            rows.append(dict(mode=mode,v=v,s=int(s),scope=scope,segment=segment,n=len(z),rmse_A=math.sqrt(sa/len(z)),rmse_B=math.sqrt(sb/len(z)),change_pct=100*(math.sqrt(sb/sa)-1),mean_residual=float(np.mean(y-a)),mean_correction=float(np.mean(b-a)),opposite_rows=int(((y-a)*(b-a)<0).sum()),needed_up=int((y>a).sum()),both_below=int(((a<y)&(b<y)).sum())))
    df=pd.DataFrame(rows);df.to_csv(H/'scores_v1.csv',index=False)
    outer=df[(df['mode']=='STATE')&(df.scope=='outer')&df.v.isin(['DIAG10','A','B'])&(df.segment=='all')];meta=df[(df['mode']=='STATE')&(df.scope=='excluded')&df.segment.isin(['all','high'])];guard=df[(df['mode']=='STATE')&(df.scope=='outer')&(df.v=='DIAG10')&(df.segment=='pass2')];assert len(outer)==9 and len(meta)==6 and len(guard)==3
    passed=bool((outer.change_pct<0).all() and (meta.change_pct<0).all() and (guard.change_pct<2).all());T.save(H/'verification_v1.json',dict(status='PASS_SCALAR_DATA_EXCLUSION_REPLAY',rows=len(allrows),files=len(rc['files']),models=len(rc['models']),fresh_repeats=repeat,max_replay_error=maximum,outer_direction_pass=int((outer.change_pct<0).sum()),outer_direction_total=9,meta_direction_pass=int((meta.change_pct<0).sum()),meta_direction_total=6,next_gate='REQUIRES_FULL_CROSSFIT_AND_FRESH_CONFIRMATION' if passed else 'STOP_NO_GATE',adoption=False,EL1_scored=False,receipt_sha=T.sha(H/'receipt_v1.json'),scores_sha=T.sha(H/'scores_v1.csv')));print('VERIFIED_B_NEXT',passed,flush=True)
if __name__=='__main__':main()

