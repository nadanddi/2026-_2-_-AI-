from support import *
from analyze import pfn,prep_z,evaluate
import time

def parity_state(observed,query,pred,delay=0):
    daily=observed.groupby(['farm','day'],sort=True).sub_ec.mean();result=[];sources=[]
    for f,day,p in zip(query.farm,query.day,pred):
        choices=[(int(d),float(y)) for (ff,d),y in daily.items() if ff==f and int(d)<int(day)-delay and (int(d)>=179)==(int(day)>=179) and int(d)%2==int(day)%2]
        if not choices:result.append([0.,0.]);sources.append(-1);continue
        d,y=choices[-1];decay=np.exp(-(int(day)-d)/6);slope=0.
        if len(choices)>1:
            d0,y0=choices[-2];slope=(y-y0)/(d-d0)
        result.append([decay*(y-float(p)),decay*slope]);sources.append(d)
    return np.asarray(result),np.asarray(sources)
def main():
    while not (HERE/'result.json').exists():time.sleep(3)
    original=pd.read_csv(OUT/'oof.csv');lab,_,_,_,_,_=loadtemp();features=lab.set_index('row_id');records=[]
    for (name,k,seed,context),d in original[original.arm=='TBIAS'].groupby(['validator','fold','seed','context'],sort=True):
        prefix=f'T_{name}_{k}';z=dict(np.load(OUT/f'{prefix}_cpu.npz'));seeds=list(range(1,9)) if context=='1-8' else list(range(17,25));p=np.column_stack([z[f'base_{seed}'],z[f'codex_{seed}'],pfn(prefix,seeds,z['row_id'])]);g=z['gate'];base=(np.column_stack([.4+.1*g,.6-.4*g,.3*g])*p).sum(1);a=z['z'];q=conditions(features.loc[d.row_id]);a=np.column_stack([a,a[:,0]*a[:,1]]);q=np.column_stack([q,q[:,0]*q[:,1]]);x,v=prep_z(a,q);reg=Ridge(alpha=100.,fit_intercept=False).fit(x,z['y']-base,sample_weight=z['w']);qg=gate(features.loc[d.row_id]);pred=d.baseline.to_numpy()+qg*np.clip(v@reg.coef_,-.35,.35)
        f=d.copy();f['arm']='TBIAS_PH';f['candidate']=pred;records.append(f);np.savez(OUT/f'{prefix}_{seed}_{context}_extra.npz',coef=reg.coef_,cal_design=x,cal_target=z['y']-base,cal_w=z['w'],outer_design=v,outer_g=qg,outer_id=d.row_id.to_numpy(dtype=str))
    lab,core,_,folds,_=loadec()
    for name,k,tm,vm in folds:
        tr,va=lab[tm],lab[vm];prefix=f'E_{name}_{k}';z=dict(np.load(OUT/f'{prefix}_cpu.npz'));b=pd.DataFrame(dict(farm=z['farm'],day=z['day'],hour=z['hour'],sub_ec=z['y']));a=pd.DataFrame(dict(row_id=z['observed_id'],sub_ec=z['observed_y']));a['farm']=a.row_id.str[:3];a['day']=a.row_id.str[4:7].astype(int);bag=pfn(prefix,[1,2,3,4],z['row_id'])
        for seed in ESEEDS:
            ip=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi']);x,_=parity_state(a,b,ip);reg=Ridge(alpha=100.,fit_intercept=False).fit(x,z['y']-ip);d=original[(original.arm=='ESTATE')&(original.validator==name)&(original.fold==k)&(original.seed==seed)].set_index('row_id').reindex(va.row_id).reset_index();q,s=parity_state(tr,va,d.baseline.to_numpy());qs,_=parity_state(tr,va,d.baseline.to_numpy(),delay=5);pred=np.clip(d.baseline.to_numpy()+np.clip(q@reg.coef_,-.15,.15),tr.sub_ec.min(),tr.sub_ec.max());f=d.copy();f['arm']='ESTATE_PAR';f['candidate']=pred;f['stress_delay5']=np.clip(d.baseline.to_numpy()+np.clip(qs@reg.coef_,-.15,.15),tr.sub_ec.min(),tr.sub_ec.max());records.append(f)
            assert np.all((s==-1)|((s<va.day.to_numpy())&(s%2==va.day.to_numpy()%2)&((s>=179)==(va.day.to_numpy()>=179))))
            np.savez(OUT/f'{prefix}_{seed}_extra.npz',coef=reg.coef_,cal_design=x,cal_target=z['y']-ip,outer_design=q,outer_source=s,outer_id=va.row_id.to_numpy(dtype=str),clip_lo=tr.sub_ec.min(),clip_hi=tr.sub_ec.max())
    oof=pd.concat(records,ignore_index=True);oof.to_csv(OUT/'extra_oof.csv',index=False)
    # Reuse the unchanged scoring routine; the unchanged TGATE rows only satisfy its shape contract.
    scoring=oof.copy();scoring['arm']=scoring.arm.map({'TBIAS_PH':'TBIAS','ESTATE_PAR':'ESTATE'});scoring=pd.concat([scoring,original[original.arm=='TGATE']],ignore_index=True);scores,decisions=evaluate(scoring);names={'TBIAS':'TBIAS_PH','ESTATE':'ESTATE_PAR'};scores=[dict(s,arm=names[s['arm']]) for s in scores if s['arm'] in names];decisions={names[k]:v for k,v in decisions.items() if k in names};pd.DataFrame(scores).to_csv(HERE/'extra_scores.csv',index=False);savej(HERE/'extra_result.json',dict(scores=scores,decisions=decisions,family_k=K,alpha=ALPHA,oof_hash=sha(OUT/'extra_oof.csv'),code_hash=sha(__file__)) );print(json.dumps(decisions),flush=True)
if __name__=='__main__':main()
