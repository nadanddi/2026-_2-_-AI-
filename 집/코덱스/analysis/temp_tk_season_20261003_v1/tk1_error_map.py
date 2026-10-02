from world import *
def main():
    lab,pfn,ct,phc,wb,wp,wv,sets,z=worlds();cal=pd.read_csv(Path(env.LOCAL)/'deep_cal_9_days.csv').set_index(['farm','day']).cal
    raw,_,_=TM.ORIG();sealed=raw.groupby(['farm','day']).act_vent.apply(lambda a:float((a==0).mean())>.85)
    lab['cal']=[cal.get((f,d),np.nan) for f,d in zip(lab.farm,lab.day)];lab['sealed']=[sealed.get((f,d),False) for f,d in zip(lab.farm,lab.day)]
    g=np.where(lab.in_temp.isna(),1,np.clip((lab.in_temp.to_numpy()-8)/2,0,1));out=[];detail=[]
    for name in ['DIAG10','EXT10','EXT12']:
        for context,p in [('1-8',np.load(Path(env.LOCAL)/f'web_tabpfn_v2_temp_{name}.npy').mean(0)),('17-24',np.load(Path(env.LOCAL)/f'web_tabpfn_v6_temp_{name}.npy')[:8].mean(0))]:
            for bs,cs in [(7,726),(101,727)]:
                b,c=z[f'{name}__MASK__{bs}'],z[f'{name}__CODEX__{cs}'];preds={'BASE':b,'CODEX':c,'PFN':p,'W30G':w30(b,c,p,g),'W30':.5*b+.2*c+.3*p,'G_C2':(.4+.2*g)*b+(.6-.4*g)*c+.2*g*p}
                good=np.isfinite(b)&np.isfinite(c)&np.isfinite(p)
                for member,pr in preds.items():
                    d=lab.loc[good,['row_id','farm','day','hour','sub_temp','cal','sealed']].copy();d['prediction']=pr[good];d['error']=d.prediction-d.sub_temp;d['validator']=name;d['member']=member;d['base_seed']=bs;d['context']=context;detail.append(d)
                    strata={'all':np.ones(len(d),bool),'early':d.day<179,'late':d.day>=179,'calendar_early':d.cal<70,'calendar_late':d.cal>=70,'late_calendar_early':(d.day>=179)&(d.cal<70),'late_calendar_late':(d.day>=179)&(d.cal>=70),'sealed':d.sealed,'unsealed':~d.sealed}
                    for f in ['F13','F47']:strata[f]=d.farm==f;strata[f+'_late']=(d.farm==f)&(d.day>=179)
                    for seg,mask in strata.items():
                        q=d[mask]
                        if not len(q):continue
                        days=q.groupby(['farm','day']).error.agg(['mean','count']);sse=float((q.error**2).sum());level=float((days['mean']**2*days['count']).sum())
                        out.append(dict(validator=name,context=context,base_seed=bs,member=member,segment=seg,n=len(q),days=len(days),rmse=rmse(q.prediction,q.sub_temp),bias=float(q.error.mean()),level_sse_fraction=level/sse,shape_sse_fraction=1-level/sse))
    pd.DataFrame(out).to_csv(HERE/'TK1_error_map.csv',index=False);pd.concat(detail).to_csv(OUT/'TK1_predictions.csv',index=False)
    print(pd.DataFrame(out).query("validator=='DIAG10' and context=='1-8' and base_seed==7 and segment in ['early','late','late_calendar_early','late_calendar_late','sealed']").to_string(index=False),flush=True)
if __name__=='__main__':main()
