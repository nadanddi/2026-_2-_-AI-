from world import *
import math
def exact_rmse(pred,label):return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(pred,label))/len(label))
def main():
    from analyze import boot
    result=json.loads((HERE/'result.json').read_text(encoding='utf-8'));o=pd.read_csv(OUT/'oof.csv');m=pd.read_csv(OUT/'members.csv');maps=pd.read_csv(HERE/'TK1_error_map.csv');md=pd.read_csv(OUT/'TK1_predictions.csv');_,rawlab,_=harness.load();labels=dict(zip(rawlab.row_id,rawlab.sub_temp));checks=[]
    assert all(abs(labels[r]-y)<1e-12 for r,y in zip(o.row_id,o.sub_temp))
    members={(r.row_id,r.validator):r._asdict() for r in m.itertuples(index=False)}
    pfns={(r.row_id,r.validator,r.context):(r.base,r.candidate) for r in o[o.variant=='TK2_PFN'].itertuples(index=False)}
    for r in result['summary']:
        if r['segment']!='all':continue
        d=o[(o.variant==r['variant'])&(o.validator==r['validator'])&(o.seed==r['seed'])&(o.context==r['context'])]
        a,b=exact_rmse(d.base,d.sub_temp),exact_rmse(d.candidate,d.sub_temp);assert abs(a-r['baseline_rmse'])<1e-12 and abs(b-r['candidate_rmse'])<1e-12
        if r['validator']=='DIAG10':assert abs(boot(d,d.base.to_numpy(),d.candidate.to_numpy())['p_worse']-r['p_worse'])<1e-12
        for row in d.itertuples():
            if row.variant!='TK2_PFN':
                mm=members[(row.row_id,row.validator)];pb,pc=pfns[(row.row_id,row.validator,row.context)]
                g=1 if pd.isna(row.in_temp) else max(0,min(1,(row.in_temp-8)/2));ref=(.4+.1*g)*mm[f'mask_base_{row.seed}']+(.6-.4*g)*mm['codex_base']+.3*g*pb
                assert abs(ref-row.base)<1e-10
                cand=(.4+.1*g)*mm[f'mask_base_{row.seed}']+(.6-.4*g)*mm['codex_base']+.3*g*pc if row.variant=='TK2_W30G' else (.4+.1*g)*mm[f'mask_season_{row.seed}']+(.6-.4*g)*mm['codex_season']+.3*g*pb
                assert abs(cand-row.candidate)<1e-10
        checks.append(dict(variant=r['variant'],validator=r['validator'],seed=r['seed'],context=r['context'],rmse_fsum_match=True))
    mapchecks=[]
    for member in ['BASE','CODEX','PFN','W30G','W30','G_C2']:
        for seg in ['early','late','late_calendar_early','late_calendar_late','sealed']:
            d=md[(md.validator=='DIAG10')&(md.member==member)&(md.base_seed==7)&(md.context=='1-8')]
            mask={'early':d.day<179,'late':d.day>=179,'late_calendar_early':(d.day>=179)&(d.cal<70),'late_calendar_late':(d.day>=179)&(d.cal>=70),'sealed':d.sealed}[seg];d=d[mask]
            r=maps[(maps.validator=='DIAG10')&(maps.member==member)&(maps.base_seed==7)&(maps.context=='1-8')&(maps.segment==seg)].iloc[0]
            er=[float(p)-float(y) for p,y in zip(d.prediction,d.sub_temp)];a=math.sqrt(math.fsum(e*e for e in er)/len(er));bias=math.fsum(er)/len(er)
            by={}
            for f,day,e in zip(d.farm,d.day,er):by.setdefault((f,int(day)),[]).append(e)
            level=math.fsum(math.fsum(es)**2/len(es) for es in by.values())/math.fsum(e*e for e in er)
            assert max(abs(a-r.rmse),abs(bias-r.bias),abs(level-r.level_sse_fraction))<1e-12;mapchecks.append(dict(member=member,segment=seg,n=len(d),rmse=a,bias=bias,level_fraction=level))
    assert max(r['maxdiff'] for r in result['reproduction'])<1e-7
    v=dict(status='PASS',scores=checks,error_map_fsum=mapchecks,reproduction_maxdiff=max(r['maxdiff'] for r in result['reproduction']),repeated_cv_not_holdout=True,training_gap_not_measured=True,weight_rank_not_fold_local=True,source_hashes=result['source_hashes'])
    (HERE/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');print('VERIFICATION PASS')
if __name__=='__main__':main()
