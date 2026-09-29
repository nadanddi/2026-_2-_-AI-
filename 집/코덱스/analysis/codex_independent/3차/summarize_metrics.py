# -*- coding: utf-8 -*-
import sys,json
sys.dont_write_bytecode=True
from run_experiments import H,np,pd
from ec_models import NAMES

def metric(recs,name,group,fold_mean=True):
    parts=[]; detail=[]
    for rec in recs:
        z=rec['frame'];y=z.sub_ec.to_numpy(); b=rec['baseline'];p=rec[name]
        m={'all':np.ones(len(z),bool),'high':y>1.5,'normal':y<=1.5,'closed':z.closed_day.to_numpy()}[group]
        if not m.sum():continue
        zz=z.loc[m,['farm','day']].copy();zz['sb']=(b[m]-y[m])**2;zz['sp']=(.8*b[m]+.2*p[m]-y[m])**2
        zz['n']=1;zz=zz.groupby(['farm','day'])[['sb','sp','n']].sum();parts.append(zz)
        detail.append({'fold':rec['fold'],'n':int(m.sum()),'days':len(zz),'base':float(np.sqrt(np.mean((b[m]-y[m])**2))),'standalone':float(np.sqrt(np.mean((p[m]-y[m])**2))),'blend':float(np.sqrt(np.mean((.8*b[m]+.2*p[m]-y[m])**2))),'corr':float(pd.Series(b[m]-y[m]).corr(pd.Series(p[m]-y[m])))})
    if not parts:return {'n':0,'folds':[]}
    union=parts[0].index
    for p in parts[1:]:union=union.union(p.index)
    sb=np.stack([p.sb.reindex(union,fill_value=0) for p in parts]);sp=np.stack([p.sp.reindex(union,fill_value=0) for p in parts]);n=np.stack([p.n.reindex(union,fill_value=0) for p in parts])
    rng=np.random.default_rng(726);counts=rng.multinomial(len(union),np.full(len(union),1/len(union)),size=2000)
    bn=counts@n.T; bs=counts@sb.T;ps=counts@sp.T
    if fold_mean:
        good=(bn>0).all(1)
        dif=(np.sqrt(ps[good]/bn[good])-np.sqrt(bs[good]/bn[good])).mean(1)
        base=float(np.sqrt(sb.sum(1)/n.sum(1)).mean());blend=float(np.sqrt(sp.sum(1)/n.sum(1)).mean())
    else:
        nn=bn.sum(1);good=nn>0
        dif=np.sqrt(ps.sum(1)[good]/nn[good])-np.sqrt(bs.sum(1)[good]/nn[good])
        base=float(np.sqrt(sb.sum()/n.sum()));blend=float(np.sqrt(sp.sum()/n.sum()))
    return {'n_occurrences':int(n.sum()),'unique_days':len(union),'base':base,'blend':blend,'delta':blend-base,'ci95':np.quantile(dif,[.025,.975]).tolist(),'bootstrap_valid':len(dif),'folds_improved':sum(f['blend']<f['base'] for f in detail),'folds_total':len(detail),'per_fold':detail,'statistic':'mean_fold_rmse' if fold_mean else 'pooled_nonoverlapping_rmse','standalone_fold_mean':float(np.mean([f['standalone'] for f in detail]))}

def main():
    d=pd.read_json(H/'validation_rows.json').set_index('row_id');meta=json.loads((H/'fold_files.json').read_text(encoding='utf-8'))
    records={'A':[],'DIAG10':[]}
    for f in meta['folds']:
        z=np.load(H/f['file'],allow_pickle=True)
        rec={'fold':f['fold'],'frame':d.loc[z['row_id']].reset_index(),**{k:z[k] for k in ['baseline']+NAMES}}
        records[f['kind']].append(rec)
    result={'candidates':{},'per_fold_ci':{},'bootstrap_design':'同一温室-日'.replace('同一温室-日','동일 온실-날')+'에 A의 모든 폴드에서 동일 복원추출 횟수 적용; 폴드별 RMSE의 산술평균 차이를 계산','counts':{'label_rows':len(d),'days':len(d.groupby(['farm','day'])),'closed_days':len(d[d.closed_day].groupby(['farm','day'])),'high_rows':int((d.sub_ec>1.5).sum())}}
    for kind,recs in records.items():
        result['candidates'][kind]={};result['per_fold_ci'][kind]={}
        for name in NAMES:
            result['candidates'][kind][name]={g:metric(recs,name,g,kind=='A') for g in ['all','high','closed','normal']}
            result['per_fold_ci'][kind][name]={str(r['fold']):{g:metric([r],name,g) for g in ['all','high','closed','normal']} for r in recs}
            print(kind,name,{g:{k:v for k,v in m.items() if k not in ['per_fold']} for g,m in result['candidates'][kind][name].items()},flush=True)
    (H/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',flush=True)
if __name__=='__main__':main()
