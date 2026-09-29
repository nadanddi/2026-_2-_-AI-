# -*- coding: utf-8 -*-
import sys,json
sys.dont_write_bytecode=True
from run_experiments import H,np,pd
from ec_models import NAMES

def main():
    d=pd.read_json(H/'validation_rows.json').set_index('row_id');meta=json.loads((H/'fold_files.json').read_text(encoding='utf-8'))
    out={'prefix_gate':{},'residuals':{}}
    for h in [0,3,12,23]:
        z=d[d.hour==h];pred=z.closed_prefix==1;actual=z.closed_day
        out['prefix_gate'][str(h)]={'days':len(z),'true_closed':int(actual.sum()),'predicted_closed':int(pred.sum()),'precision':float((pred&actual).sum()/pred.sum()) if pred.sum() else None,'recall':float((pred&actual).sum()/actual.sum())}
    for kind in ['A','DIAG10']:
        pieces=[]
        for f in meta['folds']:
            if f['kind']!=kind:continue
            pack=np.load(H/f['file'],allow_pickle=True);z=d.loc[pack['row_id']].copy();z['fold']=f['fold']
            for k in ['baseline']+NAMES:z[k]=pack[k]
            pieces.append(z)
        allz=pd.concat(pieces,ignore_index=True)
        out['residuals'][kind]={}
        for name in NAMES:
            z=allz.copy();z['base_error']=z.baseline-z.sub_ec;z['new_error']=z[name]-z.sub_ec;z['mix_error']=.8*z.base_error+.2*z.new_error
            entry={}
            for group,m in [('all',np.ones(len(z),bool)),('high',z.sub_ec>1.5),('closed',z.closed_day)]:
                zz=z[m]
                # A의 반복 출현 행을 유지한 진단이며 주 평가 평균과 다르다.
                entry[group]={'occurrences':len(zz),'base_bias':float(zz.base_error.mean()),'candidate_bias':float(zz.new_error.mean()),'mix_bias':float(zz.mix_error.mean()),'error_correlation':float(zz.base_error.corr(zz.new_error)),'directional_cross_term':float(np.mean(zz.base_error*(zz.new_error-zz.base_error))),'candidate_difference_mse':float(np.mean((zz.new_error-zz.base_error)**2))}
            out['residuals'][kind][name]=entry
    (H/'diagnostics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
