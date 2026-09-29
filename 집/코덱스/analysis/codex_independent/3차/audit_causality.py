# -*- coding: utf-8 -*-
import sys,json
sys.dont_write_bytecode=True
from run_experiments import H,np,pd,common
from ec_features import build_features,FEATURE_COLUMNS

def same(a,b):
    assert a.row_id.tolist()==b.row_id.tolist()
    x=np.ascontiguousarray(a[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).view(np.uint64)
    y=np.ascontiguousarray(b[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).view(np.uint64)
    return [c for c,z in zip(FEATURE_COLUMNS,(x!=y).any(axis=0)) if z]

def corrupt(df,mask,seed):
    q=df.copy(deep=True);cols=[c for c in q if c not in ['row_id','farm','day','hour','t']];rng=np.random.default_rng(seed)
    q.loc[mask,cols]=q.loc[mask,cols]*10+rng.normal(0,5,(int(mask.sum()),len(cols)))
    ix=np.flatnonzero(mask); ix=ix[rng.random(len(ix))<.33];q.loc[q.index[ix],cols]=np.nan
    return q

def main():
    tx,ty,sx=common.load_raw();b=build_features(tx,sx,ty);tests=[]
    for farm in ['F13','F47']:
        for day in [30,220]:
            for hour in [0,5,7,13,23]:
                t=day*24+hour
                a=corrupt(tx,((tx.farm==farm)&(tx.t>t)).to_numpy(),t)
                c=corrupt(sx,((sx.farm==farm)&(sx.t>t)).to_numpy(),t+1)
                z=build_features(a,c,ty);m=(b.farm==farm)&(b.t<=t)
                bad=same(b[m],z[m]);tests.append({'kind':'future','farm':farm,'day':day,'hour':hour,'n':int(m.sum()),'bad':bad,'pass':not bad})
                print(farm,day,hour,'PASS' if not bad else bad,flush=True)
        a=corrupt(tx,(tx.farm!=farm).to_numpy(),72);c=corrupt(sx,(sx.farm!=farm).to_numpy(),73)
        z=build_features(a,c,ty);m=b.farm==farm;bad=same(b[m],z[m]);tests.append({'kind':'other_farm','farm':farm,'bad':bad,'pass':not bad})
    yy=ty.copy();rng=np.random.default_rng(72)
    for col in ['sub_temp','sub_ec']:yy[col]=rng.permutation(yy[col].to_numpy())
    bad=same(b,build_features(tx,sx,yy));tests.append({'kind':'label_shuffle','bad':bad,'pass':not bad})
    bad=same(b,build_features(common.DATA+'/train_X.csv',common.DATA+'/test_X.csv'));tests.append({'kind':'csv_interface','bad':bad,'pass':not bad})
    res={'tests':tests,'feature_count':len(FEATURE_COLUMNS),'all_pass':all(t['pass'] for t in tests),'rows':len(b),'test_rows':len(sx)}
    (H/'causality_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    assert res['all_pass'];print('ALL PASS',len(tests),flush=True)
if __name__=='__main__':main()
