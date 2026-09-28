# -*- coding: utf-8 -*-
import sys,json
sys.dont_write_bytecode=True
from evaluate_task1 import H,common,np,pd,TF
from resid_reset_features import build_features,FEATURE_COLUMNS,USABLE

def bitcheck(a,b):
    assert a.row_id.tolist()==b.row_id.tolist()
    assert a.columns.tolist()==b.columns.tolist()
    for c in ['farm','day','hour','t']:
        assert a[c].equals(b[c])
    av=np.ascontiguousarray(a[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).view(np.uint64)
    bv=np.ascontiguousarray(b[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).view(np.uint64)
    bad=np.any(av!=bv,axis=0)
    return [c for c,x in zip(FEATURE_COLUMNS,bad) if x]

def corrupt(df,mask,seed):
    q=df.copy(deep=True); rng=np.random.default_rng(seed)
    cols=[c for c in q.columns if c not in ['row_id','farm','day','hour','t']]
    n=int(mask.sum())
    q.loc[mask,cols]=q.loc[mask,cols]*10+rng.normal(0,5,(n,len(cols)))
    positions=np.flatnonzero(mask)
    if n:
        blank=positions[rng.random(n)<.33]
        q.loc[q.index[blank],cols]=np.nan
    return q

def main():
    tx,ty,sx=common.load_raw(); base=build_features(tx,sx,ty)
    tests=[]
    for farm in ['F13','F47']:
        for day in [30,220]:
            for hour in [0,5,7,13,23]:
                cut=day*24+hour
                a=corrupt(tx,((tx.farm==farm)&(tx.t>cut)).to_numpy(),cut)
                c=corrupt(sx,((sx.farm==farm)&(sx.t>cut)).to_numpy(),cut+1)
                alt=build_features(a,c,ty)
                m=(base.farm==farm)&(base.t<=cut)
                bad=bitcheck(base.loc[m],alt.loc[m])
                tests.append({'kind':'future','farm':farm,'day':day,'hour':hour,'rows':int(m.sum()),'bad_columns':bad,'pass':not bad})
                print('future',farm,day,hour,'PASS' if not bad else bad,flush=True)
        a=corrupt(tx,(tx.farm!=farm).to_numpy(),726)
        c=corrupt(sx,(sx.farm!=farm).to_numpy(),727)
        alt=build_features(a,c,ty);m=base.farm==farm
        bad=bitcheck(base.loc[m],alt.loc[m]); tests.append({'kind':'other_farm','farm':farm,'rows':int(m.sum()),'bad_columns':bad,'pass':not bad})
    shuffled=ty.copy(); rng=np.random.default_rng(19)
    for col in ['sub_temp','sub_ec']: shuffled[col]=rng.permutation(shuffled[col].to_numpy())
    bad=bitcheck(base,build_features(tx,sx,shuffled));tests.append({'kind':'labels','rows':len(base),'bad_columns':bad,'pass':not bad})
    actual=build_features(common.DATA+'/train_X.csv',common.DATA+'/test_X.csv',common.DATA+'/train_y.csv')
    bad=bitcheck(base,actual);tests.append({'kind':'csv_dataframe_parity','rows':len(base),'bad_columns':bad,'pass':not bad})
    lab=tx[tx.farm.isin(['F13','F47'])].sort_values(['farm','t']).reset_index(drop=True)
    w=TF.row_weights(lab,.2,w_noisy=.2)
    original=common.load_raw
    try:
        changed=corrupt(sx,np.ones(len(sx),bool),123)
        common.load_raw=lambda:(tx.copy(),shuffled.copy(),changed.copy())
        w2=TF.row_weights(lab,.2,w_noisy=.2)
    finally:common.load_raw=original
    same=np.array_equal(w.view(np.uint64),w2.view(np.uint64)); tests.append({'kind':'weights_test_and_label_independence','rows':len(lab),'pass':same})
    result={'feature_count':len(FEATURE_COLUMNS),'feature_rows':len(base),'test_rows':len(sx),'bit_comparison':'float64를 uint64로 재해석; NaN 비트 및 부호 있는 0까지 일치','tests':tests,'all_pass':all(t['pass'] for t in tests)}
    (H/'causality_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert result['all_pass']
    print('ALL PASS',len(tests),flush=True)
if __name__=='__main__':main()
