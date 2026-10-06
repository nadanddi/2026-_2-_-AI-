from pathlib import Path
import sys,importlib.util
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import pandas as pd,numpy as np
spec=importlib.util.spec_from_file_location('tested_sg2',Path(sys.argv[1]));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data=[]
for farm in ['F13','F47']:
    for day in [1,2,3,179,180]:
        for hour in range(24):
            r=dict(row_id=f'{farm}_{day:03}_{hour:02}',out_temp=day*.1+hour*.03,out_hum=55+day*.02+hour*.01,out_rad=(hour+1)*day,out_wspd=day*.005+hour*.02,in_temp=12+hour*.1+day*.01,in_hum=70+hour*.2+day*.02,in_co2=400+hour+day*.05)
            r.update(dict(act_vent=50*(hour>=10),act_thermal=100*(hour<8),act_shade=100*(hour<5),act_heating=20*(hour<7),act_co2=0,act_circfan=0,act_fog=0));data.append(r)
raw=pd.DataFrame(data);ref={(f,d) for f in ['F13','F47'] for d in [1,2]}
prepare=lambda x:m.prepare(x,ref) if m.prepare.__code__.co_argcount==2 else m.prepare(x)
base=prepare(raw);changed=raw.copy();sel=changed.row_id.str[4:7].eq('003');changed.loc[sel,m.W]=changed.loc[sel,m.W]*13+97
test=prepare(changed)
pd.testing.assert_frame_equal(base['WV'].loc[sorted(ref)],test['WV'].loc[sorted(ref)])
print('PASS_TRAIN_REF_WEATHER_INVARIANCE',flush=True)
if m.prepare.__code__.co_argcount==2:
    oldspec=importlib.util.spec_from_file_location('original_sg2',ROOT/'집/클로드/submission14_ec_sg2/sg2post.py');old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
    allref={(f,d) for f in ['F13','F47'] for d in [1,2,3,179,180]};a=m.prepare(raw,allref);b=old.prepare(raw)
    pd.testing.assert_frame_equal(a['WV'],b['WV']);assert a['second']==b['second'] and a['days']==b['days']
    for h in range(24):pd.testing.assert_frame_equal(a['SIG'][h],b['SIG'][h])
    ec=pd.Series([.8]*len(ref),index=pd.MultiIndex.from_tuples(sorted(ref)));cal=m.ref_calendar(base,ref)
    q=raw[raw.row_id.str[4:7].isin(['179','180'])][['row_id']].reset_index(drop=True);p=np.full(len(q),.7);original=m.correct(q,p,base,ec,ref,cal)
    for farm in ['F13','F47']:
        allowed=raw.row_id.str[:3].eq(farm)&((raw.row_id.str[4:7].astype(int)<179)|((raw.row_id.str[4:7].astype(int)==179)&(raw.row_id.str[8:10].astype(int)<=5)))
        later=raw.copy();mask=(~allowed)&~raw.row_id.str[4:7].isin(['001','002']);cols=[c for c in raw if c!='row_id'];later.loc[mask,cols]=later.loc[mask,cols]*13+97
        sq=m.prepare(later,ref);pp=p.copy();qa=q.row_id.str[:3].eq(farm)&q.row_id.str[4:7].eq('179')&(q.row_id.str[8:10].astype(int)<=5);pp[~qa]=pp[~qa]+1
        altered=m.correct(q,pp,sq,ec,ref,m.ref_calendar(sq,ref));np.testing.assert_array_equal(original[qa],altered[qa])
    print('PASS_FULLREF_PACKAGE_EQUIVALENCE_AND_PREFIX_CAUSALITY',flush=True)

