"""Independent arithmetic verification, no model fit and no validation scoring."""
from pathlib import Path
import sys,csv,json,math,hashlib,shutil
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_submission10_season_20261002_v2/artifact'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    source=OUT/'source';meta=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert len(meta['features']['FULL'])==38 and len(meta['features']['BASE'])==14
    for cols in meta['features'].values():assert 'day' not in cols and cols[-1]=='season'
    # Final-fit training labels are used only to check clip limits, never to score locked days.
    limits=[float(r['sub_ec']) for r in read(OUT/'stage/data/train_y.csv')]
    low,high=min(limits),max(limits)
    with np.load(source/'prediction_details.npz') as z:
        ids=z['row_id'].tolist();r3=z['raw_r3_by_seed'];pfn=z['raw_pfn_by_context'];actual=z['prediction']
        assert r3.shape==(3,1440) and pfn.shape==(4,1440)
        member_gaps=[]
        for i,seed in enumerate([7,101,2024]):
            check=np.array([math.fsum([.6*e,.3*l,.1*m]) for e,l,m in zip(z[f'et_{seed}'],z[f'lgb_{seed}'],z[f'mlp_{seed}'])])
            gap=float(np.max(np.abs(check-r3[i])));assert gap<1e-12;member_gaps.append(gap)
        raw=np.array([.8*math.fsum(r3[:,i])/3+.2*math.fsum(pfn[:,i])/4 for i in range(1440)])
        groups={};pred=np.empty(1440)
        for i,k in enumerate(ids):groups.setdefault(k[:7],[]).append((k,i))
        for g in groups.values():
            history=[]
            for key,i in sorted(g):
                history.append(float(raw[i]));v=.5*raw[i]+.5*math.fsum(history)/len(history)
                pred[i]=max(low,min(high,v))
        gap=float(np.max(np.abs(pred-actual)));assert gap<1e-12
    csvrows=read(source/'submission_10.csv');table=dict(zip(ids,actual))
    assert all(r['sub_temp']=='' and float(r['sub_ec'])==table[r['row_id']] for r in csvrows)
    report={'status':'PASS','rows':1440,'temperature_blank':True,'recipe_independent_method':'scalar math.fsum for 3-member R3, 3-seed and 4-context averages, serial causal prefix average, training min/max clip','member_max_differences':member_gaps,'full_recipe_max_difference':gap,'clip_min':low,'clip_max':high,'consumed_lock_scored':False,'final_training_labels_used_for_clip_limits_only_in_this_audit':True,'source_csv_sha256':hashlib.sha256((source/'submission_10.csv').read_bytes()).hexdigest()}
    target=HERE/'prediction_recipe_audit_v1.json';assert not target.exists();target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(target,OUT/'stage/prediction_recipe_audit_v1.json')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
