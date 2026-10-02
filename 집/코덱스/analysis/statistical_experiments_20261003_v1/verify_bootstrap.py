from support import *
import csv,time,math
def main():
    while not (HERE/'verification.json').exists():time.sleep(3)
    result=json.loads((HERE/'result.json').read_text(encoding='utf-8'));groups={}
    with (OUT/'oof.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['validator']!='DIAG10':continue
            key=(row['arm'],int(row['seed']),row['context']);block=(row['farm'],int(float(row['day']))//5);target=float(row['y']);difference=(float(row['candidate'])-target)**2-(float(row['baseline'])-target)**2;groups.setdefault(key,{}).setdefault(block,[]).append(difference)
    checked=[]
    for score in result['scores']:
        if score['validator']!='DIAG10':continue
        entries=groups[(score['arm'],score['seed'],score['context'])];keys=sorted(entries);sums=[math.fsum(entries[k]) for k in keys];counts=[len(entries[k]) for k in keys];rng=np.random.default_rng(20261003);indices=[]
        for farm in ['F13','F47']:
            ids=np.array([i for i,k in enumerate(keys) if k[0]==farm]);indices.append(ids[rng.integers(0,len(ids),(20000,len(ids)))])
        ix=np.concatenate(indices,axis=1);samples=np.array([math.fsum(sums[i] for i in row)/sum(counts[i] for i in row) for row in ix]);p=sum(float(s)>=0 for s in samples)/len(samples);ci=np.quantile(samples,[ALPHA,1-ALPHA]);assert abs(p-score['p_worse'])<1e-10;assert np.max(np.abs(ci-score['ci_mse']))<1e-11;checked.append(dict(arm=score['arm'],seed=score['seed'],context=score['context'],p_worse=p,ci_mse=ci.tolist()))
    assert len(checked)==11;savej(HERE/'bootstrap_verification.json',dict(status='PASS',cells=len(checked),method='csv scalar errors + math.fsum blocks/resamples',records=checked));print('BOOTSTRAP_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
