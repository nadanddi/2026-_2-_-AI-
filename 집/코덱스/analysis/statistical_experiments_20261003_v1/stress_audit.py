"""Report the preregistered correction-only observation delay stress."""
from support import *
import math

def main():
    frames=[pd.read_csv(OUT/'oof.csv'),pd.read_csv(OUT/'extra_oof.csv')]
    d=pd.concat(frames,ignore_index=True)
    d=d[d.target=='EC']
    records=[]
    for (arm,val,seed),q in d.groupby(['arm','validator','seed'],sort=True):
        values={}
        for field in ['baseline','candidate','stress_delay5']:
            y=q.y.to_numpy(); p=q[field].to_numpy()
            scalar=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(p,y))/len(q))
            assert abs(scalar-np.sqrt(np.mean((p-y)**2)))<1e-12
            values[field]=scalar
        records.append(dict(arm=arm,validator=val,seed=int(seed),n=len(q),**values,
                            candidate_delta_pct=100*(values['candidate']/values['baseline']-1),
                            delay5_delta_pct=100*(values['stress_delay5']/values['baseline']-1)))
    assert len(records)==30
    pd.DataFrame(records).to_csv(HERE/'stress_audit.csv',index=False)
    savej(HERE/'stress_audit.json',dict(status='PASS',cells=len(records),method='numpy mean versus independent math.fsum',
                                     scope='correction only; experts frozen; no additional adoption validator',records=records))
    print(json.dumps(records),flush=True)

if __name__=='__main__':
    main()
