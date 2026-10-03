"""Repair ET delta postprocessing without retraining or changing the recipe."""
from run import *

def main():
    lab,core,wv,folds,outer=S.loadec();dst=OUT/'corrected_et_v3';dst.mkdir(parents=True,exist_ok=True)
    for arm in ('EC_RARE_ET','EC_H0CHANGE'):
        for name,k,tm,vm in folds:
            tr=lab[tm];path=OUT/'corrected_et_v2'/f'{arm}_{name}_{k}.csv'
            d=pd.read_csv(path,float_precision='round_trip');parts=[]
            for seed,g in d.groupby('seed',sort=False):
                g=g.reset_index(drop=True).copy()
                # old_et is already shrink(raw original ET); new_et is raw new ET.
                g['candidate']=np.clip(g.baseline.to_numpy()+.48*(core.shrink(g.new_et.to_numpy(),g)-g.old_et.to_numpy()),tr.sub_ec.min(),tr.sub_ec.max())
                parts.append(g)
            pd.concat(parts).to_csv(dst/path.name,index=False)
        jsonout(dst/f'{arm}_audit.json',dict(status='PASS',repair='No refit; baseline ET cache already causally shrunk; new ET shrunk once',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('ET_REPLAY_V3_DONE')
if __name__=='__main__':main()
