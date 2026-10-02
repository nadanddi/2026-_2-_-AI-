from world import *
import time
def main():
    start=time.time();lab,pfn,ct,phc,wb,wp,wv,sets,z=worlds();out=[]
    print('BASE day hinges:',[c for c in ct if 'day_hinge' in c],flush=True)
    for name,folds in sets:
        for k,fd in enumerate(folds):
            path=OUT/f'members_{name}_{k}.npz';tm,vm=common.split_mask(lab,fd);tr,va=season_fold(lab[tm],lab[vm],wv,name,k)
            if path.exists():continue
            payload={'row_id':va.row_id.to_numpy(dtype=str)}
            for tag in ['base','season']:
                a,b=(tr,va) if tag=='base' else (tr.assign(day=tr.season),va.assign(day=va.season))
                payload['codex_'+tag]=TM.codex_fit_predict(a,b,wb[tm],726)
                for seed in [7,101]:payload[f'mask_{tag}_{seed}']=base_predict(a,b,wb[tm],ct,phc,seed)
            np.savez(path,**payload)
            print(f'{name}/{k} independent all BASE/CODEX fits, {len(va)//24} days, elapsed{time.time()-start:.0f}s',flush=True)
    (OUT/'members_done.json').write_text(json.dumps(dict(elapsed=time.time()-start,code_hash=sha(HERE/'tk3_reproduce.py')),indent=2),encoding='utf-8')
if __name__=='__main__':main()
