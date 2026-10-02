"""실행 중 생성 완료된 캐시만 검사. 후보 성능/잠금 정답을 읽지 않는다."""
import sys,json
sys.dont_write_bytecode=True
import run_dc4 as d
import numpy as np

def main():
    raw,lab,folds,locks,core,base=d.setup();sc=d.seasonal_core(core)
    checked=[];incomplete=[]
    for fold in folds:
        tr,va=d.common.split_fold(raw,lab,fold,locks)
        tr,va,notes=d.transform(raw,tr,va,check=True)
        provenance=d.engine.fold_provenance(tr,va,sc,base)|{'season_checks':notes}
        for kind,seeds,description in [('r3',d.SEEDS,'R3 original .6ET+.3LGB+.1MLP'),
                                      ('pfn',[1,2,3,4],'TabPFN V2 CPU float32 context2000 estimator4')]:
            for seed in seeds:
                path=d.engine.cache_path(fold,kind,seed)
                if not path.exists():continue
                if not path.with_suffix('.json').exists():
                    incomplete.append(str(path));continue
                key=d.engine.canonical_hash(provenance|{'kind':description,'seed':seed})
                arrays,metadata=d.engine.load_cache(path,key,va.row_id)
                assert np.array_equal(arrays['sub_ec'],va.sub_ec.to_numpy(float))
                if kind=='r3':
                    assert arrays['train_row_id'].tolist()==tr.row_id.tolist()
                    assert np.array_equal(arrays['raw_r3'],.6*arrays['raw_et']+.3*arrays['raw_lgb']+.1*arrays['raw_mlp'])
                else:
                    ix=np.random.default_rng(seed).choice(len(tr),size=min(2000,len(tr)),replace=False)
                    assert np.array_equal(arrays['context_index'],ix)
                    assert arrays['context_row_id'].tolist()==tr.row_id.iloc[ix].tolist()
                    assert metadata['repeat_first8']=='PASS'
                checked.append({'file':path.name,'sha256':d.engine.sha(path),'kind':kind,'seed':seed,'fold':list(fold[:2])})
    record={'status':'PASS','r3_caches':sum(c['kind']=='r3' for c in checked),
            'pfn_caches':sum(c['kind']=='pfn' for c in checked),'checked':checked,
            'in_progress_cache_pairs':incomplete,'final_lock_scored':False,'performance_selection':False,
            'validation':'provenance/sha/row order/labels/context index and row IDs/ensemble arithmetic'}
    d.engine.atomic_json(d.HERE/'completed_cache_audit_v1.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='checked'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
