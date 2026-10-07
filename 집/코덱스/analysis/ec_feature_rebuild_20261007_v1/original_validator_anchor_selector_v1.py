"""Structure-only endpoint extension of original folds; never changes held-out IDs."""
from pathlib import Path
import csv,json,hashlib
from blk_context_v1 import key
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'공용/대회자료/정형데이터/참가자_배포'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def select(query_ids,train_ids):
    queries={key(r)[:2] for r in query_ids}
    training={key(r)[:2] for r in train_ids}
    assert not(queries&training)
    blocks=[];anchors=[];unavailable=[]
    for farm in ['F13','F47']:
        ds=sorted(d for f,d in queries if f==farm)
        runs=[]
        for d in ds:
            if not runs or d!=runs[-1][-1]+1:runs.append([d])
            else:runs[-1].append(d)
        reference=sorted(d for f,d in training if f==farm)
        for days in runs:
            before=[d for d in reference if d<days[0]]
            after=[d for d in reference if d>days[-1]]
            if not before or not after:
                unavailable.extend((farm,d) for d in days);continue
            left,right=before[-1],after[0]
            # Same >=3 support requirement, selected only by row/label presence.
            lf=list(range(left-2,left+1));rf=list(range(right,right+3))
            if not all((farm,d) in training for d in lf+rf):
                unavailable.extend((farm,d) for d in days);continue
            block={'farm':farm,'query_days':days,'length':len(days),'flank_left':lf,'flank_right':rf,
                   'left_missing_record_count':days[0]-left-1,'right_missing_record_count':right-days[-1]-1,
                   'original_fold_unchanged':True}
            anchor={'farm':farm,'query_days':days,'left_23h':f'{farm}_{left:03d}_23','right_0h':f'{farm}_{right:03d}_00'}
            assert anchor['left_23h'] in train_ids and anchor['right_0h'] in train_ids
            blocks.append(block);anchors.append(anchor)
    return blocks,anchors,sorted(unavailable)

def main():
    registry=HERE/'fold_registry_v1.json'
    folds=json.loads(registry.read_text(encoding='utf-8'))
    lockpath=ROOT/'집/코덱스/analysis/codex_independent/ec_final_lock/locked_days.json'
    lock={(v['farm'],int(v['day'])) for v in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
    with (DATA/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:
        available={r['row_id'] for r in csv.DictReader(f) if r['sub_ec'].strip() and key(r['row_id'])[0] in ['F13','F47']}
    results=[]
    for fold in folds['folds']:
        query=set(fold['query_ids'])
        vd={key(r)[:2] for r in query}
        forbidden={(f,d+j) for f,d in vd|lock for j in [-1,0,1]}
        train={r for r in available if key(r)[:2] not in forbidden}
        blocks,anchors,unavailable=select(query,train)
        covered={f'{b["farm"]}_{d:03d}_{h:02d}' for b in blocks for d in b['query_days'] for h in range(24)}
        missing={f'{f}_{d:03d}_{h:02d}' for f,d in unavailable for h in range(24)}
        assert covered|missing==query and not(covered&missing)
        assert all(set(range(24))=={key(r)[2] for r in train if key(r)[:2]==fd} for fd in {key(r)[:2] for r in train})
        result={'validator':fold['validator'],'fold':fold['fold'],'ordered_query_ids':sorted(query),'ordered_train_ids':sorted(train),
            'blocks':blocks,'endpoint_anchor_ids':anchors,'baseline_fallback_ids':sorted(missing),
            'covered_query_rows':len(covered),'all_original_query_rows':len(query),
            'input_forbidden_ids':sorted(available-train-query)}
        results.append(result)
    out=HERE/'original_validator_endpoint_registry_v1.json';assert not out.exists()
    registration={'status':'SEALED_STRUCTURE_ONLY_BEFORE_ENDPOINT_SCORES','selector_code_sha256':sha(__file__),
        'fold_registry_sha256':sha(registry),'lock_sha256':sha(lockpath),
        'source_sha256':{n:sha(DATA/n) for n in ['train_X.csv','train_y.csv']},
        'rule':'Original folds and score population unchanged. Nearest train record on each side of contiguous query run, with >=3 contiguous labeled support each side; missing either => exact baseline fallback for all methods.',
        'limits':['gap lengths may exceed1, so these are original validation conditions, not exact BLK','relative record-hour interpolation retains calendar/source uncertainty','TM scored only original111-day subset; anchors selected using whole DIAG10 heldout fold, not only TM score subset','future query inputs and both heldout labels remain forbidden'],
        'folds':results}
    out.write_text(json.dumps(registration,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=HERE/'original_validator_endpoint_audit_v1.json';assert not audit.exists()
    audit.write_text(json.dumps({'status':'STRUCTURE_PASS','folds':len(results),
        'covered_query_rows_sum_folds':sum(r['covered_query_rows'] for r in results),'query_rows_sum_folds':sum(r['all_original_query_rows'] for r in results),
        'train_query_overlap':0,'numeric_labels_used_to_select':False,'performance_tested':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'folds':len(results),'covered_sum':sum(r['covered_query_rows'] for r in results),'query_sum':sum(r['all_original_query_rows'] for r in results),'performance_tested':False}),flush=True)

if __name__=='__main__':main()
