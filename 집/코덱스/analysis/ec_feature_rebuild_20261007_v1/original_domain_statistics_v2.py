"""Prospective original-domain diagnostic design; no training or target access."""
from pathlib import Path
from collections import Counter
import hashlib,json,random,struct,math
HERE=Path(__file__).resolve().parent
DRAW_COUNT=200000
SEED=2026100703
ALPHA=.025/84
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def design(folds,tm_days):
    ids=[rid for f in folds if f['validator']=='DIAG10' for rid in f['ordered_query_ids']]
    assert len(ids)==len(set(ids))==8640
    tm={tuple(p) for p in tm_days};assert len(tm)==111
    members={}
    for rid in ids:
        farm,day,hour=rid.split('_');day=int(day);hour=int(hour)
        assert farm in ['F13','F47'] and 0<=hour<24
        members[rid]=(farm,day//5)
    keys=sorted(set(members.values()))
    index={k:i for i,k in enumerate(keys)}
    n=[0]*len(keys);nt=[0]*len(keys)
    tm_ids=[]
    for rid in ids:
        i=index[members[rid]];n[i]+=1
        farm,day,_=rid.split('_')
        if (farm,int(day)) in tm:
            nt[i]+=1;tm_ids.append(rid)
    assert sum(n)==8640 and sum(nt)==len(tm_ids)==2664
    groups=[[i for i,k in enumerate(keys) if k[0]==farm] for farm in ['F13','F47']]
    assert all(groups) and all(n) and max(map(len,groups))<65536
    return {'block_keys':[list(k) for k in keys],'groups':groups,'rows_per_block':n,
            'TM_rows_per_block':nt,'ordered_DIAG_ids':ids,'ordered_TM_ids':tm_ids,
            'membership':[index[members[rid]] for rid in ids]}

def draw_counts(layout):
    rng=random.Random(SEED);accepted=0;rejected=0
    while accepted<DRAW_COUNT:
        counts=[0]*len(layout['block_keys'])
        for group in layout['groups']:
            for _ in group:counts[group[rng.randrange(len(group))]]+=1
        if not sum(c*n for c,n in zip(counts,layout['TM_rows_per_block'])):
            rejected+=1;continue
        accepted+=1
        yield counts,rejected

def bootstrap_loss(draw_path,layout,row_delta,subset):
    """Seedmean squared-error differences provided only AFTER the full model gate."""
    ids=layout['ordered_DIAG_ids'];assert set(row_delta)==set(ids)
    assert subset in ['DIAG10','TM111']
    assert all(math.isfinite(float(x)) for x in row_delta.values())
    allowed=set(ids if subset=='DIAG10' else layout['ordered_TM_ids'])
    sums=[[] for _ in layout['block_keys']]
    for rid,b in zip(ids,layout['membership']):
        if rid in allowed:sums[b].append(float(row_delta[rid]))
    sums=[math.fsum(x) for x in sums]
    lengths=layout['rows_per_block'] if subset=='DIAG10' else layout['TM_rows_per_block']
    width=2*len(sums);fmt='<'+str(len(sums))+'H';values=[]
    with Path(draw_path).open('rb') as h:
        for _ in range(DRAW_COUNT):
            data=h.read(width);assert len(data)==width
            counts=struct.unpack(fmt,data)
            denom=sum(c*n for c,n in zip(counts,lengths));assert denom>0
            values.append(math.fsum(c*s for c,s in zip(counts,sums))/denom)
        assert h.read(1)==b''
    p=(1+sum(v>=0 for v in values))/(DRAW_COUNT+1)
    ordered=sorted(values)
    def quantile(q):
        point=q*(len(ordered)-1);lo=int(point);fraction=point-lo
        return ordered[lo]+fraction*(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])
    return {'p_worse':p,'individual95_MSE_delta_CI':[quantile(.025),quantile(.975)],
            'alpha':ALPHA,'draws':DRAW_COUNT,'block_loss_sums':sums}

def main():
    regpath=HERE/'original_validator_endpoint_registry_v1.json'
    original=HERE/'fold_registry_v1.json'
    plan=HERE/'DOMAIN24_original_statistics_plan_v1.json'
    layout=design(json.loads(regpath.read_text(encoding='utf-8'))['folds'],
                  json.loads(original.read_text(encoding='utf-8'))['TM_days'])
    out=HERE/'DOMAIN24_original_bootstrap_draws_v2.bin'
    fmt='<'+str(len(layout['block_keys']))+'H';rejected=0
    with out.open('xb') as h:
        for counts,rejected in draw_counts(layout):h.write(struct.pack(fmt,*counts))
    assert out.stat().st_size==DRAW_COUNT*2*len(layout['block_keys'])
    receipt={'status':'ORIGINAL_DOMAIN_DIAGNOSTIC_DRAWS_SEALED_BEFORE_FIT_NO_LABELS',
        'source_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),regpath,original,plan,HERE/'preregistration_v2.json']},
        'draw_file_sha256':sha(out),'draw_file_bytes':out.stat().st_size,'draws':DRAW_COUNT,
        'rejected_zero_TM_denominator':rejected,'random_seed':SEED,'layout':layout,
        'layout_sha256':digest(layout),'alpha':ALPHA,'comparisons':84,
        'model_fit':False,'target_values_read':False,'adoption_permitted':False,
        'limits':['Record-day blocks are nominal, not physical chronology','DIAG and TM are overlapping intersection tests',
                  'Prospective to original fit but after BLK exposure; not fresh confirmation']}
    with (HERE/'DOMAIN24_original_statistics_registration_v2.json').open('x',encoding='utf-8') as h:
        json.dump(receipt,h,ensure_ascii=False,indent=2)
    print(f'Original diagnostic draws sealed: {DRAW_COUNT}, {len(layout["block_keys"])} blocks; no fit/labels')

if __name__=='__main__':main()

