"""Input/EC-component coverage diagnostics; no held-out targets or model accuracy."""
import json,math
from collections import Counter,defaultdict
from blk_context_v1 import BLKContext,key,RAW
from blk_endpoint_methods_v1 import finite,INPUT_COVERAGE,INPUT_RELATIVE_MARGIN
from blk_endpoint_methods_v2 import EndpointMethodsV2
from pathlib import Path
HERE=Path(__file__).resolve().parent
layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
ctx=BLKContext(layout);m=EndpointMethodsV2(layout,ctx.reference_inputs,ctx.reference_labels)
size=Counter(m.root(d) for d in m.days)
next_link=dict(m.links);cycles=set()
for d in next_link:
    path=[];current=d
    while current in next_link and current not in path:
        path.append(current);current=next_link[current]
    if current in path:
        cyc=path[path.index(current):]
        cycles.add(tuple(sorted(cyc)))
reference_signature={(day,h):m.signature(obs,h) for day,obs in m.days.items() for h in range(24)}
records=[]
for rid in sorted(ctx.query_ids):
    f,d,h=key(rid);prefix=ctx.query_prefix(rid)
    query={key(k)[2]:v for k,v in prefix.items() if key(k)[1]==d}
    q=m.signature(query,h)
    best={};dims={}
    for day in m.days:
        if day[0]!=f:continue
        x=reference_signature[day,h]
        differences=[((a-b)/m.scale[RAW[i//2]])**2 for i,(a,b) in enumerate(zip(q,x)) if finite(a) and finite(b)]
        if len(differences)<INPUT_COVERAGE*len(q):continue
        dist=math.sqrt(math.fsum(differences)/len(differences));root=m.root(day)
        if dist<best.get(root,float('inf')):best[root]=dist;dims[root]=len(differences)
    ranked=sorted((v,k) for k,v in best.items())
    chosen=None
    if len(ranked)>=2 and ranked[1][0]-ranked[0][0]>=INPUT_RELATIVE_MARGIN*max(ranked[1][0],1e-8):chosen=ranked[0][1]
    assert m.assigned_chain(rid,prefix)==chosen
    _,a=m.blocks[f,d];ld=key(a['left_23h'])[:2];rd=key(a['right_0h'])[:2]
    same=m.root(ld)==m.root(rd)
    active=same and chosen==m.root(ld)
    records.append({'row_id':rid,'hour':h,'components_compared':len(ranked),'chosen_component':list(chosen) if chosen else None,
        'best_distance':ranked[0][0] if ranked else None,'second_distance':ranked[1][0] if len(ranked)>1 else None,
        'best_dimensions':dims[ranked[0][1]] if ranked else 0,'best_component_size':size[ranked[0][1]] if ranked else 0,
        'same_endpoint_component':same,'guard_active':active})
summary={'status':'DIAGNOSTICS_COMPLETE_NO_ACCURACY_CLAIM','train_only_links':len(m.links),'components':len(size),
    'component_size_distribution':dict(sorted(Counter(size.values()).items())),'directed_cycle_count':len(cycles),
    'cycles':[list(c) for c in sorted(cycles)],'guard_active_rows':sum(r['guard_active'] for r in records),
    'query_rows':len(records),'guard_by_hour':{str(h):{'active':sum(r['guard_active'] for r in records if r['hour']==h),'rows':sum(r['hour']==h for r in records)} for h in range(24)},
    'same_endpoint_component_blocks':sum(m.root(key(a['left_23h'])[:2])==m.root(key(a['right_0h'])[:2]) for a in layout['endpoint_anchor_ids']),
    'records':records,'heldout_truth_loaded':False,'performance_evaluated':False,
    'limitations':['EC similarity components may contain cycles and are not physical-time chains','large components have more chances for small nearest distances','relative distance margin does not guarantee absolute similarity','no threshold changed after these diagnostics']}
out=HERE/'BLK_component_diagnostics_v1.json';assert not out.exists()
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['train_only_links','components','directed_cycle_count','guard_active_rows','same_endpoint_component_blocks','performance_evaluated']}),flush=True)
