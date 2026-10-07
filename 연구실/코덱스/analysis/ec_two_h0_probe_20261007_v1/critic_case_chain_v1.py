from pathlib import Path
import csv, json, math, hashlib
H=Path(__file__).resolve().parent
L=H.parents[3]/'연구실/코덱스/local'/H.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((H/'case_chain_v1.json').read_text(encoding='utf-8'))
assert receipt['source_sha']==sha(H/'case_chain_v1.py')
assert receipt['csv_sha']==sha(H/'case_chain_v1.csv')
with (H/'case_chain_v1.csv').open(encoding='utf-8-sig',newline='') as f:summary=list(csv.DictReader(f))
targets={(r['farm'],int(r['day'])) for r in summary}
assert targets=={('F13',231),('F13',233),('F47',130),('F47',161)} and len(summary)==4
selected={}
with (L/'rows.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        key=(r['farm'],int(r['day']))
        if r['seed']=='ensemble' and key in targets:selected.setdefault((r['arm'],*key),[]).append(r)
err=0.; result=[]
def near(a,b):
    global err
    e=abs(float(a)-float(b));err=max(err,e);assert e<1e-12,(a,b)
def means(rows):
    d={int(r['hour']):r for r in rows};assert len(rows)==24 and set(d)==set(range(24))
    raw=[float(d[h]['raw_et']) for h in range(24)]
    smooth=[(raw[h]+math.fsum(raw[:h+1])/(h+1))/2 for h in range(24)]
    for r in rows:near(r['pre_sg2'],r['smooth']);near(r['prediction'],r['sg2_raw'])
    return {'raw':math.fsum(raw)/24,'et_smooth':math.fsum(smooth)/24,'pre':math.fsum(float(r['pre_sg2']) for r in rows)/24,'correction':math.fsum(float(r['sg2_raw'])-float(r['pre_sg2']) for r in rows)/24,'final':math.fsum(float(r['prediction']) for r in rows)/24,'truth':math.fsum(float(r['sub_ec']) for r in rows)/24,'gate':sum(r['gate']=='True' for r in rows)}
for r in summary:
    f,d=r['farm'],int(r['day']);b=means(selected['BASE',f,d]);a=means(selected['TWO_H0_DROP',f,d])
    direct=.48*(a['et_smooth']-b['et_smooth']);sg=a['correction']-b['correction'];change=a['final']-b['final']
    near(direct,a['pre']-b['pre']);near(direct+sg,change);near(a['truth'],b['truth'])
    for key,v in {'truth':b['truth'],'base_final':b['final'],'alt_final':a['final'],'final_change':change,'ET_retraining_smoothed_weighted_change':direct,'SG2_change':sg,'base_raw_et':b['raw'],'alt_raw_et':a['raw']}.items():near(r[key],v)
    assert int(r['base_gate_rows'])==b['gate'] and int(r['alt_gate_rows'])==a['gate']
    result.append({'farm':f,'day':d,'final_change':change,'weighted_ET_change':direct,'SG2_change':sg})
out={'status':'PASS_FOUR_SELECTED_CASE_MEAN_IDENTITIES','selected_hour_rows':sum(map(len,selected.values())),'new_fit':0,'tree_routing':0,'max_absdiff':err,'cases':result,'scope':'Four post-hoc selected cases only; saved ensemble row means, clip inactivity, arithmetic and source/CSV hash. No forest-feature causality or generalization verification.'}
with (H/'critic_case_chain_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
