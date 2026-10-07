from pathlib import Path
import csv,json,math,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'연구실/코덱스/local'/H.name
TARGETS={('F13',231),('F13',233),('F47',130),('F47',161)};groups={}
with (L/'rows.csv').open(encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        key=(r['farm'],int(r['day']))
        if r['seed']=='ensemble' and key in TARGETS:groups.setdefault((r['arm'],*key),[]).append(r)
def mean(v):return math.fsum(v)/len(v)
stages={}
for key,rows in groups.items():
    rows.sort(key=lambda r:int(r['hour']));assert [int(r['hour']) for r in rows]==list(range(24))
    et=[float(r['raw_et']) for r in rows];smooth=[.5*et[i]+.5*mean(et[:i+1]) for i in range(24)]
    a=dict(raw_et_mean=mean(et),smoothed_et_mean=mean(smooth),pre_sg2_mean=mean([float(r['pre_sg2']) for r in rows]),sg2_correction_mean=mean([float(r['sg2_raw'])-float(r['pre_sg2']) for r in rows]),final_mean=mean([float(r['prediction']) for r in rows]),truth=mean([float(r['sub_ec']) for r in rows]),gate_rows=sum(r['gate']=='True' for r in rows))
    assert all(abs(float(r['prediction'])-float(r['sg2_raw']))<1e-12 and abs(float(r['pre_sg2'])-float(r['smooth']))<1e-12 for r in rows)
    assert abs(a['final_mean']-a['pre_sg2_mean']-a['sg2_correction_mean'])<1e-12;stages[key]=a
out=[]
for farm,day in sorted(TARGETS):
    b=stages['BASE',farm,day];a=stages['TWO_H0_DROP',farm,day];direct=.48*(a['smoothed_et_mean']-b['smoothed_et_mean']);correction=a['sg2_correction_mean']-b['sg2_correction_mean'];change=a['final_mean']-b['final_mean']
    assert abs(direct-(a['pre_sg2_mean']-b['pre_sg2_mean']))<1e-12 and abs(direct+correction-change)<1e-12
    out.append(dict(farm=farm,day=day,truth=b['truth'],base_final=b['final_mean'],alt_final=a['final_mean'],final_change=change,ET_retraining_smoothed_weighted_change=direct,SG2_change=correction,base_gate_rows=b['gate_rows'],alt_gate_rows=a['gate_rows'],base_raw_et=b['raw_et_mean'],alt_raw_et=a['raw_et_mean']))
p=H/'case_chain_v1.csv'
with p.open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
receipt=dict(status='PASS_FOUR_POSTHOC_CASE_STAGE_IDENTITIES',new_fit=0,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),csv_sha=hashlib.sha256(p.read_bytes()).hexdigest(),cases=out,scope='selected after result; ET-retraining difference plus SG2 interaction, not fixed-feature attribution or generalization proof')
with (H/'case_chain_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print('CASE_STAGE_IDENTITIES_PASS',len(out),flush=True)
