"""BLK loader: gaps never parsed as observations; held-out labels never exposed."""
from pathlib import Path
import csv
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'공용/대회자료/정형데이터/참가자_배포'
RAW=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
def key(rid):
    f,d,h=rid.split('_');return f,int(d),int(h)
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

class BLKContext:
    def __init__(self,layout):
        assert layout['status']=='REGISTERED','exact BLK structure infeasible; do not silently relax'
        self.train_ids=set(layout['train_ids']);self.query_ids=set(layout['query_ids'])
        self.gap_ids=set(layout['gap_ids_REMOVE_INPUT_AND_BOTH_LABELS'])
        self.reference_inputs={};self.reference_labels={};self._query={}
        # Filter IDs before numeric conversion. No global pivot/feature/chain sees a gap.
        with (DATA/'train_X.csv').open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                rid=row['row_id']
                if rid not in self.train_ids and rid not in self.query_ids:continue
                obs={c:(float(row[c]) if row[c].strip() else None) for c in RAW}
                if rid in self.train_ids:self.reference_inputs[rid]=obs
                else:self._query[rid]=obs
        with (DATA/'train_y.csv').open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                rid=row['row_id']
                if rid not in self.train_ids:continue
                self.reference_labels[rid]={c:float(row[c]) for c in ['sub_ec','sub_temp']}
        assert set(self.reference_inputs)==set(self.reference_labels)==self.train_ids
        assert set(self._query)==self.query_ids
        assert not(self.train_ids&self.query_ids or self.train_ids&self.gap_ids or self.query_ids&self.gap_ids)
    def query_prefix(self,rid):
        assert rid in self.query_ids
        farm,day,hour=key(rid)
        return {k:dict(v) for k,v in sorted(self._query.items()) if key(k)[0]==farm and (key(k)[1],key(k)[2])<=(day,hour)}
    def packet(self,rid):
        return {'reference_inputs':self.reference_inputs,'reference_labels':self.reference_labels,
                'query_prefix':self.query_prefix(rid)}

if __name__=='__main__':
    path=HERE/'BLK_layout_v2.json';layout=json.loads(path.read_text(encoding='utf-8'))
    if layout['status']!='REGISTERED':
        out=HERE/'BLK_context_audit_v1.json';assert not out.exists()
        out.write_text(json.dumps({'status':'NOT_RUN_STRUCTURE_INFEASIBLE','source':str(path)},ensure_ascii=False),encoding='utf-8')
        raise SystemExit('BLK exact shape infeasible with protected lock; no context/model fit')
    ctx=BLKContext(layout);tests=0
    for rid in sorted(ctx.query_ids):
        prefix=ctx.query_prefix(rid);f,d,h=key(rid)
        assert rid in prefix and not(set(prefix)&ctx.gap_ids)
        assert all(key(k)[0]==f and key(k)[1:]<=(d,h) for k in prefix)
        assert not(set(ctx.reference_labels)&ctx.query_ids)
        tests+=1
    probes=[f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks'] for i in [0,len(b['query_days'])//2,len(b['query_days'])-1] for h in [0,3,6,12,23]]
    for rid in probes:
        original=digest(ctx.query_prefix(rid));f,d,h=key(rid)
        future=[k for k in ctx._query if key(k)[0]!=f or key(k)[1:]>(d,h)]
        saved={k:ctx._query[k] for k in future}
        try:
            for k in future:ctx._query[k]={c:99999.0 for c in RAW}
            assert digest(ctx.query_prefix(rid))==original
        finally:ctx._query.update(saved)
        tests+=1
    anchors=layout['endpoint_anchor_ids']
    assert all(a['left_23h'] in ctx.reference_labels and a['right_0h'] in ctx.reference_labels for a in anchors)
    result={'status':'PASS','query_rows':len(ctx.query_ids),'train_rows':len(ctx.train_ids),
        'gap_rows_absent':len(ctx.gap_ids),'query_checks_and_future_other_farm_perturbations':tests,
        'both_endpoint_anchor_pairs':len(anchors),'hidden_labels_exposed':0,
        'gap_values_parsed':False,'numeric_labels_used_to_choose_layout':False,
        'performance_tested':False,'whole_chain_model_causality_tested':False,
        'limitations':['문맥 API 검사. 이 API 밖에서 CSV를 직접 읽는 사슬/모델 구현은 별도 감사 필요',
            '가짜 평가 전 시각의 앞쪽 query 기록은 입력만 전달하며 정답은 없음']}
    out=HERE/'BLK_context_audit_v1.json';assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
