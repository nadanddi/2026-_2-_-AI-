"""Synthetic behavior/boundary audits only; no competition query input or target read."""
from pathlib import Path
from copy import deepcopy
import json,hashlib,math
from ch2_prefix_sources_v2 import PrefixSourceCH2,RAW,METHODS,WEATHER
HERE=Path(__file__).resolve().parent

def fixture(cycle=False):
    records={};links=[];train=[];query=[];gap=[];blocks=[];scales={}
    for farm in ['F13','F47']:
        left=[10,11,12];right=[20,21,22];days=list(range(14,19))
        for day in left+right:
            b=day in [11,21];hours={}
            for h in range(24):
                obs={c:0. for c in RAW};obs['out_temp']=100.*b;obs['in_temp']=50.*b
                hours[h]={**obs,'sub_ec':(.4 if not b else 1.2)+h*.001}
                train.append(f'{farm}_{day:03d}_{h:02d}')
            records[farm,day]=hours
        for a,b in [(10,12),(12,20),(20,22),(11,21)]+([(22,10)] if cycle else []):
            links.append({'source':(farm,a),'target':(farm,b)})
        query.extend(f'{farm}_{d:03d}_{h:02d}' for d in days for h in range(24))
        gap.extend(f'{farm}_{d:03d}_{h:02d}' for d in [13,19] for h in range(24))
        blocks.append({'farm':farm,'flank_left':left,'flank_right':right,'query_days':days})
        scales[farm]={c:1. for c in WEATHER}
    return {'train_ids':train,'query_ids':query,'gap_ids_REMOVE_INPUT_AND_BOTH_LABELS':gap,'blocks':blocks},records,links,scales

def observation(b=False):
    values={c:0. for c in RAW};values['out_temp']=100.*b;values['in_temp']=50.*b
    return values

def signature(model):
    value={'records':{str(k):{h:dict(v) for h,v in hours.items()} for k,hours in model.records.items()},
        'roots':{str(k):v for k,v in model.root_by_node.items()},'cyclic':sorted(model.cyclic_roots),
        'scales':{k:dict(v) for k,v in model.scales.items()},'bounds':model.bounds,
        'support':{str(k):v for k,v in model.supported_columns.items()}}
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

class ExplodesOnFloat:
    def __float__(self):raise RuntimeError('forbidden numeric value was parsed')

def main():
    checks=[]
    def checked(name):checks.append(name)
    layout,records,links,scales=fixture();model=PrefixSourceCH2(layout,records,links,scales)
    before=signature(model);rid='F13_014_00'
    for b in [False,True]:
        packet={rid:observation(b)}
        for method in METHODS:
            value,diag=model.predict(rid,.7,packet,method)
            active=not (b and method=='CH2_REFONLY_GUARD')
            assert diag['active']==active
            if active:
                left,right=(11,21) if b else (12,20)
                weight=(14*24-(left*24+23))/(right*24-(left*24+23))
                level=1.2 if b else .4
                expected=.8*.7+.2*((1-weight)*(level+.023)+weight*level)
                assert abs(value-expected)<1e-12
                assert diag['left_day']==left and diag['right_day']==right
            else:assert value==.7
            checked(f'{method}_h0_source{int(b)}_independent_anchor_arithmetic')
    # A source change remains ambiguous in history mode; current-only mode can choose a new source.
    packet={'F13_014_00':observation(False),'F13_015_00':observation(False),'F13_016_00':observation(True)}
    value,diag=model.predict('F13_016_00',.7,packet,'PAST_QUERY_PREFIX_STATE')
    assert value==.7 and not diag['active'] and diag['reason']=='current_history_disagree'
    assert model.predict('F13_016_00',.7,packet,'FLANK_SOURCE_MATCH')[1]['active']
    checked('current_history_source_change_abstains')
    for method in METHODS:
        current={rid:observation(False)}
        current[rid]['out_temp']=50.;current[rid]['in_temp']=25.
        value,diag=model.predict(rid,.7,current,method)
        assert value==.7 and diag['reason']=='signature_tie'
        checked(f'{method}_exact_signature_tie_abstains')
        missing={rid:{c:None for c in RAW}}
        value,diag=model.predict(rid,.7,missing,method)
        assert value==.7 and not diag['active']
        checked(f'{method}_missing_signature_abstains')
        extreme={rid:observation(False)};extreme[rid]['out_temp']=1e308
        value,diag=model.predict(rid,.7,extreme,method)
        assert value==.7 and not diag['active']
        checked(f'{method}_finite_extreme_signature_abstains')
    # Reject temporal/identity/label violations before any forbidden float conversion.
    for other in ['F13_014_01','F13_015_00','F47_014_00','F13_013_00']:
        packet={rid:observation(False),other:{c:ExplodesOnFloat() for c in RAW}}
        rejected=False
        try:model.predict(rid,.7,packet,'FLANK_SOURCE_MATCH')
        except AssertionError:rejected=True
        assert rejected
        checked(f'forbidden_packet_{other}_rejected_before_value_parse')
    for label in ['sub_ec','sub_temp']:
        packet={rid:{**observation(False),label:ExplodesOnFloat()}}
        rejected=False
        try:model.predict(rid,.7,packet,'FLANK_SOURCE_MATCH')
        except AssertionError:rejected=True
        assert rejected
        checked(f'query_{label}_rejected_before_value_parse')
    packet={'F13_014_00':observation(False),'F13_014_23':observation(False),
            'F13_015_00':observation(False),'F13_016_00':observation(False)}
    for method in METHODS:
        direct=model.predict('F13_016_00',.7,packet,method)
        reverse=model.predict('F13_016_00',.7,dict(reversed(list(packet.items()))),method)
        assert direct==reverse==model.predict('F13_016_00',.7,deepcopy(packet),method)
        checked(f'{method}_prefix_order_repeat_replay_invariant')
    assert signature(model)==before
    checked('inference_never_mutates_reference_signature')
    records['F13',10][0]['out_temp']=999.;scales['F13']['out_temp']=999.
    assert signature(model)==before
    checked('constructor_copies_reference_and_scale_inputs')
    for mutation in [lambda:setattr(model,'bounds',(0.,100.)),
                     lambda:model.records['F13',10][0].__setitem__('out_temp',999.),
                     lambda:WEATHER.__setitem__('out_temp',999.)]:
        rejected=False
        try:mutation()
        except (AttributeError,TypeError):rejected=True
        assert rejected
        checked('immutable_reference_or_metric_mutation_rejected')
    l,r,e,s=fixture(cycle=True);cyclic=PrefixSourceCH2(l,r,e,s)
    for method in METHODS:
        value,diag=cyclic.predict(rid,.7,{rid:observation(False)},method)
        assert value==.7 and diag['reason']=='cyclic_component'
        checked(f'{method}_entire_cyclic_component_abstains')
    assert cyclic.predict(rid,.7,{rid:observation(True)},'FLANK_SOURCE_MATCH')[1]['active']
    checked('unrelated_cyclic_component_does_not_relabel_acyclic_source')
    l,r,e,s=fixture();s['F13']['out_temp']=0.;zero=PrefixSourceCH2(l,r,e,s)
    assert zero.predict(rid,.7,{rid:observation(False)},'FLANK_SOURCE_MATCH')[0]==.7
    checked('zero_training_weather_scale_abstains_without_query_refit')
    l,r,e,s=fixture();r['F13',10][0]['out_temp']=None;common=PrefixSourceCH2(l,r,e,s)
    assert 'out_temp' not in common.supported_columns[0,0]
    assert common.predict(rid,.7,{rid:observation(True)},'FLANK_SOURCE_MATCH')[1]['active']
    checked('training_common_available_columns_fixed_before_query')
    out=HERE/'CH2_prefix_synthetic_audit_v1.json';assert not out.exists()
    paths=[Path(__file__),HERE/'ch2_prefix_sources_v2.py']
    result={'status':'SYNTHETIC_FIXED_FLANK_PREFIX_AUDIT_PASS_NOT_REAL_QUERY_VALIDATION',
        'checks':checks,'check_count':len(checks),'sources_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'competition_query_inputs_parsed':0,'competition_labels_parsed':0,'model_fit':False,
        'performance_evaluated':False,'adoption_permitted':False,
        'limits':['Hand-constructed source components; not evidence of real physical source or predictive accuracy',
                  'Real reference artifact, full registered inference/gate and original validators still required',
                  'Fixed flank search lineage must be compared with Claude PF1/PF2 before candidate execution']}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'CH2 synthetic prefix boundary tests PASS: {len(checks)} checks; no competition prediction/score',flush=True)

if __name__=='__main__':main()
