from pathlib import Path
import ast
here=Path(__file__).resolve().parent
source=(here/'ch2_prefix_sources_v2.py').read_text(encoding='utf-8')
start=source.index('    def _packet(')
end=source.index('    def _distance(',start)
source=source[:start]+'''    def _packet(self,rid,prefix):
        assert rid in self.query_ids and rid in prefix
        farm,day,hour=key(rid);index=self.query_to_block[rid]
        block=self.blocks[index]
        expected={f'{farm}_{d:03d}_{h:02d}' for d in block['query_days'] if d<=day
                  for h in range(24 if d<day else hour+1)}
        # Inspect identities and schema for the entire packet before numeric conversion.
        assert set(prefix)==expected, 'complete same-block past/current prefix required'
        for other,obs in prefix.items():
            f,d,h=key(other)
            assert other in self.query_ids and self.query_to_block[other]==index
            assert f==farm and (d,h)<=(day,hour)
            assert set(obs)==set(RAW), 'query packet must contain raw input only, never EC/temperature labels'
        copied={}
        for other,obs in prefix.items():
            values={c:float(v) if v is not None else None for c,v in obs.items()}
            assert all(v is None or finite(v) for v in values.values())
            copied[other]=values
        return copied

'''+source[end:]
ast.parse(source)
out=here/'ch2_prefix_sources_v3.py';assert not out.exists()
out.write_text(source,encoding='utf-8')
audit=(here/'audit_ch2_prefix_synthetic_v1.py').read_text(encoding='utf-8')
audit=audit.replace('ch2_prefix_sources_v2','ch2_prefix_sources_v3').replace('CH2_prefix_synthetic_audit_v1.json','CH2_prefix_synthetic_audit_v2.json')
audit=audit.replace('    value,diag=model.predict(\'F13_016_00\',.7,packet',"    packet=complete_packet(model,'F13_016_00',packet)\n    value,diag=model.predict('F13_016_00',.7,packet",1)
audit=audit.replace("    for method in METHODS:\n        direct=model.predict('F13_016_00'", "    packet=complete_packet(model,'F13_016_00',packet)\n    for method in METHODS:\n        direct=model.predict('F13_016_00'",1)
marker='def signature(model):'
helper='''def complete_packet(model,rid,partial):
    farm,day,hour=rid.split('_');day=int(day);hour=int(hour)
    block=model.blocks[model.query_to_block[rid]];packet={}
    for d in block['query_days']:
        if d>day:continue
        first=partial.get(f'{farm}_{d:03d}_00',observation(False))
        for h in range(24 if d<day else hour+1):
            other=f'{farm}_{d:03d}_{h:02d}'
            packet[other]=deepcopy(partial.get(other,first))
    return packet

'''
audit=audit.replace(marker,helper+marker)
marker="    out=HERE/'CH2_prefix_synthetic_audit_v2.json';assert not out.exists()"
extra='''    # Complete prefix enforcement is checked for every method, before numeric access.
    target='F13_015_02'
    full=complete_packet(model,target,{target:observation(False)})
    for method in METHODS:
        assert model.predict(target,.7,full,method)[1]['active']
        checked(f'{method}_complete_previous_day_and_current_prefix_accepted')
        for omitted in ['F13_014_12','F13_015_01']:
            broken={k:{c:ExplodesOnFloat() for c in RAW} for k in full if k!=omitted}
            rejected=False
            try:model.predict(target,.7,broken,method)
            except AssertionError:rejected=True
            assert rejected
            checked(f'{method}_missing_{omitted}_rejected_before_numeric_parse')
    # A second block of the same farm is valid layout, but its earlier query is forbidden.
    l,r,e,s=fixture()
    for node,hours in list(r.items()):
        if node[0]!='F13':continue
        new=(node[0],node[1]+30);r[new]=deepcopy(hours)
        l['train_ids'].extend(f'F13_{new[1]:03d}_{h:02d}' for h in range(24))
    for link in list(e):
        if link['source'][0]=='F13':
            e.append({'source':('F13',link['source'][1]+30),'target':('F13',link['target'][1]+30)})
    l['blocks'].append({'farm':'F13','flank_left':[40,41,42],'flank_right':[50,51,52],'query_days':list(range(44,49))})
    l['query_ids'].extend(f'F13_{d:03d}_{h:02d}' for d in range(44,49) for h in range(24))
    l['gap_ids_REMOVE_INPUT_AND_BOTH_LABELS'].extend(f'F13_{d:03d}_{h:02d}' for d in [43,49] for h in range(24))
    multiblock=PrefixSourceCH2(l,r,e,s)
    for method in METHODS:
        packet={'F13_044_00':{c:ExplodesOnFloat() for c in RAW},'F13_014_00':{c:ExplodesOnFloat() for c in RAW}}
        rejected=False
        try:multiblock.predict('F13_044_00',.7,packet,method)
        except AssertionError:rejected=True
        assert rejected
        checked(f'{method}_earlier_other_block_rejected_before_any_numeric_parse')
'''
assert marker in audit
audit=audit.replace(marker,extra+marker)
ast.parse(audit)
out=here/'audit_ch2_prefix_synthetic_v2.py';assert not out.exists()
out.write_text(audit,encoding='utf-8')
print('Created immutable prefix3/audit2; original versions preserved')
