"""Fixed-flank CH2 source hypotheses using allowed input prefixes only.

This is a proposed inference primitive, not a fitted or adopted competition model.
All search templates are the six registered flanks; no whole-training nearest-neighbor search.
"""
from types import MappingProxyType
from collections import defaultdict
import math

RAW=('out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2',
     'act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog')
WEATHER={'out_temp':1.,'out_hum':1.,'out_wspd':.3,'out_rad':1.}
INDOOR={'in_temp':1.,'in_hum':5.,'in_co2':40.}
CONTROLS=('act_heating','act_thermal','act_circfan','act_vent')
METHODS=('CH2_REFONLY_GUARD','FLANK_SOURCE_MATCH','PAST_QUERY_PREFIX_STATE')
TIE_TOL=1e-12

def key(rid):
    farm,day,hour=rid.split('_');hour=int(hour)
    assert 0<=hour<24
    return farm,int(day),hour

def finite(value):return value is not None and math.isfinite(value)

class PrefixSourceCH2:
    __slots__=('records','root_by_node','cyclic_roots','blocks','query_to_block','query_ids',
               'scales','bounds','supported_columns','_sealed')
    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('Reference hypothesis is immutable')
        object.__setattr__(self,name,value)

    def __init__(self,layout,records,links,weather_scales):
        train=set(layout['train_ids']);query=set(layout['query_ids'])
        gap=set(layout['gap_ids_REMOVE_INPUT_AND_BOTH_LABELS'])
        assert not(train&query or train&gap or query&gap)
        copied={};actual=set();levels=[]
        for node,hours in records.items():
            farm,day=node;assert set(hours)==set(range(24))
            frozen={}
            for hour,obs in hours.items():
                assert set(obs)==set(RAW)|{'sub_ec'}
                values={c:float(v) if v is not None else None for c,v in obs.items()}
                assert all(v is None or finite(v) for v in values.values())
                assert finite(values['sub_ec']);levels.append(values['sub_ec'])
                frozen[hour]=MappingProxyType(values);actual.add(f'{farm}_{day:03d}_{hour:02d}')
            copied[node]=MappingProxyType(frozen)
        assert actual==train
        parent={node:node for node in copied};successor={};incoming=set()
        def root(node):
            while parent[node]!=node:node=parent[node]
            return node
        for link in links:
            a,b=tuple(link['source']),tuple(link['target'])
            assert a in copied and b in copied and a[0]==b[0] and a!=b
            assert a not in successor and b not in incoming
            successor[a]=b;incoming.add(b)
            ra,rb=root(a),root(b)
            if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
        cyclic=set()
        for start in successor:
            path=[];node=start
            while node in successor and node not in path:path.append(node);node=successor[node]
            if node in path:cyclic.add(root(node))
        roots={node:root(node) for node in copied}
        blocks=[];query_map={};support={}
        columns=tuple(WEATHER)+tuple(INDOOR)+CONTROLS
        for index,block in enumerate(layout['blocks']):
            farm=block['farm'];left=tuple(block['flank_left']);right=tuple(block['flank_right'])
            days=tuple(block['query_days'])
            assert len(left)==len(right)==3 and days==tuple(sorted(days)) and len(days)==len(set(days))
            assert max(left)<min(days) and max(days)<min(right)
            assert all((farm,d) in copied for d in left+right)
            for day in days:
                assert (farm,day) not in copied
                for h in range(24):
                    rid=f'{farm}_{day:03d}_{h:02d}';assert rid in query and rid not in query_map
                    query_map[rid]=index
            blocks.append(MappingProxyType({'farm':farm,'left':left,'right':right,'query_days':days}))
            for h in range(24):
                # Common reference availability makes distances comparable across all six templates.
                support[index,h]=tuple(c for c in columns if all(finite(copied[farm,d][h][c]) for d in left+right))
        assert set(query_map)==query
        self.records=MappingProxyType(copied);self.root_by_node=MappingProxyType(roots)
        self.cyclic_roots=frozenset(cyclic);self.blocks=tuple(blocks)
        self.query_to_block=MappingProxyType(query_map);self.query_ids=frozenset(query)
        self.scales=MappingProxyType({f:MappingProxyType({c:float(s[c]) for c in WEATHER}) for f,s in weather_scales.items()})
        self.bounds=(min(levels),max(levels));self.supported_columns=MappingProxyType(support)
        self._sealed=True

    def _packet(self,rid,prefix):
        assert rid in self.query_ids and rid in prefix
        farm,day,hour=key(rid);copied={}
        for other,obs in prefix.items():
            f,d,h=key(other)
            assert other in self.query_ids and f==farm and (d,h)<=(day,hour)
            assert set(obs)==set(RAW), 'query packet must contain raw input only, never EC/temperature labels'
            values={c:float(v) if v is not None else None for c,v in obs.items()}
            assert all(v is None or finite(v) for v in values.values())
            copied[other]=values
        return copied

    def _distance(self,index,template,observations):
        block=self.blocks[index];farm=block['farm'];scales=self.scales[farm]
        if any(not finite(value) or value<=0 for value in scales.values()):return None
        terms=[]
        for h,query in sorted(observations.items()):
            reference=self.records[farm,template][h]
            for c in self.supported_columns[index,h]:
                if not finite(query[c]):continue
                delta=query[c]-reference[c]
                if c in WEATHER:terms.append(WEATHER[c]*(delta/scales[c])**2)
                elif c in INDOOR:terms.append(.25*(delta/INDOOR[c])**2)
                else:terms.append(.25*abs(delta)/50)
        return math.fsum(terms)/len(terms) if terms else None

    def _rank(self,index,rid,packet,history):
        block=self.blocks[index];farm,day,hour=key(rid)
        observed=defaultdict(dict)
        for other,obs in packet.items():
            f,d,h=key(other)
            if d not in block['query_days'] or (not history and d!=day):continue
            observed[d][h]=obs
        by_component=defaultdict(lambda:defaultdict(list))
        for side in ['left','right']:
            for template in block[side]:
                distances=[self._distance(index,template,hours) for hours in observed.values()]
                if not distances or any(v is None for v in distances):continue
                value=math.fsum(distances)/len(distances)
                component=self.root_by_node[farm,template]
                by_component[component][side].append(value)
        scores=[]
        for component,sides in by_component.items():
            means=[math.fsum(values)/len(values) for values in sides.values()]
            scores.append((math.fsum(means)/len(means),component))
        scores.sort()
        if not scores:return None,{'reason':'no_common_observed_signature'}
        if len(scores)>1 and scores[1][0]-scores[0][0]<=TIE_TOL:return None,{'reason':'signature_tie'}
        return scores[0][1],{'distance':scores[0][0],'component_options':len(scores)}

    def choose(self,rid,prefix,method):
        assert method in METHODS
        packet=self._packet(rid,prefix);index=self.query_to_block[rid];block=self.blocks[index]
        farm,day,hour=key(rid)
        component,diag=self._rank(index,rid,packet,False)
        if component is None:return None,diag
        if component in self.cyclic_roots:return None,{'reason':'cyclic_component'}
        if method=='PAST_QUERY_PREFIX_STATE':
            history,hdiag=self._rank(index,rid,packet,True)
            if history!=component:return None,{'reason':'current_history_disagree','current':diag,'history':hdiag}
        if method=='CH2_REFONLY_GUARD':
            left=max(block['left']);right=min(block['right'])
            if self.root_by_node[farm,left]!=component or self.root_by_node[farm,right]!=component:
                return None,{'reason':'fixed_endpoints_not_supported'}
        else:
            lefts=[d for d in block['left'] if self.root_by_node[farm,d]==component]
            rights=[d for d in block['right'] if self.root_by_node[farm,d]==component]
            if not lefts or not rights:return None,{'reason':'no_two_sided_source_support'}
            left=max(lefts);right=min(rights)
        assert left<day<right
        return (farm,left,right),{'reason':'source_hypothesis_supported','component':component,**diag}

    def predict(self,rid,baseline,prefix,method):
        assert finite(baseline) and self.bounds[0]<=baseline<=self.bounds[1]
        anchors,diag=self.choose(rid,prefix,method)
        if anchors is None:return float(baseline),{'active':False,**diag}
        farm,left,right=anchors;_,day,hour=key(rid)
        left_hour=24*left+23;right_hour=24*right;query_hour=24*day+hour
        weight=(query_hour-left_hour)/(right_hour-left_hour)
        assert 0<weight<1
        anchor=(1-weight)*self.records[farm,left][23]['sub_ec']+weight*self.records[farm,right][0]['sub_ec']
        value=.8*baseline+.2*anchor
        clipped=max(self.bounds[0],min(self.bounds[1],value))
        return clipped,{'active':True,'left_day':left,'right_day':right,'relative_record_hour_weight':weight,**diag}
