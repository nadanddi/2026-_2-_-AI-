"""Original CH2 cost/assignment on permitted training references only."""
from blk_baseline_data_v1 import *
from scipy.optimize import linear_sum_assignment

WEATHER_WEIGHT={'out_temp':1.,'out_hum':1.,'out_wspd':.3,'out_rad':1.}
INDOOR_SCALE={'in_temp':1.,'in_hum':5.,'in_co2':40.}
CONTROLS=['act_heating','act_thermal','act_circfan','act_vent']
NO_LINK_COST=12.

class ReferenceCH2:
    def __init__(self,ctx):
        # Numeric records are copies of already filtered reference stores.
        self.records={}
        for rid,obs in ctx.reference_inputs.items():
            f,d,h=key(rid)
            self.records.setdefault((f,d),{})[h]={**obs,'sub_ec':ctx.reference_labels[rid]['sub_ec']}
        assert all(set(v)==set(range(24)) for v in self.records.values())
        self.links=[];self.costs={};self.scales={};self.parent={d:d for d in self.records}
        for farm in ['F13','F47']:
            days=sorted(d for f,d in self.records if f==farm);n=len(days)
            columns=list(WEATHER_WEIGHT)+list(INDOOR_SCALE)+CONTROLS+['sub_ec']
            arrays={v:np.array([[self.records[farm,d][h][v] for h in range(24)] for d in days],float) for v in columns}
            def jumps(v):
                a=arrays[v]
                return (a[:,0][None,:]-a[:,23][:,None])-.5*((a[:,23]-a[:,22])[:,None]+(a[:,1]-a[:,0])[None,:])
            cost=(jumps('sub_ec')/.02)**2
            with np.errstate(invalid='ignore',divide='ignore'):
                for v,w in WEATHER_WEIGHT.items():
                    sd=float(np.nanstd(np.diff(arrays[v],axis=1)))
                    self.scales[farm,v]=sd
                    cost+=w*(jumps(v)/sd)**2
                inside=sum((jumps(v)/s)**2 for v,s in INDOOR_SCALE.items())
                inside+=sum(abs(arrays[v][:,0][None,:]-arrays[v][:,23][:,None])/50 for v in CONTROLS)
            cost+=inside/4
            cost=np.where(np.isfinite(cost),cost,1e6);np.fill_diagonal(cost,1e6)
            big=np.full((2*n,2*n),1e6);big[:n,:n]=cost
            big[:n,n:]=np.where(np.eye(n)==1,NO_LINK_COST,1e6)
            big[n:,:n]=np.where(np.eye(n)==1,NO_LINK_COST,1e6);big[n:,n:]=0
            rr,cc=linear_sum_assignment(big)
            for i,j in zip(rr,cc):
                if i<n and j<n:
                    source,target=(farm,days[i]),(farm,days[j])
                    self.links.append({'source':source,'target':target,'cost':float(cost[i,j]),
                        'EC_endpoint_jump':float(arrays['sub_ec'][j,0]-arrays['sub_ec'][i,23])})
                    a,b=self.root(source),self.root(target)
                    if a!=b:self.parent[max(a,b)]=min(a,b)
            cost.setflags(write=False);self.costs[farm]=(days,cost)
        # Safe finite cycle handling; never follow an unbounded successor loop.
        successor={v['source']:v['target'] for v in self.links};cycles=set()
        for start in successor:
            path=[];node=start
            while node in successor and node not in path:
                path.append(node);node=successor[node]
            if node in path:cycles.add(tuple(sorted(path[path.index(node):])))
        self.cycles=sorted(cycles)

    def root(self,node):
        while self.parent[node]!=node:node=self.parent[node]
        return node

    def scalar_cost(self,f,a,b):
        def jump(v):
            p,q=self.records[f,a],self.records[f,b]
            vals=[p[23][v],p[22][v],q[0][v],q[1][v]]
            if any(x is None for x in vals):return float('nan')
            a23,a22,b0,b1=vals
            return (b0-a23)-.5*((a23-a22)+(b1-b0))
        with np.errstate(divide='ignore',invalid='ignore'):
            cost=(jump('sub_ec')/.02)**2+sum(w*(jump(v)/self.scales[f,v])**2 for v,w in WEATHER_WEIGHT.items())
            inside=sum((jump(v)/s)**2 for v,s in INDOOR_SCALE.items())
            controls=[]
            for v in CONTROLS:
                p,q=self.records[f,a][23][v],self.records[f,b][0][v]
                controls.append(abs(q-p)/50 if p is not None and q is not None else float('nan'))
            cost+= (inside+sum(controls))/4
        return float(cost) if np.isfinite(cost) else 1e6

if __name__=='__main__':
    import json
    from collections import Counter
    from checkpoint_v1 import atomic
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    ctx=BLKContext(layout);graph=ReferenceCH2(ctx)
    differences=[]
    for f,(days,matrix) in graph.costs.items():
        rng=np.random.default_rng(2026100703)
        for _ in range(100):
            a,b=rng.choice(len(days),size=2,replace=False)
            difference=abs(float(matrix[a,b])-graph.scalar_cost(f,days[a],days[b]))
            assert difference<=1e-8
            differences.append(difference)
    size=Counter(graph.root(d) for d in graph.records)
    result={'status':'REF_ONLY_CH2_COST_DIAGNOSTIC_PASS','train_records':len(graph.records),'links':graph.links,
        'link_count':len(graph.links),'components':len(size),'component_sizes':dict(Counter(size.values())),
        'cycles':graph.cycles,'independent_scalar_cost_checks':len(differences),'max_cost_difference':max(differences),
        'same_component_endpoint_blocks':sum(graph.root(key(a['left_23h'])[:2])==graph.root(key(a['right_0h'])[:2]) for a in layout['endpoint_anchor_ids']),
        'no_link_cost':12.,'dummy_semantics':'outgoing and incoming missing each cost12; a link competes with combined24, not strict cost<12',
        'old_source_sha256':sha(ROOT/'집/클로드/research/ch2_label_chains_training_v1.py'),
        'new_code_sha256':sha(__file__),'layout_sha256':sha(HERE/'BLK_layout_v2.json'),
        'query_inputs_used':False,'heldout_truth_loaded':False,'performance_evaluated':False,
        'limits':['Cost agreement is training processing, not proof of physical source or heldout prediction value','EC continuity statistics are partly true by construction','No additional predictive candidate registered or tuned from these diagnostics yet']}
    out=HERE/'CH2_reference_diagnostics_v1.json';assert not out.exists()
    atomic(out,json.dumps(result,ensure_ascii=False,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['status','train_records','link_count','components','same_component_endpoint_blocks','max_cost_difference']}),flush=True)
