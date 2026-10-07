"""Independent stdlib CH2 training-only cost/assignment audit; no query input or score."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib,math,random,itertools,ast
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'공용/대회자료/정형데이터/참가자_배포'
WEATHER={'out_temp':1.,'out_hum':1.,'out_wspd':.3,'out_rad':1.}
INDOOR={'in_temp':1.,'in_hum':5.,'in_co2':40.}
CONTROLS=['act_heating','act_thermal','act_circfan','act_vent']
RAW=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2',
     'act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
NO_LINK=12.;FORBIDDEN=1e6;TOL=1e-6
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def key(rid):
    farm,day,hour=rid.split('_');return farm,int(day),int(hour)

def hungarian(cost):
    """Independent augmenting-path Hungarian with a dual feasibility certificate."""
    n=len(cost);assert n and all(len(row)==n for row in cost)
    u=[0.]*(n+1);v=[0.]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
    for i in range(1,n+1):
        p[0]=i;j0=0;minimum=[math.inf]*(n+1);used=[False]*(n+1)
        while True:
            used[j0]=True;i0=p[j0];delta=math.inf;j1=0
            for j in range(1,n+1):
                if used[j]:continue
                reduced=cost[i0-1][j-1]-u[i0]-v[j]
                if reduced<minimum[j]:minimum[j]=reduced;way[j]=j0
                if minimum[j]<delta:delta=minimum[j];j1=j
            assert math.isfinite(delta)
            for j in range(n+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    assignment=[None]*n
    for j in range(1,n+1):assignment[p[j]-1]=j-1
    assert sorted(assignment)==list(range(n))
    value=math.fsum(cost[i][j] for i,j in enumerate(assignment))
    dual=math.fsum(u[1:])+math.fsum(v[1:])
    violation=max(0.,max(u[i+1]+v[j+1]-cost[i][j] for i in range(n) for j in range(n)))
    tightness=max(abs(cost[i][j]-u[i+1]-v[j+1]) for i,j in enumerate(assignment))
    assert violation<TOL and tightness<TOL and abs(value-dual)<TOL
    return assignment,value,{'dual_violation':violation,'assigned_edge_tightness':tightness,'primal_dual_gap':abs(value-dual)}

def dummy_matrix(real):
    n=len(real);large=[[FORBIDDEN]*(2*n) for _ in range(2*n)]
    for i in range(n):
        large[i][:n]=real[i]
        large[i][n+i]=NO_LINK;large[n+i][i]=NO_LINK
        for j in range(n):large[n+i][n+j]=0.
    return large

def synthetic_checks():
    rng=random.Random(2026100704);checks=[]
    for n in range(2,7):
        matrix=[[float(rng.randrange(-3,30)) for _ in range(n)] for _ in range(n)]
        _,value,_=hungarian(matrix)
        exhaustive=min(math.fsum(matrix[i][j] for i,j in enumerate(order)) for order in itertools.permutations(range(n)))
        assert value==exhaustive
        checks.append({'kind':'exhaustive_assignment','size':n,'permutations':math.factorial(n)})
    for edge in [11.,13.,20.,25.]:
        real=[[FORBIDDEN,edge],[FORBIDDEN,FORBIDDEN]]
        assignment,value,_=hungarian(dummy_matrix(real))
        selected=sum(i<2 and j<2 for i,j in enumerate(assignment))
        assert selected==int(edge<24.) and value==min(48.,edge+24.)
        checks.append({'kind':'combined_dummy24','edge_cost':edge,'linked':bool(selected)})
    return checks

def main():
    toy=synthetic_checks()
    layout=read('BLK_layout_v2.json');prior=read('CH2_reference_diagnostics_v1.json')
    source=HERE/'ch2_reference_graph_v1.py'
    assert prior['new_code_sha256']==sha(source)
    assert prior['old_source_sha256']==sha(ROOT/'집/클로드/research/ch2_label_chains_training_v1.py')
    assert prior['layout_sha256']==sha(HERE/'BLK_layout_v2.json')
    assert prior['query_inputs_used'] is False and prior['heldout_truth_loaded'] is False
    tree=ast.parse(source.read_text(encoding='utf-8'))
    constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
               if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
               and node.targets[0].id in ['WEATHER_WEIGHT','INDOOR_SCALE','CONTROLS','NO_LINK_COST']}
    assert constants=={'WEATHER_WEIGHT':WEATHER,'INDOOR_SCALE':INDOOR,'CONTROLS':CONTROLS,'NO_LINK_COST':NO_LINK}
    source_paths=[source,HERE/'BLK_layout_v2.json',HERE/'CH2_reference_diagnostics_v1.json',
        ROOT/'집/클로드/research/ch2_label_chains_training_v1.py',DATA/'train_X.csv',DATA/'train_y.csv',Path(__file__)]
    hashes={str(path.resolve()):sha(path) for path in source_paths}
    assert all(sha(DATA/name)==value for name,value in layout['source_sha256'].items())
    wanted=set(layout['train_ids']);query=set(layout['query_ids']);gap=set(layout['gap_ids_REMOVE_INPUT_AND_BOTH_LABELS'])
    assert not(wanted&query or wanted&gap or query&gap)
    inputs={};labels={}
    with (DATA/'train_X.csv').open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            rid=row['row_id']
            if rid not in wanted:continue
            assert rid not in inputs
            values={c:float(row[c]) if row[c].strip() else None for c in RAW}
            assert all(v is None or math.isfinite(v) for v in values.values())
            inputs[rid]=values
    with (DATA/'train_y.csv').open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            rid=row['row_id']
            if rid not in wanted:continue
            assert rid not in labels
            labels[rid]=float(row['sub_ec']);assert math.isfinite(labels[rid])
    assert set(inputs)==set(labels)==wanted
    records=defaultdict(dict)
    for rid,obs in inputs.items():
        f,d,h=key(rid);records[f,d][h]={**obs,'sub_ec':labels[rid]}
    assert all(set(hours)==set(range(24)) for hours in records.values())
    assert len(records)==prior['train_records'] and len(records)*24==len(wanted)
    links=prior['links'];successor={};incoming=set();parent={node:node for node in records}
    def root(node):
        while parent[node]!=node:node=parent[node]
        return node
    for link in links:
        a,b=tuple(link['source']),tuple(link['target'])
        assert a in records and b in records and a!=b and a[0]==b[0]
        assert a not in successor and b not in incoming
        successor[a]=b;incoming.add(b)
        ra,rb=root(a),root(b)
        if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
    cycles=set()
    for start in successor:
        path=[];node=start
        while node in successor and node not in path:path.append(node);node=successor[node]
        if node in path:cycles.add(tuple(sorted(path[path.index(node):])))
    assert sorted(cycles)==sorted(tuple(sorted(tuple(node) for node in cycle)) for cycle in prior['cycles'])
    sizes=Counter(root(node) for node in records)
    assert len(sizes)==prior['components']
    assert {str(k):v for k,v in Counter(sizes.values()).items()}==prior['component_sizes']
    farms=[];differences=[]
    for f in ['F13','F47']:
        days=sorted(d for farm,d in records if farm==f);n=len(days)
        scales={}
        for c in WEATHER:
            increments=[records[f,d][h][c]-records[f,d][h-1][c] for d in days for h in range(1,24)
                        if records[f,d][h][c] is not None and records[f,d][h-1][c] is not None]
            mean=math.fsum(increments)/len(increments) if increments else math.nan
            scales[c]=math.sqrt(math.fsum((x-mean)**2 for x in increments)/len(increments)) if increments else math.nan
        def cost(a,b):
            if a==b:return FORBIDDEN
            left,right=records[f,a],records[f,b]
            def jump(c):
                values=[left[23][c],left[22][c],right[0][c],right[1][c]]
                if any(v is None for v in values):return math.nan
                a23,a22,b0,b1=values
                return (b0-a23)-.5*((a23-a22)+(b1-b0))
            if any(not math.isfinite(s) or s==0 for s in scales.values()):return FORBIDDEN
            terms=[(jump('sub_ec')/.02)**2]+[w*(jump(c)/scales[c])**2 for c,w in WEATHER.items()]
            inside=[(jump(c)/scale)**2 for c,scale in INDOOR.items()]
            inside += [abs(right[0][c]-left[23][c])/50 if right[0][c] is not None and left[23][c] is not None else math.nan for c in CONTROLS]
            total=math.fsum(terms)+math.fsum(inside)/4
            return total if math.isfinite(total) else FORBIDDEN
        real=[[cost(a,b) for b in days] for a in days]
        _,optimal,certificate=hungarian(dummy_matrix(real))
        current=[link for link in links if link['source'][0]==f]
        recomputed=[]
        for link in current:
            value=cost(link['source'][1],link['target'][1]);recomputed.append(value)
            difference=abs(value-link['cost']);assert difference<TOL;differences.append(difference)
        old_assignment_cost=math.fsum(recomputed)+NO_LINK*(2*n-2*len(current))
        assert abs(old_assignment_cost-optimal)<TOL
        farms.append({'farm':f,'train_records':n,'original_link_count':len(current),
            'independent_optimal_assignment_cost':optimal,'original_assignment_recomputed_cost':old_assignment_cost,
            'original_primal_optimal_gap':abs(old_assignment_cost-optimal),'certificate':certificate,
            'weather_scales_train_only':scales})
        print(f'{f} independent CH2 costs and optimality certificate PASS',flush=True)
    assert len(differences)==len(links)==prior['link_count']
    assert all(sha(path)==value for path,value in hashes.items())
    result={'status':'TRAIN_ONLY_CH2_COST_ASSIGNMENT_AUDIT_PASS_NOT_PREDICTIVE_VALIDATION',
        'source_sha256':hashes,'train_rows':len(wanted),'train_records':len(records),'links_checked':len(differences),
        'maximum_original_link_cost_difference':max(differences),'farms':farms,'synthetic_checks':toy,
        'components_checked':len(sizes),'cycle_count_checked':len(cycles),'cycles':sorted(cycles),
        'query_inputs_parsed':0,'heldout_labels_parsed':0,'gap_inputs_parsed':0,
        'model_fit':False,'performance_evaluated':False,'adoption_permitted':False,
        'limits':['Training-label-conditioned graph optimality does not establish physical source identity or predictive gain',
                  'Original chosen assignment may tie with other optima; original graph remains unchanged',
                  'Future query-prefix source assignment, cycle fallback and full predictive validation remain']}
    out=HERE/'CH2_reference_independent_audit_v1.json';assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Independent train-only CH2 assignment audit complete; no predictive score',flush=True)

if __name__=='__main__':main()
