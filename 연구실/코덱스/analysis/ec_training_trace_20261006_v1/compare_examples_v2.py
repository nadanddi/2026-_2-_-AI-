from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import numpy as np,pandas as pd
L=ROOT/'연구실/코덱스/local'/H.name
prep=json.loads((H/'preparation_v4.json').read_text(encoding='utf-8'));columns=prep['full_columns']
cases=[('actual1_7','query',['F47_161_00']),('actual0_7','query',['F47_160_00']),('actual1_7','train',['F47_139_00','F47_231_02','F47_157_00','F47_145_00']),('common_7','query',['F13_098_00','F13_112_00']),('common_7','train',['F47_119_00','F13_130_00','F13_104_00','F13_093_00'])]
out=[]
for context,role,ids in cases:
    p=L/(context+'.npz');meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['npz_sha']
    with np.load(p,allow_pickle=False) as z:
        rid=z[role+'_row_id'].tolist();matrix=z[role+'_X'];y=z[role+'_y']
        for target in ids:
            i=rid.index(target);record=dict(context=context,role=role,row_id=target,ec=float(y[i]));record.update({c:float(matrix[i,j]) for j,c in enumerate(columns)})
            if role=='train':
                query='F47_161_00' if context=='actual1_7' else 'F13_112_00';qi=z['query_row_id'].tolist().index(query);record['support_weight_for_query']=float(z['weights'][qi,i])
            out.append(record)
dest=H/'examples_v2.csv';assert not dest.exists();pd.DataFrame(out).to_csv(dest,index=False)
print(pd.DataFrame(out)[['context','role','row_id','ec','in_temp','in_hum','in_co2','act_heating','act_vent','act_circfan','season']].to_string(index=False))

