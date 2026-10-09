"""최종9joblib 구성원 혼합을 별도 특징 계산으로 검산. run/recipe import 없음."""
from pathlib import Path
import csv,json,sys,math,hashlib,subprocess,gc
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
L=ROOT/'집/코덱스/local'/H.name;D=L/'model_full_clean_v1'
RAW=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog'];ACT=RAW[3:]
WEIGHTS={'et':.6,'lgb':.3,'mlp':.1};SEEDS=(7,101,2024)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf8'))
def cr(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def identify(r):p=r['row_id'].split('_');return p[0],int(p[1]),int(p[2])
def number(x):return float(x) if x else np.nan

def independent_features(rows,m):
    grouped=defaultdict(list)
    for r in rows:grouped[identify(r)[:2]].append(r)
    result=[]
    for (farm,day),g in sorted(grouped.items()):
        g.sort(key=lambda r:identify(r)[2]);assert [identify(r)[2] for r in g]==list(range(len(g)))
        history={c:[] for c in ACT};h0={c:number(g[0][c]) for c in RAW};seal=heat=co2count=openhours=switchth=switchsh=0;first=24;vmax=0;last=-1;oldth=oldsh=None
        early=sorted(d for f,d in m['training_days'] if f==farm and d<179)
        season=float(np.interp(day,early,early)) if day<179 else float(np.interp(day,m['season_notes'][farm]['anchor_day'],m['season_notes'][farm]['anchor_season_pav']))
        for k,r in enumerate(g):
            h=identify(r)[2];z={c:number(r[c]) for c in RAW};f=dict(row_id=r['row_id'],farm=farm,day=day,hour=h,season=season,**z)
            f.update(hr_sin=float(np.sin(2*np.pi*h/24)),hr_cos=float(np.cos(2*np.pi*h/24)),midnight=float(h==0))
            for c in RAW:f[c+'_h0']=h0[c]
            for c in ACT:
                if not math.isnan(z[c]):history[c].append(z[c])
                f[c+'_tdm']=math.fsum(history[c])/len(history[c]) if history[c] else np.nan
                f[c+'_tdz']=sum(v==0 for v in history[c])/len(history[c]) if history[c] else np.nan
            v=0 if math.isnan(z['act_vent']) else z['act_vent'];th=(0 if math.isnan(z['act_thermal']) else z['act_thermal'])>0;sh=(0 if math.isnan(z['act_shade']) else z['act_shade'])>0
            heaton=(0 if math.isnan(z['act_heating']) else z['act_heating'])>0;co2on=(0 if math.isnan(z['act_co2']) else z['act_co2'])>0
            seal=seal+1 if v==0 else 0;heat=heat+1 if heaton else 0;co2count+=int(co2on);openhours+=int(v>0);vmax=max(vmax,v)
            if v>0:first=min(first,h)
            if k and th!=oldth:switchth+=1;last=k
            if k and sh!=oldsh:switchsh+=1;last=k
            oldth,oldsh=th,sh
            f.update(seal_run=float(seal),vent_open_hours=float(openhours),first_open_hour=float(first),thermal_switches=float(switchth),shade_switches=float(switchsh),since_curtain_change=float(k-last if last>=0 else k+1),heat_run=float(heat),co2_hours=float(co2count),vent_max=float(vmax))
            result.append(f)
    return pd.DataFrame(result)
def predict_independent(rows,m):
    q=independent_features(rows,m);raw=np.zeros(len(q));parts={}
    for item in m['models']:
        p=D/item['file'];assert sha(p)==item['sha'];model=joblib.load(p)
        with threadpool_limits(limits=2):v=np.asarray(model.predict(q[item['columns']]),float)
        assert np.isfinite(v).all();parts[f"{item['kind']}_{item['seed']}"]=v.copy();raw+=WEIGHTS[item['kind']]*v/3
        del model;gc.collect()
    pred=np.empty(len(q));groups=defaultdict(list)
    for i,r in q.iterrows():groups[r.farm,int(r.day)].append(i)
    for ids in groups.values():
        prefix=[]
        for i in ids:
            prefix.append(float(raw[i]));pred[i]=min(max(.5*prefix[-1]+.5*math.fsum(prefix)/len(prefix),m['bounds'][0]),m['bounds'][1])
    return q,raw,pred,parts

def main():
    reg=js(H/'registration_v1.json');m=js(D/'model_manifest_v1.json');checks=js(H/'final_model_checks_v1.json')
    assert m['registration_sha']==sha(H/'registration_v1.json') and m['cv_score_sha']==sha(H/'final_score_v1.json')
    assert js(H/'final_score_v1.json')['status']=='FULL_DIAG10' and m['adoption'] is False and m['submission_artifact'] is False
    assert m['weights']==WEIGHTS and m['seeds']==list(SEEDS) and len(m['models'])==9
    assert {(x['kind'],x['seed']) for x in m['models']}=={(k,s) for k in WEIGHTS for s in SEEDS}
    for name,h in m['code_hashes'].items():assert sha(H/name)==h
    data=L/'dataset_full_train';cl=js(data/'cleaning_manifest_v1.json')
    for name,h in cl['hashes'].items():assert sha(data/name)==h and m['clean_data_hashes'][name]==h
    x=cr(data/'train_X_clean_v1.csv');y=cr(data/'train_y_clean_v1.csv');assert [r['row_id'] for r in x]==[r['row_id'] for r in y]
    assert len(x)==m['training_rows']==9600-24*len(m['removed_days']) and len(x)==len(y)
    assert Counter_days(x)=={tuple(v) for v in m['training_days']}
    receipts=js(D/'fit_receipts_v1.json');assert len(receipts)==9 and {(r['kind'],r['seed']) for r in receipts}=={(k,s) for k in WEIGHTS for s in SEEDS}
    assert all(r['train_rows']==len(x) and r['reload_maxdiff']<1e-12 for r in receipts)
    assert checks['status']=='PASS' and checks['model_manifest_sha']==sha(D/'model_manifest_v1.json') and checks['consumed_40_days_validation_reads']==0
    public={rid for rec in reg['folds'] for rid in rec['query_ids']};days=defaultdict(list)
    for r in x:
        if r['row_id'] in public:days[identify(r)[:2]].append(r)
    chosen=[]
    for farm in ('F13','F47'):
        for p2 in (False,True):
            ds=sorted(d for f,d in days if f==farm and (d>=179)==p2);assert len(ds)>=2
            chosen.extend((farm,d) for d in ds[:2])
    probe=[r for day in chosen for r in days[day]];assert len(probe)==192
    q,raw,pred,parts=predict_independent(probe,m)
    P=L/'independent_model_probe_v1';assert not P.exists();P.mkdir()
    ip=P/'input_v1.csv';op=P/'actual_prediction_v1.csv'
    with ip.open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(probe[0]));w.writeheader();w.writerows(probe)
    # Actual production prediction path runs in a separate process; this checker imports neither run nor recipe.
    completed=subprocess.run([sys.executable,str(H/'predict_model_v1.py'),'--model',str(D),'--input',str(ip),'--output',str(op)],capture_output=True,text=True,encoding='utf8')
    assert completed.returncode==0,(completed.stdout,completed.stderr)
    actual={r['row_id']:float(r['prediction']) for r in cr(op)};gap=max(abs(float(pred[i])-actual[rid]) for i,rid in enumerate(q.row_id));assert gap<1e-10,gap
    # Original saved in-memory probe is independently reconstructed too.
    oldx=cr(D/'replay_probe_input_v1.csv');oldy={r['row_id']:float(r['prediction']) for r in cr(D/'replay_probe_prediction_v1.csv')};oq,_,opr,_=predict_independent(oldx,m)
    oldgap=max(abs(float(opr[i])-oldy[rid]) for i,rid in enumerate(oq.row_id));assert oldgap<1e-10,oldgap
    pq=pd.DataFrame({'row_id':q.row_id,'raw_mix':raw,'prediction':pred,**parts});pq.to_csv(P/'independent_components_v1.csv',index=False)
    out=dict(status='FINAL_ARTIFACT_INDEPENDENT_PASS',training_rows=len(x),training_days=len(m['training_days']),removed_days=len(m['removed_days']),removed_rows=9600-len(x),final_models=9,probe_rows=192,probe_farm_pass_days=[list(x) for x in chosen],independent_vs_actual_prediction_maxdiff=gap,independent_vs_saved_probe_maxdiff=oldgap,source_sha=sha(__file__),manifest_sha=sha(D/'model_manifest_v1.json'),new_submission=False,adoption=False,consumed40_revalidation=False,components_sha=sha(P/'independent_components_v1.csv'),limitations=['최종추가40일은훈련자료무결성만확인했고새검증으로채점하지않음','모델일반화성능은공개360CV이며400finalmodel성능을직접채점한것이아님'])
    with (H/'final_artifact_independent_v1.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('FINAL_ARTIFACT_INDEPENDENT_PASS',out,flush=True)
def Counter_days(rows):
    g=defaultdict(list)
    for r in rows:g[identify(r)[:2]].append(identify(r)[2])
    assert all(sorted(v)==list(range(24)) for v in g.values());return set(g)
if __name__=='__main__':main()
