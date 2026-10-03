from diagnose import *
def main():
 a=pd.read_csv(H/'hours.csv',float_precision='round_trip');d=pd.read_csv(H/'all_days.csv',float_precision='round_trip');rows=[]
 keys=sorted({(s[:3],int(s[4:7])) for s in np.load(R/'집/코덱스/local/statistical_experiments_20261003_v1/T_DIAG10_5_cpu.npz')['outer_train_id']})
 d['in_train']=[(f,int(day)) in keys for f,day in zip(d.farm,d.day)]
 for f in ['F13','F47']:
  for late in [False,True]:
   for scope in ['all','train','heldout']:
    q=d[(d.farm==f)&((d.day>=179)==late)&(d.air_n==24)]
    if scope!='all':q=q[q.in_train.eq(scope=='train')]
    pos=q[q.gap>=2];rows.append(dict(farm=f,late=late,scope=scope,n=len(q),warm_n=len(pos),warm_days=pos.day.tolist(),max_gap=float(q.gap.max()),median_gap=float(q.gap.median())))
 out=dict(coverage=rows,pair_delta=dict(air=float(a[a.farm=='F47'].in_temp.mean()-a[a.farm=='F13'].in_temp.mean()),truth=float(a[a.farm=='F47'].sub_temp.mean()-a[a.farm=='F13'].sub_temp.mean()),pred=float(a[a.farm=='F47'].W30G.mean()-a[a.farm=='F13'].W30G.mean())))
 (H/'coverage.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
