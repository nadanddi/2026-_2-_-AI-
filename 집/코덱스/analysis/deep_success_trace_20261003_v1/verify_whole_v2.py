from prepare import *
import math
def mean(v):return math.fsum(float(x) for x in v)/len(v)
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def main():
 rows=pd.read_csv(H/'whole_model_effect_hours_v2.csv',float_precision='round_trip');summary=pd.read_csv(H/'whole_model_effects_v2.csv',float_precision='round_trip');total=0
 for key,q in rows.groupby(['target','farm','day','seed','group','donor_day']):
  z=summary[(summary.target==key[0])&(summary.farm==key[1])&(summary.day==key[2])&(summary.seed==key[3])&(summary.group==key[4])&(summary.donor_day==key[5])].iloc[0];assert len(q)==24
  y=q.y.to_numpy();p=q.ALL.to_numpy();b=q.base.to_numpy();e=p-y;be=b-y;score=math.sqrt(mean(e*e));bs=math.sqrt(mean(be*be));close(z['mean'],mean(p));close(z['bias'],mean(e));close(z['rmse'],score);close(z.all_rmse_change,score-bs);close(z.all_mean_shift,mean(p-b));total+=5
  for col,field in [('CPU','cpu_rmse_change'),('PFN_only','pfn_only_rmse_change')]:
   ee=q[col].to_numpy()-y;close(z[field],math.sqrt(mean(ee*ee))-bs);total+=1
 expected={('TEMP',k,s) for k in [3,5,7,8] for s in range(1,9)}|{('EC',k,s) for k in [3,6] for s in range(1,5)};got=set();batch=[]
 for folder,target in [(O/'pfn_v3','TEMP'),(O/'pfn_ec_cpu','EC')]:
  for path in folder.glob(f'PFN_{target}_*.npz'):
   _,target,k,s=path.stem.split('_');got.add((target,int(k),int(s)));z=np.load(path);batch.append((target,float(z['maxdiff'])));spec=[json.loads(s) for s in z['specs']];assert len(z['context_row_id'])==2000;assert np.isfinite(z['prediction']).all();total+=2
 assert got==expected;assert max(v for target,v in batch if target=='EC')<1e-5
 cov=pd.read_csv(H/'temperature_context_rare_coverage.csv');assert len(cov)==32;warm=cov[cov.fold==5];assert warm.F47_178_rows.tolist()==[1,2,1,0,0,1,0,0]
 # Source temperature labels / public EC values, no reserved target ingestion.
 ty=pd.read_csv(Path(S.env.DATA)/'train_y.csv',usecols=['row_id','sub_temp']);tm=dict(zip(ty.row_id,ty.sub_temp));ec=pd.read_csv(R/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv',usecols=['row_id','sub_ec','validator','seed']);ec=ec[(ec.validator=='DIAG10')&(ec.seed==7)];em=dict(zip(ec.row_id,ec.sub_ec))
 for q in rows.itertuples():close(q.y,(tm if q.target=='TEMP' else em)[f'{q.farm}_{q.day:03d}_{q.hour:02d}']);total+=1
 savej(H/'whole_verification_v2.json',dict(status='PASS_WITH_DOCUMENTED_TEMP_PFN_NUMERICAL_APPROXIMATION',checks=total,contexts=len(got),temp_small_batch_maxdiff=max(v for t,v in batch if t=='TEMP'),ec_cpu_replay_maxdiff=max(v for t,v in batch if t=='EC'),first_exact_temp_full_batch_replay=0.,first_temp_response_batch_difference=.000118255615234375,notes=['Full mixture assembled with source formula and matching donor; manual fsum arithmetic','Temperature PFN small-batch sensitivity approximate; context/donor counts do not constitute independent samples','EC GPU supplementary failed and excluded; native CPU outputs match exactly','No adopted model, reserved EC access or test predictions']))
 print('WHOLE_VERIFICATION',total,flush=True)
if __name__=='__main__':main()
