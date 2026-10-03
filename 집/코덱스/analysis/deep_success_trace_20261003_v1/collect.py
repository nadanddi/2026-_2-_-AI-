from prepare import *
def main():
 folders=[O,O/'ec_v3'];selected=pd.read_csv(H/'selected_cases.csv');done=[]
 for target,folder in zip(['TEMP','EC'],folders):
  expected={(int(k),s) for k in selected[selected.target==target].fold.unique() for s in [7,101]};files=list(folder.glob(f'{target}_*_*_done.json'));assert len(files)==len(expected),(target,files)
  done.extend([json.loads(p.read_text(encoding='utf-8')) for p in files])
 for kind in ['effects','baseline','parts','replay']:
  frames=[pd.read_csv(p,float_precision='round_trip').assign(target='TEMP' if p.name.startswith('TEMP') else 'EC') for folder in folders for p in folder.glob(f'*_*_*_{kind}.csv') if p.name.startswith(('TEMP_','EC_'))]
  pd.concat(frames).to_csv(H/(kind+'.csv'),index=False)
 a=pd.read_csv(H/'effects.csv');b=pd.read_csv(H/'baseline.csv');assert not a.duplicated(['target','farm','day','seed','group','donor_day']).any()
 agg=a.groupby(['target','farm','day','group']).agg(mean_rmse_change=('rmse_change','mean'),median_rmse_change=('rmse_change','median'),min_rmse_change=('rmse_change','min'),max_rmse_change=('rmse_change','max'),mean_prediction_shift=('mean_shift','mean'),positive_fraction=('rmse_change',lambda v:float((v>0).mean())),n=('rmse_change','size')).reset_index()
 agg['rank']=agg.groupby(['target','farm','day']).mean_rmse_change.rank(ascending=False,method='min');agg.to_csv(H/'group_effect_summary.csv',index=False)
 savej(H/'model_replay.json',dict(status='PASS',fits=len(done),max_difference=max(q['replay_maxdiff'] for q in done),effects=len(a),baseline_rows=len(b),invalid_ec_v2='Column order used replace(day) vs original remove(day)+append(season); excluded, correct ec_v3 reproduces source'))
 print(agg[agg['rank']<=3].sort_values(['target','farm','day','rank']).to_string(index=False));print(pd.read_csv(H/'raw_temperature_shift.csv').to_string(index=False));print(pd.read_csv(H/'parts.csv').query('target == "TEMP" and seed == 7 and day in [191,194]')[['farm','day','component','mean','rmse']].to_string(index=False))
if __name__=='__main__':main()
