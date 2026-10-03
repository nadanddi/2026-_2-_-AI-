from prepare import *
def main():
 world=joblib.load(O/'world.joblib');lab=world['lab'];rows=[]
 for fold in [3,5,7,8]:
  fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==fold);tm,vm=S.common.split_mask(lab,fd);tr=lab[tm];daily=tr.groupby(['farm','day']).agg(y=('sub_temp','mean'),air=('in_temp','mean'),n=('in_temp','count'));warm=set(daily[(daily.n==24)&(daily.y-daily.air>=2)].index)
  for seed in range(1,9):
   z=np.load(R/f'집/코덱스/local/temp_tk_season_20261003_v1/pfn_DIAG10_{fold}_{seed}.npz');context=z['context_row_id'].astype(str);keys=[(s[:3],int(s[4:7])) for s in context];assert set(context).issubset(set(tr.row_id));count=sum(k in warm for k in keys);count178=sum(s.startswith('F47_178_') for s in context);rows.append(dict(fold=fold,context=seed,context_rows=len(context),warm_training_days=';'.join(f'{f}/{d}' for f,d in sorted(warm)),warm_context_rows=count,F47_178_rows=count178))
 pd.DataFrame(rows).to_csv(H/'temperature_context_rare_coverage.csv',index=False);print(pd.DataFrame(rows).query('fold == 5').to_string(index=False))
if __name__=='__main__':main()
