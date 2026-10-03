from models import *
def main():
 world=joblib.load(O/'world.joblib');rows=[];checks=[];gr=json.loads((H/'TEMP_groups.json').read_text(encoding='utf-8'));groupof={c:g for g,cs in gr.items() for c in cs}
 for case in [z for z in world['selected'] if z['target']=='TEMP']:
  f,day,k=case['farm'],case['day'],case['fold'];q=world['lab'][(world['lab'].farm==f)&(world['lab'].day==day)].sort_values('hour')
  for seed in [7,101]:
   model=joblib.load(O/f'TEMP_{k}_{seed}_model.joblib')
   for part,tree,cols in [('BASE_residual',model.res,model.ct),('CODEX_residual',model.tree,S.FEATURE_COLUMNS)]:
    contribution=tree.booster_.predict(q[cols],pred_contrib=True);pred=tree.predict(q[cols]);checks.append(dict(farm=f,day=day,seed=seed,part=part,maxdiff=float(np.max(np.abs(contribution.sum(axis=1)-pred)))))
    for g in gr:
     ids=[i for i,c in enumerate(cols) if groupof[c]==g];rows.append(dict(farm=f,day=day,seed=seed,part=part,group=g,mean_contribution=float(contribution[:,ids].sum(axis=1).mean())))
    rows.append(dict(farm=f,day=day,seed=seed,part=part,group='expected_tree_output',mean_contribution=float(contribution[:,-1].mean())))
 assert max(z['maxdiff'] for z in checks)<1e-8
 pd.DataFrame(rows).to_csv(H/'temperature_tree_contributions.csv',index=False);savej(H/'tree_contribution_verification.json',dict(status='PASS',checks=checks))
if __name__=='__main__':main()
