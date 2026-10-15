"""Full frozen-array audit reusing the critic's scalar prefix preparation only."""
from pathlib import Path
core=Path(__file__).with_name('n5_critic_first_v1.py')
prefix=core.read_text(encoding='utf-8').split("fd=reg['folds'][0];name=")[0]
exec(compile(prefix,str(core),'exec'))
result=json.loads((H/'n5_results_v2.json').read_text(encoding='utf-8'))
parent_result=json.loads((H/'n4_results_v2.json').read_text(encoding='utf-8'))
root_check=json.loads((H/'n5_verify_all_v1.json').read_text(encoding='utf-8'))
O=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1'
y=pd.read_csv(data/'train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').sub_temp
ans={s:{'ids':[],'a':[],'b':[],'q':[],'g':[]} for s in ['DIAG10','EXT10','EXT12']}
maxgap=0.;checks=0
def close(a,b):
 global maxgap,checks
 gap=abs(a-b);maxgap=max(maxgap,gap);checks+=1;assert gap<1e-12,(a,b)
for fd in reg['folds']:
 name=fd['set']+'_'+str(fd['fold']);pf=O/'n4_v1'/(name+'.npz');nf=O/'n5_v1'/(name+'.npz')
 pm=json.loads(pf.with_suffix('.json').read_text(encoding='utf-8'));nm=json.loads(nf.with_suffix('.json').read_text(encoding='utf-8'))
 assert sha(pf)==reg['parents'][name]['npz_sha']==pm['npz_sha']==nm['parent_npz_sha']
 assert sha(pf.with_suffix('.json'))==reg['parents'][name]['receipt_sha']
 assert sha(nf)==nm['npz_sha'] and nm['recipe_sha']==canon(reg)
 z=np.load(pf,allow_pickle=True);v=np.load(nf,allow_pickle=True);ids=v['row_id'].tolist()
 assert ids==z['row_id'].tolist()==fd['query_ids']==nm['query_ids']==pm['query_ids']
 assert len(ids)==len(set(ids)) and not set(ids)&set(fd['train_ids'])
 dayhours={}
 for rid in ids:dayhours.setdefault(rid[:7],[]).append(int(rid[8:10]))
 assert all(sorted(hr)==list(range(24)) for hr in dayhours.values())
 mn=np.array([states[i][0] for i in ids]);q=np.array([states[i][1] for i in ids]);g=np.array([states[i][2] for i in ids])
 assert np.array_equal(mn,v['prefix_min'],equal_nan=True) and np.array_equal(q,v['q']) and np.array_equal(g,v['g']) and np.array_equal(g,z['gate'])
 assert np.array_equal(v['BASELINE'],z['BASELINE']) and np.array_equal(v['N5'][q==g],z['N4'][q==g])
 assert np.array_equal(v['N5'][q==0],z['BASELINE'][q==0]) and np.array_equal(v['N5'][g==0],z['BASELINE'][g==0])
 assert np.isfinite(v['N5']).all()
 close(int((q==0).sum()),nm['q0_rows']);close(int((q<g).sum()),nm['q_reduced_rows'])
 for b,c,irr,p,qq,gg,out in zip(z['BASE'],z['CODEX'],z['IRR'],z['PFN'],q,g,v['N5']):
  calc=math.fsum([.4*float(b),(.6-.4*float(gg))*(float(c)+float(qq)*(float(irr)-float(c))),.4*float(gg)*float(p)]);close(calc,float(out))
 for k,seq in [('ids',ids),('a',v['BASELINE']),('b',v['N5']),('q',q),('g',g)]:ans[fd['set']][k].extend(seq)
metrics={}
for s,d in ans.items():
 ids=d['ids'];assert len(ids)==len(set(ids));truth=y.loc[ids].to_numpy();a=np.array(d['a']);b=np.array(d['b']);q=np.array(d['q']);g=np.array(d['g'])
 ea=np.array([float(p)-float(t) for p,t in zip(a,truth)]);eb=np.array([float(p)-float(t) for p,t in zip(b,truth)])
 mse=lambda v:math.fsum(float(x)**2 for x in v)/len(v)
 ra=math.sqrt(mse(ea));rb=math.sqrt(mse(eb));pct=100*(rb/ra-1)
 for k,v in [('base_rmse',ra),('candidate_rmse',rb),('pct',pct)]:close(v,result['results'][s][k])
 close(ra,parent_result['results'][s]['base_rmse'])
 corr=b-a;cross=2*math.fsum(float(x)*float(t) for x,t in zip(ea,corr))/len(ids);cmse=mse(corr)
 close(cross+cmse,rb*rb-ra*ra)
 m={'base_rmse':ra,'candidate_rmse':rb,'pct':pct,'rows':len(ids),'q0_rows':int((q==0).sum()),'g0_rows':int((g==0).sum()),'q_reduced_rows':int((q<g).sum()),'cross_2E':cross,'correction_mse':cmse,'delta_mse':rb*rb-ra*ra}
 for k in ['base_rmse','candidate_rmse','q0_rows','q_reduced_rows','rows']:close(m[k],root_check['results'][s][k])
 close(cross,root_check['results'][s]['error_delta_cross_2E']);close(cmse,root_check['results'][s]['correction_mse'])
 m['farm_segment']={}
 for f in ['F13','F47']:
  for late in [False,True]:
   ix=[i for i,rid in enumerate(ids) if rid[:3]==f and (int(rid[4:7])>=179)==late]
   if ix:m['farm_segment'][f+('_second' if late else '_first')]={'rows':len(ix),'base_rmse':math.sqrt(mse(ea[ix])),'candidate_rmse':math.sqrt(mse(eb[ix]))}
 if s=='DIAG10':
  blocks={}
  for i,rid in enumerate(ids):blocks.setdefault(rid[:3]+'_'+str(int(rid[4:7])//5),[]).append(i)
  keys=sorted(blocks);sa=np.array([math.fsum(float(ea[i])**2 for i in blocks[k]) for k in keys]);sb=np.array([math.fsum(float(eb[i])**2 for i in blocks[k]) for k in keys]);sz=np.array([len(blocks[k]) for k in keys]);rng=np.random.default_rng(reg['bootstrap_seed']);ds=[]
  for _ in range(20):
   draw=rng.integers(len(keys),size=(1000,len(keys)));den=sz[draw].sum(1);ds.extend((np.sqrt(sb[draw].sum(1)/den)-np.sqrt(sa[draw].sum(1)/den)).tolist())
  pv=float(np.mean(np.array(ds)>=0));ci=np.quantile(ds,[reg['alpha'],1-reg['alpha']]).tolist();close(pv,result['results'][s]['p_worse'])
  for x,zv in zip(ci,result['results'][s]['ci']):close(x,zv)
  m.update(p_worse=pv,ci=ci)
 metrics[s]=m
passed=all(v['candidate_rmse']<v['base_rmse'] for v in metrics.values()) and metrics['DIAG10']['p_worse']<reg['alpha'] and metrics['DIAG10']['ci'][1]<0
assert passed==result['pilot_pass']==False and result['adoption']==False and result['new_fit']==0
receipt={'status':'PASS_NUMERICS_PILOT_FAIL','checks':checks,'max_abs_gap':maxgap,'all12_scalar_prefix_arrays_source_pass':True,'results':metrics,'prefix_critic_source_sha':sha(core),'new_fit':0,'ec_numeric_labels_read':False,'full_prediction_causality_test':False}
with (H/'n5_critic_all_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
