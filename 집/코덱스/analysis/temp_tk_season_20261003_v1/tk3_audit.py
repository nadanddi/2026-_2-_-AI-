from world import *
from analyze import boot,summary
lab,pfn,ct,phc,wb,wp,wv,sets,z=worlds();frames=[];members=[];audit=[]
for name,folds in sets[:3]:
    for k,fd in enumerate(folds):
        tm,vm=common.split_mask(lab,fd);d=lab.loc[vm,['row_id','farm','day','hour','sub_temp','in_temp']].copy();d['validator']=name;d['fold']=k;mp=dict(np.load(OUT/f'members_{name}_{k}.npz',allow_pickle=True));assert np.array_equal(d.row_id,mp['row_id'])
        m=d.copy()
        for key,v in mp.items():
            if key!='row_id':m[key]=v
        members.append(m);g=np.where(d.in_temp.isna(),1,np.clip((d.in_temp.to_numpy()-8)/2,0,1))
        for seed in [7,101]:
            assert np.max(np.abs(mp[f'mask_base_{seed}']-z[f'{name}__MASK__{seed}'][vm]))<1e-7
            for context in ['1-8','17-24']:
                old=np.load(Path(env.LOCAL)/f'web_tabpfn_{"v2" if context=="1-8" else "v6"}_temp_{name}.npy');p=old.mean(0)[vm] if context=='1-8' else old[:8].mean(0)[vm]
                f=d.copy();f['variant']='TC2_original_PFN';f['seed']=seed;f['context']=context;f['base']=w30(mp[f'mask_base_{seed}'],mp['codex_base'],p,g);f['candidate']=w30(mp[f'mask_season_{seed}'],mp['codex_season'],p,g);frames.append(f)
m=pd.concat(members,ignore_index=True);o=pd.concat(frames,ignore_index=True);o.to_csv(OUT/'TK3_original_mix.csv',index=False)
for tag in ['TC1','TC2']:
    old=pd.read_csv(Path(env.LOCAL)/f'temp_{tag}_oof.csv');j=m.merge(old,on=['row_id','validator','fold'],suffixes=('_own','_claude'),validate='one_to_one')
    for c in (['mask_base','mask_seas','codex_base','codex_seas'] if tag=='TC1' else ['mask_base_7','mask_seas_7','mask_base_101','mask_seas_101','codex_base','codex_seas']):
        own=c.replace('seas','season')
        if tag=='TC1' and c.startswith('mask'):own+='_7'
        own=own+'_own' if own in old.columns else own;theirs=c+'_claude' if c in m.columns else c;diff=float(np.max(np.abs(j[own]-j[theirs])));assert diff<1e-7;audit.append(dict(source=tag,column=c,n=len(j),maxdiff=diff))
    if tag=='TC2':
        for seed in [7,101]:
            f=o[(o.seed==seed)&(o.context=='1-8')].merge(old,on=['row_id','validator','fold'],validate='one_to_one')
            for own,theirs in [('base',f'w30_base_{seed}'),('candidate',f'w30_seas_{seed}')]:
                diff=float(np.max(np.abs(f[own]-f[theirs])));assert diff<1e-7;audit.append(dict(source='TC2_mixed',column=theirs,n=len(f),maxdiff=diff))
scores=summary(o);pd.DataFrame(scores).to_csv(HERE/'TK3_scores.csv',index=False);(HERE/'TK3_reproduction.json').write_text(json.dumps(dict(status='PASS',comparisons=audit,scores=scores,hinge_columns=[c for c in ct if 'day_hinge' in c]),ensure_ascii=False,indent=2),encoding='utf-8')
print('TC1/TC2 independent reproduction PASS, maxdiff',max(r['maxdiff'] for r in audit))
for r in scores:
    if r['segment']=='all':print(json.dumps(r))
