from world import *
z=dict(np.load(Path(env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True));rows={r:i for i,r in enumerate(z['row_id'])};items=[dict(np.load(OUT/f'pfn_DIAG10_0_{s}.npz')) for s in range(1,9)];old=np.load(Path(env.LOCAL)/'web_tabpfn_v2_temp_DIAG10.npy').mean(0);new=np.mean([p['base'] for p in items],axis=0);ix=[rows[r] for r in items[0]['row_id']];delta=new-old[ix]
answer=dict(maxdiff=float(np.max(np.abs(delta))),mean_absdiff=float(np.mean(np.abs(delta))),n=len(ix));(HERE/'cache_probe.json').write_text(json.dumps(answer,indent=2),encoding='utf-8');print(json.dumps(answer))
