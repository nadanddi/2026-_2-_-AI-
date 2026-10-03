"""Preserve draft v1; add cache guards before any model experiment is run."""
from pathlib import Path
H=Path(__file__).resolve().parent
text=(H/'run.py').read_text(encoding='utf-8')
old=" z=arrayfile(INNER/f'{v}_{k}_cpu.npz')\n a=idx.reindex(z['inner_train_id']).reset_index();b=idx.reindex(z['row_id']).reset_index()"
new=""" z=arrayfile(INNER/f'{v}_{k}_cpu.npz')
 old=arrayfile(S.OUT/f'E_{v}_{k}_cpu.npz')
 assert idx.index.is_unique and len(z['inner_train_id'])>0 and len(z['row_id'])>0
 assert len(np.unique(z['inner_train_id']))==len(z['inner_train_id']) and len(np.unique(z['row_id']))==len(z['row_id'])
 assert np.array_equal(z['inner_train_id'],old['inner_train_id']) and np.array_equal(z['row_id'],old['row_id'])
 assert np.array_equal(old['outer_train_id'],tr.row_id)
 im,iv=S.inner(tr)
 for f in ['F13','F47']:
  if ((tr.farm==f)&(tr.day>=179)).any() and not ((tr.farm==f)&(tr.day>=179)&im).any():iv &= ~((tr.farm==f)&(tr.day>=179)).to_numpy()
 selected=set(tr.loc[iv,['farm','day']].itertuples(index=False,name=None))
 banned={(f,int(day)+j) for f,day in selected for j in [-1,0,1]}
 im=np.asarray([(f,int(day)) not in banned for f,day in zip(tr.farm,tr.day)])
 assert np.array_equal(z['inner_train_id'],tr.loc[im,'row_id']) and np.array_equal(z['row_id'],tr.loc[iv,'row_id'])
 a=idx.reindex(z['inner_train_id']).reset_index();b=idx.reindex(z['row_id']).reset_index()"""
assert old in text;text=text.replace(old,new)
old=" if missing:return record"
new=""" if missing:
  idx=lab.set_index('row_id');checked=[]
  for v,k,tm,vm in folds:
   needed=[INNER/f'{v}_{k}_cpu.npz']+[INNER/f'{v}_{k}_pfn_{s}.npz' for s in [1,2,3,4]]
   if not all(p.exists() for p in needed):continue
   tr,va=lab[tm].reset_index(drop=True),lab[vm].reset_index(drop=True)
   a,b,z,bag=full_inner(v,k,tr,idx)
   for s in [7,101,2024]:original_r3(v,k,s,tr,va)
   checked.append(dict(validator=v,fold=k,n_inner_train=len(a),n_inner_query=len(b)))
  record['ready_fold_id_audit']=checked;return record"""
assert old in text;text=text.replace(old,new)
old=" idx=lab.set_index('row_id');checked=[]\n for v,k,tm,vm in folds:"
new=""" expected={(v,k,s,r) for v,k,tm,vm in folds for s in [7,101,2024] for r in lab.loc[vm,'row_id']}
 observed=set(zip(od.validator,od.fold,od.seed,od.row_id));assert observed==expected
 idx=lab.set_index('row_id');checked=[]
 for v,k,tm,vm in folds:"""
assert old in text;text=text.replace(old,new)
old="  for s in [7,101,2024]:original_r3(v,k,s,tr,va)\n  checked.append"
new="""  for s in [7,101,2024]:
   original_r3(v,k,s,tr,va)
   g=od[(od.validator==v)&(od.fold==k)&(od.seed==s)].set_index('row_id').reindex(va.row_id)
   assert g[['y','baseline','candidate']].notna().all().all() and np.isfinite(g[['y','baseline','candidate']]).all().all()
   assert np.max(abs(g.y.to_numpy()-va.sub_ec.to_numpy()))<1e-12
   ref=preflight_outer[(preflight_outer.validator==v)&(preflight_outer.validation_fold==k)&(preflight_outer.seed==s)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
   assert np.array_equal(g.baseline.to_numpy(),ref)
  checked.append"""
assert old in text;text=text.replace(old,new)
text=text.replace('def preflight(lab,folds):','def preflight(lab,folds,preflight_outer):').replace('check=preflight(lab,folds)','check=preflight(lab,folds,outer)')
text=text.replace("S.sha(H/'run.py')","S.sha(Path(__file__))").replace("H/'preparation_v1.json'","H/'preparation_v2.json'")
text=text.replace("assert model is not None\n     fresh=", "if model is None:continue\n     fresh=")
text=text.replace("   oflag=prefix(rr,q)>=.9;od=", "   assert np.isfinite(reference).all()\n   oflag=prefix(rr,q)>=.9;od=")
dest=H/'run_v2.py';assert not dest.exists();compile(text,str(dest),'exec');dest.write_text(text,encoding='utf-8')
note='''# 실행 전 ID 정합성 보강 v2 · 가족17 유지

v1 코드와 --prepare 기록을 보존한다. 실행 파일은 run_v2.py다. 예측 수식/시드/범위/λ/게이트/판정 기준은 v1과 같으며 새 모델 학습은 아직0이다.

비평_v7의 반례를 반영해 빈/중복/누락 ID, 원 E cache의 순서, 현재 S.inner+기존 innerseasonal 선택 규칙의 순서까지 비교한다. 완료된 캐시만이라도 준비 단계에서 이 검사를 실행하며 전체 완료 전 학습은 하지 않는다. 최종관문에서는 CPU보정 OOF의 정확한 fold×seed×row key와 현재공개정답/outer baseline 값을 연결해 검사한다. NaN/무한값은 중단한다. 첫fold 반복 감사는 specialist가 없는 fallback일 때 건너뛰는 것으로 정책과 맞춘다.

원 HM1 캐시를 새 모델 적합 대신 쓰지 않는다. outer specialist를 모두 새 적합한다. 이전공개 최적비중을 고르거나 게이트/λ를 변경하지 않는다. 진행 중 CPU의 소스/캐시는 수정하지 않는다. 전체가족이18로 늘면 v1의17보다 엄격한 .025/18로 공개선별하며 이미본후보채택을 바꾸지 않는다.
'''
dest=H/'preregistration_v2.md';assert not dest.exists();dest.write_text(note,encoding='utf-8')
print('Wrote run_v2.py and preregistration_v2.md; no fitting.')
