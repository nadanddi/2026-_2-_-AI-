from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent/'ec_anchor_trust_20261005_v2';N.mkdir(exist_ok=False)
s=(H/'run_v6.py').read_text(encoding='utf-8').replace("scale=StandardScaler();xx=scale.fit_transform(xb[eligible]);", """if not eligible.any():
  fit=dict(eligible_ids=ids(b.loc[eligible,'row_id']),features_sha=ar(xb[eligible]),cost_sha=ar(cost[eligible]),mode=mode,seed=seed,n=0,positive=0,mean=[0.]*23,scale=[1.]*23,coef=None,intercept=None,model_none=True,constant=0.,empty_fallback='unchanged v2')
  return ob.copy(),np.zeros(len(q)),np.zeros(len(q)),fit
 scale=StandardScaler();xx=scale.fit_transform(xb[eligible]);""")
s=s.replace("model_none=model is None)","model_none=model is None,constant=float(label[0]) if model is None else None)")
s=s.replace("if fit['model_none']:return np.full(len(x),float(fit['positive']==fit['n']))", "if fit['model_none']:return np.full(len(x),float(fit['constant']))")
s=s.replace("raw_ec_reads=0", "empty_training='do not fit; keep baseline unchanged',raw_ec_reads=0")
(N/'run_v6.py').write_text(s,encoding='utf-8')
v=(H/'verify_v5.py').read_text(encoding='utf-8')
v=v.replace("if m['model_none']:z=np.full(len(q),float(m['positive']==m['n']))", "if m['model_none']:z=np.full(len(q),float(m['constant']))")
v=v.replace("else:z=xx@np.asarray(m['coef'])+float(m['intercept']);delta=np.where(ok,.2*np.clip(z,-.3,.3),0.)", "else:z=np.full(len(q),m['constant']) if m['model_none'] else xx@np.asarray(m['coef'])+float(m['intercept']);delta=np.where(ok,.2*np.clip(z,-.3,.3),0.)")
v=v.replace("scores=[];segments=[];boots={};alpha=", "assert R.sha(H/'verify_learning_v4.py')==registration['learning_sha'] and learning['mode']==mode and learning['fit_receipt']==R.sha(H/f'fit_{mode}_v1.json');assert [(c['validator'],c['fold'],c['seed']) for c in learning['checks']]==[(v,k,s) for v,k in R.KEYS for s in R.SEEDS];scores=[];segments=[];boots={};alpha=")
v=v.replace("aggregate=fit['aggregate']))", "aggregate=fit['aggregate'],learning_receipt_sha=R.sha(lp)))")
(N/'verify_v5.py').write_text(v,encoding='utf-8')
l=(H/'verify_learning_v4.py').read_text(encoding='utf-8')
l=l.replace("xx=x[sel];means=", """if not sel.any():
   m=R.load(R.OUT/mode/f'{v}_{k}_{s}_fit.json');assert m['n']==0 and m['model_none'] and m['constant']==0 and m['mean']==[0.]*23 and m['scale']==[1.]*23;assert m['eligible_ids']==R.ids(b.loc[sel,'row_id']) and m['features_sha']==R.ar(x[sel]) and m['cost_sha']==R.ar(cost[sel]);checks.append(dict(validator=v,fold=k,seed=s,eligible_ids=m['eligible_ids'],target_cost_sha=m['cost_sha'],feature_sha=m['features_sha'],objective_gradient_max=0.,empty=True));continue
  xx=x[sel];means=""")
l=l.replace("scales[scales==0]=1", "eps=np.finfo(float).eps;variance=scales**2;constant=variance<=len(xx)*eps*variance+(len(xx)*means*eps)**2;scales[constant]=1")
l=l.replace("fit=0,predict=0,source=", "fit_receipt=R.sha(H/f'fit_{mode}_v1.json'),fit=0,predict=0,source=")
(N/'verify_learning_v4.py').write_text(l,encoding='utf-8')
(N/'crosscheck_v1.py').write_text((H/'crosscheck_v1.py').read_text(encoding='utf-8').replace('ec_anchor_trust_20261005_v1','ec_anchor_trust_20261005_v2'),encoding='utf-8')
(N/'preregistration_v1.md').write_text((H/'preregistration_v1.md').read_text(encoding='utf-8')+'\n\n## 점수 전 빈 내부 학습 집합 보완\n\nv1 actual GATE는 DIAG10 fold7에서 eligible0으로중단(부분21셀보존/score0). 새v2폴더에서 eligible0이면 보정0/분류확률0/scaler무학습/기존v2유지로고정한다. 첫가족25/두번째26 유지·alpha.025/26·기존규칙불변. 내부표본확보를위해split/reference/미래제한을완화하지않는다. worker run_v6.py·verify_v5.py·verify_learning_v4.py·crosscheck_v1.py를이새폴더에서사전등록한뒤실행한다. 검증기StandardScaler상수판정은sklearn의near-constant floating-bound 공식을재현하며 학습모델설정변경0. 실패원본재학습/덮어쓰기0.\n',encoding='utf-8')
print(N)
