# -*- coding: utf-8 -*-
"""RCH1 (2026-10-14 집 클로드; 사용자: '일단 해봐' — 좋아진 링커가 평가의 어려운 기록까지 닿는지).  서술(판정 아님, 평가 입력은 규정대로 짝 계산에만).
평가 60기록: 짝 후보 = v15r15_eval(prev0..5, p).  규정: 평가 기록을 거치는 단계는 같은 ID 앞 번호 평가 기록만(다른 ID·뒤 번호 평가 기록 후보는 버림), 학습 기록은 자유.
GP-RS: 후보 a 를 '정답 있는 학습 기록이 거울 짝 표 1순위로 더 높은 확률로 지목' 하면 제외 → 다음 후보.
LK3 q: 거울 표 cert 짝으로 적합한 LK3 모형(LK2 10특징 + 0~5시 불변값 차 5 + 결측 표시)을 평가 짝에 적용.
사슬: GP-RS 짝을 따라가며 (가) 모든 단계 q ≥ .8, (나) 누적 Π q ≥ .3 (DS9 식) — 정답 있는 학습 기록(조상)에 닿으면 멈춤, 최대 6단계.
비교 기준: VC1(v15R p0, 누적 ≥ .3) — 예측 고EC 8기록 중 조상 고EC 2.
보고: 60기록·예측 고EC(14회차 EC 하루 평균 ≥ 1.0)별 조상 도달 여부·단계·조상 하루 평균 EC.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LK2SRC = open(os.path.join(HERE, "lk2_hop_correct_model_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "lk2_hop_correct_model_v1.py")
exec(LK2SRC[:LK2SRC.index("PMr = PM.reset_index()")])   # hopfeats, FEATS, CERT, TRAIN, idx, mirror_lab, PM ...
__file__ = os.path.join(HERE, "rch1_eval_reach_v1.py")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC  # noqa: E402
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
LK3SRC = open(os.path.join(HERE, "lk3_invariant_hop_model_v1.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(HERE, "lk3_invariant_hop_model_v1.py"), "__name__": "lk3"}; exec(LK3SRC[:LK3SRC.index("fit, ev = ")], ns)
INV, F1, MT = ns["INV"], ns["F1"], ns["MT"]; DF = [f"d_{c}" for c in INV.columns]
fit = MT[MT.ok.notna()]; sc = StandardScaler().fit(fit[F1]); LR3 = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(fit[F1]), fit.ok.astype(int))
def q3(r, a, p0, p1, pn):
    f = hopfeats(r, a, p0, p1, pn, TRAIN.copy())
    for c in INV.columns:
        d = abs(INV[c].get(r, np.nan) - INV[c].get(a, np.nan)); f[f"d_{c}"] = d if np.isfinite(d) else MT[f"d_{c}"].median(); f[f"d_{c}_na"] = float(not np.isfinite(d))
    return float(LR3.predict_proba(sc.transform(pd.DataFrame([f])[F1]))[0, 1])
E = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv"))
MIR = {r: (pv, p0) for r, pv, p0 in zip(PM.index, PM.prev0, PM.p0)}
labeled = lambda r: bool(TRAIN[idx[r]]) and np.isfinite(DM[idx[r]])
def allowed(r, a): return bool(TRAIN[idx[a]]) or (a[:3] == r[:3] and NUM[idx[a]] < NUM[idx[r]])
def pick(r):
    if r in set(E.rec):
        q = E[E.rec == r].iloc[0]; cands = [(q[f"prev{j}"], float(q[f"p{j}"])) for j in range(6) if isinstance(q[f"prev{j}"], str)]; pn = float(q.p_null)
    else:
        q = PM.loc[r]; cands = [(q[f"prev{j}"], float(q[f"p{j}"])) for j in range(6) if isinstance(q[f"prev{j}"], str)]; pn = float(q.p_null)
    cands = [(a, p) for a, p in cands if a != r and allowed(r, a)]
    for k, (a, p) in enumerate(cands):
        rivals = [b for b, (pb, ppb) in MIR.items() if b != r and labeled(b) and pb == a and ppb > p]
        if rivals: continue
        p1 = cands[k + 1][1] if k + 1 < len(cands) else 0.0
        return a, p, q3(r, a, p, p1, pn)
    return None, 0.0, 0.0
def chain(r, mode):
    cur, P, steps = r, 1.0, 0
    for _ in range(6):
        a, p, q = pick(cur)
        if a is None: return None
        P *= q; steps += 1
        if (mode == "all8" and q < .8) or (mode == "prod3" and P < .3): return None
        if labeled(a): return a, steps, P
        cur = a
    return None
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; pm = P14.groupby("rec").sub_ec.mean()
rows = []
for r in E.rec:
    a, p, q = pick(r); out = dict(rec=r, pred=round(float(pm[r]), 3), predH=pm[r] >= 1.0, pick=a, q=round(q, 3), pick_labeled=labeled(a) if a else False)
    for mode in ("all8", "prod3"):
        c = chain(r, mode); out[f"anc_{mode}"] = c[0] if c else None; out[f"steps_{mode}"] = c[1] if c else np.nan; out[f"ancEC_{mode}"] = round(float(DM[idx[c[0]]]), 3) if c else np.nan
    rows.append(out)
R = pd.DataFrame(rows); pd.set_option("display.width", 220)
print("eval 60: 1-step pick labeled & q>=.8:", int((R.pick_labeled & (R.q >= .8)).sum()), "| reach labeled ancestor all-hops q>=.8:", int(R.anc_all8.notna().sum()), "| prod q>=.3:", int(R.anc_prod3.notna().sum()))
print("predicted-high eval records:"); print(R[R.predH][["rec", "pred", "pick", "q", "anc_all8", "steps_all8", "ancEC_all8", "anc_prod3", "steps_prod3", "ancEC_prod3"]].to_string(index=False))
print("\nancestor high (>=1.0) among predicted-high: all8", int((R.predH & (R.ancEC_all8 >= 1.0)).sum()), "| prod3", int((R.predH & (R.ancEC_prod3 >= 1.0)).sum()), "(VC1 baseline: 2)")
R.to_csv(os.path.join(HERE, "rch1_eval_reach_v1.csv"), index=False)
