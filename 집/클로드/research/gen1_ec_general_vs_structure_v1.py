# -*- coding: utf-8 -*-
"""GEN1-EC (2026-10-14 집 클로드; 사용자: '완전 일반 구성 측정 진행해봐').  서술(판정 아님).  EC.
같은 검증기(ec_rv1 CUR: DIAG10·P2LOO·EL1 x 시드 47·1414·6464)에서:
  N   = nosg (R3 + TabPFN + 수축, SG2 전 — 일반 쪽; 단 계절 지표는 외기 일치 추정 포함)
  NM  = 0.9 N + 0.1 MT1(MTL GRU, 6.894 의 B 모델, 자기 기록 입력만)
  NH  = N + HR5 보정(.5; 보정기 학습 기준도 nosg = DIAG10 nosg 시드 평균으로 다시)
  NHM = 0.9 NH + 0.1 MT1
  S   = sg (9회차형, SG2 포함 — 구조 의존 모듈 하나 포함)
W2·B5·앵커는 검증기 재현이 어려워 리더보드(9회차 .1384 → 14회차 .1077)로 대신.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py")
exec(src[:src.index("TRK = sorted(TR)")])
src = open(os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py")
exec(src[:src.index("O = pd.read_csv")])
__file__ = os.path.join(HERE, "gen1_ec_general_vs_structure_v1.py")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
# HR5 with nosg base
NB = O[O.validator == "DIAG10"].groupby("row_id").nosg.mean()
TR = {}
for r, g in A.groupby("rec"):
    rid_ = [f"{r}_{h:02d}" for h in g.hour]; b = NB.reindex(rid_).values
    if len(g) == 24 and np.isfinite(b).all() and np.isfinite(g.y.values).sum() >= 12 and r in INP:
        TR[r] = (b, float(np.nanmean(g.y.values) - b.mean()))
TRK = sorted(TR); TF = {h: np.stack([feat(INP[r], TR[r][0], h) for r in TRK]) for h in HS}; TD = np.array([TR[r][1] for r in TRK])
for h in HS: TF[h] = np.where(np.isfinite(TF[h]), TF[h], np.nanmean(TF[h], 0))
tfarm = np.array([r[:3] for r in TRK]); tnum = np.array([int(r[4:]) for r in TRK]); print("HR5-nosg training records", len(TRK), flush=True)
DH = {}
for v in ("DIAG10", "EL1", "P2LOO"):
    for k, Gk in O[O.validator == v].groupby("fold"):
        val = set(Gk.rec); hide = np.zeros(len(TRK), bool)
        for r in val:
            f, n = r[:3], int(r[4:]); hide |= ((tfarm == f) & (np.abs(tnum - n) <= 1)) | ((tfarm == OTHER[f]) & (np.abs(tnum - n) <= 3))
        models = {}
        for h in HS:
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, n_estimators=4, random_state=0, ignore_pretraining_limits=True); m.fit(TF[h][~hide], TD[~hide]); models[h] = m
        for (sd, r), g in Gk.groupby(["seed", "rec"]):
            g = g.sort_values("hour"); base = g.nosg.values
            if len(base) != 24: continue
            Q = np.stack([feat(INP[r], base, h) for h in HS]); Q = np.where(np.isfinite(Q), Q, np.stack([TF[h].mean(0) for h in HS]))
            dh = {h: float(models[h].predict(Q[j:j + 1])[0]) for j, h in enumerate(HS)}
            DH[(v, k, sd, r)] = np.array([dh[max(x for x in HS if x <= hh_)] for hh_ in range(24)])
    print("HR5-nosg", v, "done", flush=True)
# MT1 (B) predictions
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
MP = {}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for k, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        p, _ = fit_predict(tr, te, "B")
        for j, r in enumerate(te): MP[(v, r)] = p[j]
    print("MT1", v, "done", flush=True)
res = []
for (v, k, sd), G in O.groupby(["validator", "fold", "seed"]):
    for r, g in G.groupby("rec"):
        g = g.sort_values("hour")
        if (v, k, sd, r) not in DH or (v, r) not in MP: continue
        N_ = g.nosg.values; mt = MP[(v, r)][g.hour.values]; NH = N_ + .5 * DH[(v, k, sd, r)]
        res.append(pd.DataFrame(dict(validator=v, seed=sd, rec=r, day=g.day.values, y=g.sub_ec.values, N=N_, NM=.9 * N_ + .1 * mt, NH=NH, NHM=.9 * NH + .1 * mt, S=g.sg.values)))
B = pd.concat(res, ignore_index=True); B["p2"] = B.day >= 179; B["hi"] = B.groupby(["validator", "seed", "rec"]).y.transform("mean") >= 1.0
out = []
for v, g in B.groupby("validator"):
    for sc, m in (("all", np.ones(len(g), bool)), ("p2", g.p2.values), ("p2_normal", (g.p2 & ~g.hi).values), ("p2_high", (g.p2 & g.hi).values)):
        if m.sum() == 0: continue
        out.append(dict(val=v, scope=sc, **{c: np.mean([rm((x[c] - x.y)[mm]) for (_, x), mm in zip(g.groupby("seed"), [m[g.seed.values == s] for s in sorted(g.seed.unique())])]) for c in ("N", "NM", "NH", "NHM", "S")}))
R = pd.DataFrame(out); pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
for c in ("NM", "NH", "NHM", "S"): R[c + "_vsN%"] = 100 * (R[c] / R.N - 1)
print(R[["val", "scope"] + [c + "_vsN%" for c in ("NM", "NH", "NHM", "S")]].round(2).to_string(index=False))
B.to_csv(os.path.join(HERE, "gen1_ec_rows_v1.csv.gz"), index=False)
