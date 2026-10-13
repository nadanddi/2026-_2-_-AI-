# -*- coding: utf-8 -*-
"""DS9 (2026-10-13 집 클로드; 사용자: 'DS9 진행').  사전 고정(커밋 후 실행).  EC.  DS8 + LK2 짝 정답 확률 q.
DS8(6.824/6.826): 실패 원인 = 불확실한 ID 교차 짝 하나가 여러 사슬로 번짐(6.827).  LK2(6.828): 짝 정답 확률 q 가 p0 보다 정확(EV AUC .907 vs .806).
변경 하나: 사슬 누적 확신 P = Π q (p0 대신), 시작 상태 섞음·조상 판정 문턱(.3)·전이 적합 문턱(.8)도 q 로.  나머지 DS8 그대로.
q 교차 적합(같은 표로 맞춘 q 를 같은 표 판정에 쓰지 않음): EV 짝 = 거울 표 cert 짝으로 맞춘 모형(LK2 그대로, lk2_ev_hops_q_v1.csv 의 q),
  거울 짝(DIAG10·P2LOO·EL1) = EV 표 cert 짝으로 같은 특징·같은 방식(표준화 + 로지스틱 C=1)으로 맞춘 모형의 q.
판정(EC 분야 배정 규칙, 메모리 skill-not-luck 2026-10-13; 링커 의존 방법): (1) 주 P2LOO·EL1 모든 시드 ΔSSE < 0 AND (P2LOO 블록 P(악화) < .025 OR 나빠진 기록 0),
  (2) EV 모든 시드 전체 행 ΔSSE ≤ 0, (3) DIAG10 손해 감시(시드 평균 악화 AND 후보 나음 비율 < .025 이면 FAIL).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DS7SRC = open(os.path.join(HERE, "ds7_silent_zone_hmm_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ds7_silent_zone_hmm_v1.py")
exec(DS7SRC[:DS7SRC.index("def fit(obs, hs, vis, key):")])   # CR1 head (v7set, L7, TNA8, PM), zflag, silent, DM, CPREV(unused here)
__file__ = os.path.join(HERE, "ds9_ancestor_hmm_qlinks_v1.py"); rng = np.random.default_rng(20261130)
PMT = {r: (pv, p0) for r, pv, p0 in zip(PM.index, PM.prev0, PM.p0)}
PEV = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv")); PEVT = {r: (pv, p0) for r, pv, p0 in zip(PEV.rec, PEV.prev0, PEV.p0)}
from sklearn.preprocessing import StandardScaler  # noqa: E402
_ET = pd.read_csv(os.path.join(HERE, "lk2_ev_hops_q_v1.csv")); _MT = pd.read_csv(os.path.join(HERE, "lk2_mirror_hops_q_v1.csv"))
_F = ["lp0", "margin", "p_null", "cross", "dTin", "dTout", "date_yes", "date_no", "same_id_earlier", "hp2_agree"]
_fit = _ET[_ET.ok.notna()]; _sc = StandardScaler().fit(_fit[_F]); _LR = LogisticRegression(C=1.0, max_iter=1000).fit(_sc.transform(_fit[_F]), _fit.ok.astype(int))
_MT["qx"] = _LR.predict_proba(_sc.transform(_MT[_F]))[:, 1]
QM = dict(zip(_MT.rec, _MT.qx)); QE = dict(zip(_ET.rec, _ET.q))
PMT = {r: (pv, QM.get(r, 0.0)) for r, (pv, _p) in PMT.items()}; PEVT = {r: (pv, QE.get(r, 0.0)) for r, (pv, _p) in PEVT.items()}
print("q tables: mirror mean q %.3f (n q>=.8 %d) | EV mean q %.3f (n q>=.8 %d)" % (np.mean(list(QM.values())), sum(v >= .8 for v in QM.values()), np.mean(list(QE.values())), sum(v >= .8 for v in QE.values())), flush=True)
def fit(obs, hs, vis, key, picks):
    tr = [r for r in obs.index if r not in hs and TRAIN[idx[r]] and np.isfinite(DM[idx[r]])]; st = {r: DM[idx[r]] >= 1.0 for r in tr}
    pr = [(st[picks[r][0]], st[r]) for r in tr if r in picks and isinstance(picks[r][0], str) and picks[r][1] >= .8 and picks[r][0] in st]
    A = np.array([[sum(1 for a, b in pr if a == s0 and b == s1) + 1 for s1 in (0, 1)] for s0 in (0, 1)], float); A /= A.sum(1, keepdims=True)
    cs = np.array([obs[r] for r in tr]); ss = np.array([st[r] for r in tr]).astype(int); ys = np.array([DM[idx[r]] for r in tr])
    zs = np.array([zflag(idx[r], 23, vis, key) for r in tr]); prior = np.array([1 - ss.mean(), ss.mean()])
    pi = [(zs[ss == s].sum() + 1) / ((ss == s).sum() + 2) for s in (0, 1)]; reg = [np.polyfit(cs[ss == s], ys[ss == s], 1) for s in (0, 1)]
    LR = LogisticRegression().fit(cs.reshape(-1, 1), ss); po = prior[1] / prior[0]
    def lik(c, z):
        p = float(np.clip(LR.predict_proba([[c]])[0, 1], 1e-4, 1 - 1e-4)); return np.array([(pi[0] if z else 1 - pi[0]), (p / (1 - p)) / po * (pi[1] if z else 1 - pi[1])])
    return A, prior, lik, reg
def ancestor(r, picks, hs):
    chain, cur, P = [], r, 1.0
    for _ in range(6):
        if cur not in picks or not isinstance(picks[cur][0], str): return None
        pv, p0 = picks[cur]; P *= float(p0)
        if P < .3: return None
        if TRAIN[idx[pv]] and pv not in hs and np.isfinite(DM[idx[pv]]): return pv, chain, P
        chain.append(pv); cur = pv
    return None
def hmm_anc(r, g, obs, model, vis, key, anc):
    A, prior, lik, reg = model; pv, chain, P = anc
    alpha = P * np.array([0.0, 1.0]) + (1 - P) * prior
    for q in reversed(chain):
        alpha = alpha @ A
        if q in obs.index: alpha = alpha * lik(obs[q], zflag(idx[q], 23, vis, key)); alpha /= alpha.sum()
    a0 = alpha @ A; out = []
    for h, c, s in zip(g.hour, g.cm, g.sg):
        post = a0 * lik(c, zflag(idx[r], int(h), vis, key)); post /= post.sum()
        out.append(np.clip(s + sum(post[k] * np.polyval(reg[k], c) for k in (0, 1)) - c, lo_, hi_))
    return np.array(out)
res, blocks, fl = [], {}, []
def score(v, sd, G, new):
    y = G.y.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    d = (new - y) ** 2 - (sg - y) ** 2; f = lambda m: 100 * (rm((new - y)[m]) / rm((sg - y)[m]) - 1) if m.any() else np.nan
    res.append(dict(val=v, seed=sd, dSSE=float(d.sum()), all=f(np.ones(len(y), bool)), p2=f(p2), p2_high=f(p2 & hi), p2_norm=f(p2 & ~hi)))
    blocks.setdefault(v, []).append(pd.Series(d).groupby((G.farm + "_" + (G.day // 5).astype(str)).values).mean())
def run(v, sd, G, obs_all, fold_hidden, excl_fn, picks):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); new = G.sg.values.copy()
    for k, Gk in G.groupby("fold"):
        te = set(Gk.rec); hs = fold_hidden(k, te); vis = TRAIN & ~np.array([rid(i) in hs for i in range(N)]); key = (v, sd, k)
        model = None
        for r in te:
            g = G[G.rec == r]
            if not silent(r, g, hs, vis, key): continue
            an = ancestor(r, picks, hs)
            if an is None or DM[idx[an[0]]] < 1.0: continue
            if model is None: model = fit(obs_all, hs | excl_fn(te), vis, key, picks)
            o = hmm_anc(r, g, obs_all, model, vis, key, an); i0 = g.index.values
            fl.append(dict(val=v, seed=sd, rec=r, trueH=DM[idx[r]] >= 1.0, steps=len(an[1]) + 1, P=an[2], dsse=float(np.sum((o - G.y.values[i0]) ** 2) - np.sum((G.sg.values[i0] - G.y.values[i0]) ** 2))))
            new[i0] = o
    score(v, sd, G, new); print("done", v, sd, flush=True)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for sd, D in O.groupby("seed"):
    dsg = D[D.validator == "DIAG10"].groupby("rec").sg.mean()
    nb = lambda te: {r for r in dsg.index for t in te if FARM[idx[r]] == FARM[idx[t]] and abs(int(NUM[idx[r]]) - int(NUM[idx[t]])) <= 1}
    for v in ("DIAG10", "P2LOO", "EL1"):
        run(v, sd, D[D.validator == v].rename(columns={"sub_ec": "y"}), dsg, lambda k, te: set(te), (lambda te: set()) if v == "DIAG10" else nb, PMT)
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy()
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.rename(columns={"SG2": "sg"}); esg = G.groupby("rec").sg.mean()
    run("EV", sd, G, esg, lambda k, te: hidEV[int(k)] | set(te), lambda te: set(), PEVT)
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
FL = pd.DataFrame(fl)
if len(FL):
    print("\nfired records per validator-seed:", FL.groupby(["val", "seed"]).size().to_dict())
    print("fired: share truly high by validator:", FL.groupby("val").trueH.mean().round(3).to_dict())
    print(FL.groupby(["val", "rec"]).agg(trueH=("trueH", "first"), steps=("steps", "first"), P=("P", "mean"), dsse=("dsse", "mean")).round(3).to_string())
def bootP(v):
    s = pd.concat(blocks[v], axis=1).mean(axis=1).values; ix = rng.integers(0, len(s), (20000, len(s))); return float((s[ix].mean(1) >= 0).mean())
pw = {v: round(bootP(v), 4) for v in ("DIAG10", "P2LOO", "EL1", "EV")}; print("block bootstrap P(worse):", pw)
J = S[S.val != "DIAG10"]; ok = bool((J.dSSE < 0).all()) and pw["EV"] < .025 and pw["P2LOO"] < .025
print(f"judged cells improved {int((J.dSSE < 0).sum())}/{len(J)}"); print("(old all-cells rule, reference only):", "PASS" if ok else "FAIL")

print("\n================ DS9 judged under the EC specialty rule (pre-registered) ================")
def boot(v):
    s = pd.concat(blocks[v], axis=1).mean(axis=1).values; ix = np.random.default_rng(20261201).integers(0, len(s), (20000, len(s))); m = s[ix].mean(1)
    return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE < 0).all()); pP2, _ = boot("P2LOO")
harmed = FL[FL.val.isin(["P2LOO", "EL1"]) & (FL.dsse > 1e-6)] if len(FL) else FL
c1b = pP2 < .025 or len(harmed) == 0
print(f"(1) main P2LOO+EL1: all cells better {c1} ({int((main.dSSE < 0).sum())}/{len(main)}) | P2LOO P(worse) {pP2:.4f} | harmed records {len(harmed)} -> {'PASS' if c1 and c1b else 'FAIL'}"
      f"{'  [via small-sample route]' if c1 and pP2 >= .025 and len(harmed) == 0 else ''}")
ev = S[S.val == "EV"]; c2 = bool((ev.dSSE <= 0).all())
print(f"(2) linker robustness EV: seed dSSE {ev.dSSE.round(2).tolist()} -> {'PASS' if c2 else 'FAIL'}")
dg = S[S.val == "DIAG10"]; _, pb = boot("DIAG10"); c3 = not (dg.dSSE.mean() > 0 and pb < .025)
print(f"(3) harm guard DIAG10: mean dSSE {dg.dSSE.mean():+.2f}, share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("DS9 (specialty rule):", "PASS" if (c1 and c1b and c2 and c3) else "FAIL")
