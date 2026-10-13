# -*- coding: utf-8 -*-
"""DS8 (2026-10-13 집 클로드; 사용자: 'DS8 해봐').  사전 고정(커밋 후 실행).  EC.  규정판 사슬(v15R 짝).
FC1(6.823): 가장 가까운 '정답 조상' 이 고EC 일 때만 HMM 이 상태를 이어받으면 가짜 고EC 연쇄가 끊김(진단 연결 상한: 진짜 14·가짜 1).
여기선 조상을 규정판 링커로 찾음 — v15R 짝(P2LOO·EL1·DIAG10: 거울 표 v15r15_mirror, EV: v15r15_ev) 1순위 prev0 를 최대 6단계,
  누적 짝 확신 P = Π p0 ≥ .3 인 동안 거슬러 처음 만나는 '보이는 정답' 기록(학습 & 그 폴드 숨김 아님) = 조상.
적용: 침묵 기록(EA2·TNA8·B5 미발동, DS7 과 같은 정의) 중 조상 하루 평균 EC ≥ 1.0 인 기록만.  아니면 그대로.
HMM: 조상에서 시작 α = P·[0,1] + (1−P)·사전, 중간(정답 없는) 날마다 α ← α·A 후 그날 예측 하루 평균·z 관측, 오늘 α·A 후 시각별 관측(0..h 누적평균 c_h, z_h).
  관측·회귀는 DS5 그대로(로지스틱 P(H|c) 우도비, z 베르누이, 상태별 직선 회귀), 매개변수는 DS7 과 같은 학습원.
  전이 A: 규정판 — 학습 기록 중 v15R 거울 짝(p0 ≥ .8, 짝도 정답 기록, 그 폴드 숨김 아님) 쌍에서 셈(진단 달력 안 씀).
검증기·판정(미리 정함, DS7 과 같음): 9칸(P2LOO·EL1·EV × 3시드) 모두 전체 행 ΔSSE < 0  AND  EV·P2LOO 블록 부트스트랩 P(악화) < .025.  DIAG10 서술.
서술: 발동 기록 수, 발동 기록의 실제 고EC 비율.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DS7SRC = open(os.path.join(HERE, "ds7_silent_zone_hmm_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ds7_silent_zone_hmm_v1.py")
exec(DS7SRC[:DS7SRC.index("def fit(obs, hs, vis, key):")])   # CR1 head (v7set, L7, TNA8, PM), zflag, silent, DM, CPREV(unused here)
__file__ = os.path.join(HERE, "ds8_ancestor_anchored_hmm_v1.py"); rng = np.random.default_rng(20261130)
PMT = {r: (pv, p0) for r, pv, p0 in zip(PM.index, PM.prev0, PM.p0)}
PEV = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv")); PEVT = {r: (pv, p0) for r, pv, p0 in zip(PEV.rec, PEV.prev0, PEV.p0)}
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
print(f"judged cells improved {int((J.dSSE < 0).sum())}/{len(J)}"); print("DS8 ancestor-anchored HMM:", "PASS" if ok else "FAIL")
