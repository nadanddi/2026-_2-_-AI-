# -*- coding: utf-8 -*-
"""DS7 (2026-10-13 집 클로드; 사용자: 'DS7 진행').  사전 고정(커밋 후 실행).  EC.  상한 관문(사슬 = 진단용 깨끗한 연결, DS1~5 와 같음).
DS1Y(6.819): HMM 이득은 세 부품(EA2·TNA8·B5) 침묵 구간에, 손해 상당수는 B5 구간에 → 조합: 현재 구성 위에 '침묵 기록에만' DS5(로지스틱 보정 HMM, 사슬 있음).
침묵 기록에서는 현재 구성 = 원 예측 sg 이므로 조합 변화 = 침묵 기록의 HMM 변화(ΔSSE 는 정확, % 는 sg 기준 RMSE 대비로 표기).
부품 발동(각 검증기·폴드의 숨김 기준): EA2 = v7 확신 짝(p ≥ .8·정답·dT ≤ 1.5)의 짝이 숨김 아님; TNA8 = v15R 거울 짝(정답, p0 ≥ .8 또는 HP2 일치) 숨김 아님;
  B5 = 어떤 h 에서 sg 0..h 누적평균 ≥ 1.0 & 같은 날 학습 쌍둥이(0..h, 숨김 아님) 하루 평균 ≥ 1.2.
HMM(DS5 그대로): 상태 N/H(하루 평균 ≥ 1.0), 전이 = 숨김 아닌 학습 기록의 깨끗한 연속 쌍, 사전 = H 비율, c 우도비 = 로지스틱 P(H|c) 오즈/사전 오즈, z 베르누이,
  상태별 직선 회귀, 사슬 최대 3일(이전 날은 예측 하루 평균·z 만 관측).
검증기: P2LOO·EL1(ec_rv1 CUR, 시드 47/1414/6464; 매개변수·이전 날 관측은 같은 시드 DIAG10 OOF 예측, 학습 적합에서 시험 기록·같은 농장 ±1 제외),
  EV(v2_ev_ec_rows BASE SG2, 시드 7/101/2024; 매개변수·관측은 같은 셀 EV 예측, 적합은 그 폴드 val∪del 제외).  DIAG10 은 서술만(구간 분할을 DIAG10 에서 보고 정함).
판정(미리 정함): 9칸(P2LOO·EL1·EV × 3시드) 모두 전체 행 ΔSSE < 0  AND  EV·P2LOO 각각 농장×5일 블록 부트스트랩(시드 평균) P(악화) < .025.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CR1SRC = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "cr1_chain_gru_temp_v1.py")
exec(CR1SRC[:CR1SRC.index("V = [\"in_temp\", \"in_hum\"")])   # v7set, L7, TNA8, idx, TRAIN, FARM, NUM, N, rid
__file__ = os.path.join(HERE, "ds7_silent_zone_hmm_v1.py")
from scipy.stats import norm  # noqa: F401,E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, S3  # noqa: E402
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN]); rng = np.random.default_rng(20261128)
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
XR = np.round(X, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CPREV = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
V7P = {r: p for r, p in zip(L7.rec, L7.prev) if r in v7set}
_zc = {}
def zflag(i, h, vis, key):
    k = (i, h, key)
    if k not in _zc:
        c = np.where(vis)[0]; c = c[c != i]; t = c[np.all(np.abs(XR[c, :h + 1] - XR[i, :h + 1]) < 1e-6, axis=(1, 2))]; t = t[np.isfinite(DM[t])]
        _zc[k] = float((DM[t] >= 1.2).any())
    return _zc[k]
def silent(r, g, hs, vis, key):
    if r in V7P and V7P[r] not in hs: return False
    if r in TNA8 and TNA8[r] not in hs: return False
    return not any(zflag(idx[r], int(h), vis, key) for h, c in zip(g.hour, g.cm) if c >= 1.0)
def fit(obs, hs, vis, key):
    """obs: rec -> predicted day mean (all records with a prediction); fit on labelled training records not in hs"""
    tr = [r for r in obs.index if r not in hs and TRAIN[idx[r]] and np.isfinite(DM[idx[r]])]; st = {r: DM[idx[r]] >= 1.0 for r in tr}
    pr = [(st[CPREV[r]], st[r]) for r in tr if r in CPREV and CPREV[r] in st]
    A = np.array([[sum(1 for a, b in pr if a == s0 and b == s1) + 1 for s1 in (0, 1)] for s0 in (0, 1)], float); A /= A.sum(1, keepdims=True)
    cs = np.array([obs[r] for r in tr]); ss = np.array([st[r] for r in tr]).astype(int); ys = np.array([DM[idx[r]] for r in tr])
    zs = np.array([zflag(idx[r], 23, vis, key) for r in tr]); prior = np.array([1 - ss.mean(), ss.mean()])
    pi = [(zs[ss == s].sum() + 1) / ((ss == s).sum() + 2) for s in (0, 1)]; reg = [np.polyfit(cs[ss == s], ys[ss == s], 1) for s in (0, 1)]
    LR = LogisticRegression().fit(cs.reshape(-1, 1), ss); po = prior[1] / prior[0]
    def lik(c, z):
        p = float(np.clip(LR.predict_proba([[c]])[0, 1], 1e-4, 1 - 1e-4)); return np.array([(pi[0] if z else 1 - pi[0]), (p / (1 - p)) / po * (pi[1] if z else 1 - pi[1])])
    return A, prior, lik, reg
def hmm_rec(r, g, obs, model, vis, key):
    A, prior, lik, reg = model; chain = []; cur = r
    for _ in range(3):
        pv = CPREV.get(cur)
        if pv is None or pv not in obs.index: break
        chain.append(pv); cur = pv
    alpha = prior.copy()
    for pv in reversed(chain): alpha = alpha @ A; alpha = alpha * lik(obs[pv], zflag(idx[pv], 23, vis, key)); alpha /= alpha.sum()
    a0 = (alpha @ A) if chain else prior; out = []
    for h, c, s in zip(g.hour, g.cm, g.sg):
        post = a0 * lik(c, zflag(idx[r], int(h), vis, key)); post /= post.sum()
        out.append(np.clip(s + sum(post[k] * np.polyval(reg[k], c) for k in (0, 1)) - c, lo_, hi_))
    return np.array(out)
res, blocks, zl = [], {}, []
def score(v, sd, G, new):
    y = G.y.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    d = (new - y) ** 2 - (sg - y) ** 2; f = lambda m: 100 * (rm((new - y)[m]) / rm((sg - y)[m]) - 1) if m.any() else np.nan
    res.append(dict(val=v, seed=sd, dSSE=float(d.sum()), all=f(np.ones(len(y), bool)), p2=f(p2), p2_high=f(p2 & hi), p2_norm=f(p2 & ~hi)))
    blocks.setdefault(v, []).append(pd.Series(d).groupby((G.farm + "_" + (G.day // 5).astype(str)).values).mean())
def run(v, sd, G, obs_all, fold_hidden, excl_fn):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); new = G.sg.values.copy()
    for k, Gk in G.groupby("fold"):
        te = set(Gk.rec); hs = fold_hidden(k, te); vis = TRAIN & ~np.array([rid(i) in hs for i in range(N)]); key = (v, sd, k)
        model = fit(obs_all, hs | excl_fn(te), vis, key)
        for r, g in Gk.groupby("rec"):
            g = G[G.rec == r]; sil = silent(r, g, hs, vis, key); zl.append(dict(val=v, seed=sd, rec=r, silent=sil))
            if sil: new[g.index.values] = hmm_rec(r, g, obs_all, model, vis, key)
    score(v, sd, G, new); print("done", v, sd, flush=True)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for sd, D in O.groupby("seed"):
    Dg = D[D.validator == "DIAG10"]; dsg = Dg.groupby("rec").sg.mean()
    nb = lambda te: {r for r in dsg.index for t in te if FARM[idx[r]] == FARM[idx[t]] and abs(int(NUM[idx[r]]) - int(NUM[idx[t]])) <= 1}
    for v in ("DIAG10", "P2LOO", "EL1"):
        G = D[D.validator == v].rename(columns={"sub_ec": "y"})
        run(v, sd, G, dsg, lambda k, te: set(te), (lambda te: set()) if v == "DIAG10" else nb)
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy()
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.rename(columns={"SG2": "sg"}); esg = G.groupby("rec").sg.mean()
    run("EV", sd, G, esg, lambda k, te: hidEV[int(k)] | set(te), lambda te: set())
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
Z = pd.DataFrame(zl); print("silent share by validator:", Z.groupby("val").silent.mean().round(3).to_dict())
def bootP(v):
    s = pd.concat(blocks[v], axis=1).mean(axis=1).values; ix = rng.integers(0, len(s), (20000, len(s))); return float((s[ix].mean(1) >= 0).mean())
pw = {v: round(bootP(v), 4) for v in ("DIAG10", "P2LOO", "EL1", "EV")}; print("block bootstrap P(worse):", pw)
J = S[S.val != "DIAG10"]; ok = bool((J.dSSE < 0).all()) and pw["EV"] < .025 and pw["P2LOO"] < .025
print(f"judged cells improved {int((J.dSSE < 0).sum())}/{len(J)}"); print("DS7 silent-zone HMM upper-bound gate:", "PASS -> build rule-safe chain" if ok else "FAIL")
