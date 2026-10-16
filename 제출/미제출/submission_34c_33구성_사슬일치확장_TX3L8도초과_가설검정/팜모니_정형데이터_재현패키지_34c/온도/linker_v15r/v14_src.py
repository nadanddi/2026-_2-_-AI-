# -*- coding: utf-8 -*-
"""V14 (2026-10-13 집 클로드; 사용자: '날짜 투표 실험해봐. 대신 후보 결정 모델은 온도가 아닌 EC 특성에 맞게').
= v13(E0 달력 DMT3e) 링커 구조 + 두 가지 변경.
(1) 대상 날짜 투표(규정: 대상 자기 0시 입력 + 학습/보이는 기록만):
    ⓐ 0시 외기 4칸이 똑같은 보이는 정답 기록의 날짜(최빈)·구간
    ⓑ 같은 기하의 v13E0 링커 짝이 보이는 정답 기록이면 그 날짜 + 1(같은 구간)
    ⓒ 보이는 정답 기록 중 외기 자정 연속 비용(v7 oj 식) 최소인 기록의 날짜 + 1
    규칙: ⓐ 있으면 ⓐ, 단 ⓑ·ⓒ 가 함께 ⓐ 와 다른 같은 날짜면 '모름'; ⓐ 없으면 ⓑ = ⓒ 일 때만 그 날짜, 아니면 '모름'.
(2) EC 맞춤 학습: L = v12/v13 과 같은 학습(정답 = 진짜 전날), M = 다중 정답(진짜 전날 ∪ 23시 EC 가 대상 실제 0시 EC 와 ±.05 안인 보이는 정답 후보).
출력(local/linker_v10/v14_{mirror,ev,eval}.csv): rec, fold, date_src(a/bc/none/veto), 그리고 L·M 각각 상위 6 후보(prev_k, p_k, lab_k) — 앵커 값·EC 합의 확신도는
  평가 스크립트에서 폴드 숨김 반영해 계산(ecd1).  링커 단독 서술: 날짜 투표 정확도(깨끗한 정답 기준 날짜 = 진짜 전날 DMT3e 날짜 + 1)와 포괄.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
from feat4 import EC, sdO
H_ = os.path.join(env.ROOT, "집", "클로드", "research")
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
DT = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DD_ = dict(zip(DT.rec, DT.day)); SG_ = dict(zip(DT.rec, DT.seg))
DAYT = np.array([DD_.get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([SG_.get(rid(i), np.nan) for i in range(N)])
idx = {rid(i): i for i in range(N)}
W4 = np.array([1, 1, .3, .5]); ojf = lambda a, b: float(np.sum(W4 * np.abs(X[a, 23] + .5 * (X[a, 23] - X[a, 22]) - X[b, 0]) / sdO))
F12 = ["oj"] + [f"v{j}" for j in range(8)] + ["fp", "excl", "cd1", "cdev", "cna"]
OUT = os.path.join(env.LOCAL, "linker_v10")
V13 = {k: pd.read_csv(os.path.join(OUT, f"v13E0_picks_{k}.csv")) for k in ("mirror", "ev", "eval")}
GEOPICK = {}  # (geometry, rec) -> v13 prev
for k, D in V13.items():
    for r, pv in zip(D.rec, D.prev): GEOPICK[(k, r)] = pv
CUR_GEO = ["mirror"]
def date_twin(i, lab):
    vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[i, 0]) < 1e-6, axis=1); m = vi[eq]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; s = pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]; return (d, s)
def vote_date(b, lab):
    a_ = date_twin(b, lab)
    pv = GEOPICK.get((CUR_GEO[0], rid(b))); bb = None
    if pv is not None and lab[idx[pv]] and np.isfinite(DAYT[idx[pv]]): bb = (DAYT[idx[pv]] + 1, SEGT[idx[pv]])
    vi = np.where(lab & np.isfinite(DAYT))[0]; vi = vi[vi != b]; cc = None
    if len(vi): a2 = vi[int(np.argmin([ojf(a, b) for a in vi]))]; cc = (DAYT[a2] + 1, SEGT[a2])
    if a_ is not None:
        if bb is not None and bb == cc and bb != a_: return None, "veto"
        return a_, "a"
    if bb is not None and bb == cc: return bb, "bc"
    return None, "none"
def date_of_cand(i, lab):
    if lab[i] and np.isfinite(DAYT[i]): return (DAYT[i], SEGT[i])
    return date_twin(i, lab)
def feats14(b, lab, removed):
    D, _ = features10(b, 0, lab, removed=removed)
    if len(D) == 0: return D, "none"
    a = D.a.values.astype(int)
    D["fp"] = np.nan_to_num(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0][None]), 4), axis=1), nan=1.0)
    labb = lab & (np.arange(N) != b); tb, src = vote_date(b, labb)
    dd = [date_of_cand(x, lab) for x in a]
    da = np.array([x[0] if x else np.nan for x in dd]); sa = np.array([x[1] if x else np.nan for x in dd])
    ok = np.isfinite(da) & (tb is not None) & (sa == (tb[1] if tb else -9))
    db = tb[0] if tb else np.nan
    D["cd1"] = np.where(ok, (da == db - 1).astype(float), 0.0); D["cdev"] = np.where(ok, np.minimum(np.abs(da - (db - 1)), 6) / 6, 0.0); D["cna"] = (~ok).astype(float)
    return D, src
def fit_mp(groups, l2=.5, iters=300, lr=.1):
    k = groups[0][0].shape[1]; w = np.zeros(k); m = np.zeros(k); v = np.zeros(k)
    for t in range(1, iters + 1):
        g = l2 * w.copy()
        for Xc, pos in groups:
            s = Xc @ w; s -= s.max(); p = np.exp(s); p /= p.sum(); q = p[pos] / p[pos].sum()
            g -= (q @ Xc[pos] - p @ Xc)
        g /= len(groups); m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
        w -= lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    return w
TRG = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    df, _ = feats14(b, lab, TEST_IDX)
    if len(df) == 0: continue
    ca = df.a.values.astype(int); y = np.where(ca == a)[0]
    if not len(y): continue
    ecok = np.array([lab[c] and np.isfinite(EC[c, 23]) and np.isfinite(EC[b, 0]) and abs(EC[c, 23] - EC[b, 0]) <= .05 for c in ca])
    pos = np.unique(np.concatenate([y, np.where(ecok)[0]]))
    TRG.append((b, a, df[F12].values.astype(float), y[0], pos))
print("training groups", len(TRG), "| mean positives (M)", round(np.mean([len(t[4]) for t in TRG]), 2), flush=True)
def fitw(touch):
    G = [t for t in TRG if not touch[t[0]] and not touch[t[1]]]
    return fit_clogit([(t[2], t[3]) for t in G], l2=0.5, iters=300), fit_mp([(t[2], t[4]) for t in G])
def pick(b, lab, removed, wL, wM):
    D, src = feats14(b, lab, removed)
    if len(D) == 0: return None
    a = D.a.values.astype(int); Z = D[F12].values.astype(float); r = dict(rec=rid(b), date_src=src)
    for nm, w in (("L", wL), ("M", wM)):
        s = Z @ w; p = np.exp(s - s.max()); p /= p.sum(); o = np.argsort(-p)[:6]
        for k, j in enumerate(o):
            r[f"{nm}_prev{k}"] = rid(a[j]); r[f"{nm}_p{k}"] = float(p[j]); r[f"{nm}_lab{k}"] = bool(lab[a[j]])
    return r
# mirror
CUR_GEO[0] = "mirror"; rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist())
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        wL, wM = fitw(touch); lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if TRAIN[b]:
                r = pick(b, lab, rem | oth, wL, wM)
                if r: rows.append(r)
M_ = pd.DataFrame(rows); M_.to_csv(os.path.join(OUT, "v14_mirror.csv"), index=False); print("mirror", len(M_), "| date_src", M_.date_src.value_counts().to_dict(), flush=True)
# EV
CUR_GEO[0] = "ev"; FO = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); rows = []
for k, Gf in FO.groupby("fold"):
    val = [idx[r] for r in Gf.rec[Gf.role == "val"]]
    hidden = set(Gf.rec[Gf.role.isin(["val", "del"])]); hid_i = np.array([rid(i) in hidden for i in range(N)])
    dele = {i for i in range(N) if rid(i) in set(Gf.rec[Gf.role == "del"])}
    wL, wM = fitw(hid_i); lab = TRAIN & ~hid_i
    for b in val:
        r = pick(b, lab, dele, wL, wM)
        if r: r["fold"] = k; rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "v14_ev.csv"), index=False); print("EV", len(rows), flush=True)
# eval
CUR_GEO[0] = "eval"; wL, wM = fitw(np.zeros(N, bool)); rows = []
for b in [i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13", "F47")]:
    r = pick(b, TRAIN.copy(), set(), wL, wM)
    if r: rows.append(r)
E = pd.DataFrame(rows); E.to_csv(os.path.join(OUT, "v14_eval.csv"), index=False); print("eval", len(E), "| date_src", E.date_src.value_counts().to_dict())
print("weights L:", dict(zip(F12, np.round(wL, 2)))); print("weights M:", dict(zip(F12, np.round(wM, 2))))
