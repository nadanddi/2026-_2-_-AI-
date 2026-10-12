# -*- coding: utf-8 -*-
"""CAS1 (2026-10-13 집 클로드; 사용자: '정확하게 걸러낸 애들은 그대로 가고, 못 걸러낸 애들은 이전 최고 모델로').  사전 고정(이 파일 커밋 후 실행).  EC 만.
계단식(위에서 걸리면 그것, 아니면 다음): 1) v7 앞 앵커(13회차 그대로) → 2) v15(PI .15) 앞 1순위 p0 ≥ .8(보이는 정답·폴드 숨김 아님) 전 시각 앵커(A = 23시 EC)
  → 3) 앞뒤 합의(BIA1: v15 p0 ≥ .5 의 A 와 NXL1 다음날 p0 ≥ .8 의 B_h 가 ±.15 → (A+B_h)/2, 그 행만) → [CAS+ 만] 3b) NXL1 다음날 단독 p0 ≥ .8 → β .9·B_h
  → 4) 이전 최고 모델 예측(sg = 앵커 없음, 12회차 EC 에 해당).  식은 EA2(sg_h + (값 − sg 0..h 누적평균)).
부품 선택은 앞선 판정(6.765·6.766·6.773·6.775)의 2차 정확도를 보고 함(밝혀 둠). 앞쪽 짝: rv1 = G1(v15g1_mirror), EV = v15pi15_ev.
안 2개(본페로니 α .0125): CAS, CAS+.  기준(NONE·V7 각각 모두): 12칸 2차 개선; DIAG10 시드별·EV P < .0125; 상위 3 제외 ΔSSE < 0(DIAG10·EV); DIAG10 2차 바뀐 기록 ≥ 5.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, IN_, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261105)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TLc = TL[TL.cert]; PRED = {}
LINKS = [(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if TRAIN[int(a)] and TRAIN[int(b)]]
for a_, b_ in zip(TLc.a1, TLc.b): PRED.setdefault(int(b_), []).append(int(a_))
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
NXT_TRUE = {}
for a_, b_ in zip(TL.a1, TL.b):
    if D11.get(rid(int(b_)), np.nan) - D11.get(rid(int(a_)), np.nan) == 1: NXT_TRUE.setdefault(rid(int(a_)), set()).add(rid(int(b_)))
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
INZ = (IN_ - np.nanmean(IN_, (0, 1))) / (np.nanstd(IN_, (0, 1)) + 1e-9)
HB = [0, 9, 15, 21]; FC = ["cd1", "cdev", "cna", "occ", "fp", "fpr", "ind"]; FN = ["nb", "nfree", "dunk", "minfp", "minind"]; FA = FC + FN; PI = .15
def tdate(b, lab):
    vi = np.where(lab)[0]; vi = vi[vi != b]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); m = vi[eq]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; return d, pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
def build(b, lab, H, td):
    labb = lab.copy(); labb[b] = False
    fpall = np.nanmean(np.minimum(np.abs(FPA[:, :H + 1] - FPA[b, :H + 1][None]), 4), axis=(1, 2))
    if td is not None:
        d, s = td; cand = np.where(labb & (SEGT == s) & (DAYT >= d - 1) & (DAYT <= d + 3))[0]
    else:
        vi = np.where(labb)[0]; cand = vi[np.argsort(fpall[vi])[:30]]
    cand = cand[np.isfinite(EC[cand, 0])]
    if len(cand) == 0: return None, None
    fp = fpall[cand]; ind = np.nanmean(np.minimum(np.abs(INZ[cand, :H + 1] - INZ[b, :H + 1][None]), 4), axis=(1, 2))
    if td is not None:
        same = SEGT[cand] == td[1]; cd1 = (same & (DAYT[cand] == td[0] + 1)).astype(float); cdev = np.where(same, np.minimum(np.abs(DAYT[cand] - (td[0] + 1)), 6) / 6, 1.0); cna = (~same).astype(float)
    else: cd1 = np.zeros(len(cand)); cdev = np.ones(len(cand)); cna = np.ones(len(cand))
    occ = np.array([float(any(labb[p] for p in PRED.get(int(c), []))) for c in cand])
    fpr = np.argsort(np.argsort(fp)) / 10
    Z = np.zeros((len(cand) + 1, len(FA))); Z[:-1, :len(FC)] = np.c_[cd1, cdev, cna, occ, fp, fpr, np.nan_to_num(ind, nan=2.0)]
    Z[-1, len(FC):] = [1.0, min(float(((cd1 > 0) & (occ == 0)).sum()), 4), float(td is None), float(fp.min()), float(np.nanmin(ind)) if np.isfinite(ind).any() else 2.0]
    return Z, cand
def fit_wt(groups, l2=.5, iters=300, lr=.1):
    k = groups[0][0].shape[1]; w = np.zeros(k); m = np.zeros(k); v = np.zeros(k); sw = sum(g[2] for g in groups)
    for t in range(1, iters + 1):
        g = l2 * w.copy()
        for Xc, y, wt in groups:
            s_ = Xc @ w; s_ -= s_.max(); p = np.exp(s_); p /= p.sum(); g -= wt * (Xc[y] - p @ Xc) * len(groups) / sw
        g /= len(groups); m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
        w -= lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    return w
# precompute training groups per H on the full training set (hidden handled by dropping groups touching hidden records)
TG = {H: [] for H in HB}
for a, b in LINKS:
    lab = TRAIN.copy(); lab[a] = False; td = tdate(a, lab)
    for H in HB:
        Z, cand = build(a, lab, H, td)
        if Z is None or b not in set(cand.tolist()): continue
        TG[H].append((a, b, Z, int(np.where(cand == b)[0][0]), 1.0, None))
        lab2 = lab.copy(); lab2[b] = False; Z2, c2 = build(a, lab2, H, td)
        if Z2 is not None: TG[H].append((a, b, Z2, len(c2), PI / (1 - PI), None))
print("training groups per H:", {H: len(v) for H, v in TG.items()}, flush=True)
WC = {}
def weights(hid_key, hid_mask):
    if hid_key not in WC:
        WC[hid_key] = {H: fit_wt([(z, y, wt) for a, b, z, y, wt, _ in TG[H] if not hid_mask[a] and not hid_mask[b]]) for H in HB}
    return WC[hid_key]
def next_anchor(r, lab, W):
    b = idx[r]; out = np.full(24, np.nan); pk = [None] * 24; td = tdate(b, lab)
    for H in HB:
        Z, cand = build(b, lab, H, td)
        if Z is None: continue
        s = Z @ W[H]; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p[:-1]))
        if p[k] >= .8:
            hi = HB[HB.index(H) + 1] if H != HB[-1] else 24
            out[H:hi] = EC[cand[k], 0]; pk[H:hi] = [rid(cand[k])] * (hi - H)
    return out, pk
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
res = []; dsse = {}; evds = {}; acc = []
FWD = {"RV": pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15g1_mirror.csv")).set_index("rec"), "EV": pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_ev.csv")).set_index("rec")}
HIDSET = lambda r: set()
def run(Gv, y, sg, p2, hours, cm, recrows, v7val, labf, wf, tag, sd):
    base_v7 = sg.copy(); arms = {"CAS": sg.copy(), "CAS+": sg.copy()}
    FP = FWD.get(tag if tag == "EV" else "RV")
    for r, ix in recrows.items():
        if r in v7val:
            base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_)
            for a in arms: arms[a][ix] = base_v7[ix]
            continue
        A_ = None; p0 = 0.0
        if r in FP.index:
            fr = FP.loc[r]; pv = fr.get("prev0")
            if isinstance(pv, str) and bool(fr["lab0"]) and pv not in HIDSET(r) and np.isfinite(EC[idx[pv], 23]): A_ = EC[idx[pv], 23]; p0 = fr["p0"]
        if A_ is not None and p0 >= .8:
            for a in arms: arms[a][ix] = np.clip(sg[ix] + (A_ - cm[ix]), lo_, hi_)
            continue
        Bh, pk = next_anchor(r, labf(r), wf(r)); Bv = Bh[hours[ix]]; fin = np.isfinite(Bv)
        agree = fin & (A_ is not None and p0 >= .5) & (np.abs(Bv - (A_ if A_ is not None else 0)) <= .15)
        for a in arms:
            if agree.any(): arms[a][ix[agree]] = np.clip(sg[ix[agree]] + ((A_ + Bv[agree]) / 2 - cm[ix[agree]]), lo_, hi_)
        rest = fin & ~agree
        if rest.any(): arms["CAS+"][ix[rest]] = np.clip(sg[ix[rest]] + .9 * (Bv[rest] - cm[ix[rest]]), lo_, hi_)
    preds = {"NONE": sg, "V7": base_v7, **arms}
    for a in arms:
        for bn in ("NONE", "V7"):
            b_, n_ = preds[bn], preds[a]
            res.append(dict(val=tag, seed=str(sd), arm=a, base=bn, all_pct=100 * (rm(n_ - y) / rm(b_ - y) - 1), p2_pct=100 * (rm((n_ - y)[p2]) / rm((b_ - y)[p2]) - 1)))
            ds = pd.Series(((n_ - y) ** 2 - (b_ - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
            if tag == "DIAG10": dsse[(a, bn, sd)] = ds
            if tag == "EV": evds.setdefault((a, bn), []).append(ds)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict()
    HM = {k: np.array([rid(i) in vv for i in range(N)]) for k, vv in hidden.items()}
    labf = lambda r: TRAIN & ~HM[fold_of[r]]; wf = lambda r: weights((v, fold_of[r]), HM[fold_of[r]]); HIDSET = lambda r: hidden[fold_of[r]]
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hidden[fold_of[r]]}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, labf, wf, v, sd); print("done", v, sd, flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
HME = {k: np.array([rid(i) in vv for i in range(N)]) for k, vv in hid.items()}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; ea = Gv.EA2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}; labf = lambda r: TRAIN & ~HME[fo[r]]; wf = lambda r: weights(("EV", fo[r]), HME[fo[r]]); HIDSET = lambda r: hid[fo[r]]
    v7val = {r: float(ea[recrows[r][0]]) for r in set(Gv.rec[Gv.anchored.astype(bool)])}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, labf, wf, "EV", sd); print("done EV", sd, flush=True)
S = pd.DataFrame(res); pd.set_option("display.width", 240)
print("pass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("all-rows % change (seed mean):"); print(S.pivot_table(index=["arm", "base"], columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for a in ("CAS", "CAS+"):
    verdict = True
    for bn in ("NONE", "V7"):
        g = S[(S.arm == a) & (S.base == bn)]; c1 = bool((g.p2_pct < 0).all())
        Pd = [boot(dsse[(a, bn, sd)].values) for sd in (47, 1414, 6464)]; Dd = pd.concat([dsse[(a, bn, sd)] for sd in (47, 1414, 6464)], axis=1).mean(axis=1)
        De = pd.concat(evds[(a, bn)], axis=1).mean(axis=1); Pe = boot(De.values)
        t3d, t3e = Dd.sort_values().iloc[3:].sum(), De.sort_values().iloc[3:].sum(); nch = int((Dd.abs() > 1e-6).sum())
        ok = c1 and all(p < .0125 for p in Pd) and Pe < .0125 and t3d < 0 and t3e < 0 and nch >= 5; verdict &= ok
        print(f"{a} vs {bn}: all p2 better {c1} | DIAG10 P {np.round(Pd, 4).tolist()} | EV P {Pe:.4f} | drop-top3 DIAG10 {t3d:+.3f} EV {t3e:+.3f} | changed p2 DIAG10 {nch} EV {int((De.abs() > 1e-6).sum())} -> {'pass' if ok else 'fail'}")
        print("   top gains DIAG10:", (-Dd).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-Dd).sort_values().head(3).round(3).to_dict())
        print("   top gains EV:", (-De).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-De).sort_values().head(3).round(3).to_dict())
    print(f"CAS1 {a}:", "ADOPT" if verdict else "FAIL")
