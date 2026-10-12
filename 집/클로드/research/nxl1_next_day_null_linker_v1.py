# -*- coding: utf-8 -*-
"""NXL1 (2026-10-13 집 클로드; 사용자: '다음날 앵커에 없음 선택지 링커 붙여 해봐').  사전 고정(이 파일 커밋 후 실행).  EC 만.
근거: NXA1(6.772) — 2차 다음날 짝 13/15 정확이나 소거만으로는 1차 오짝 많음(.61), EV F47_209 오짝 −9.5. v15 '없음' 선택지가 앞쪽 오연결을 막음(6.765).
다음날 링커(조건부 로짓 + '없음' 후보), 시간 구간 H ∈ {0, 9, 15, 21} 마다 따로 학습; h 시 행은 H ≤ h 인 가장 늦은 구간 모델(대상 0..H 입력만 사용).
  후보 = 보이는 정답 기록 중: 대상 날짜 (d, s) 를 알면 같은 구간 날짜 d−1..d+3, 모르면 0..H 지문 최근접 30.
  날짜 (d, s) = 0시 외기 4칸이 똑같은 보이는 정답 기록들의 DMT3e 날짜(최빈)·구간.
  후보 특징: cd1(날짜 = d+1 & 같은 구간), cdev(|날짜 − (d+1)|/6, 같은 구간일 때, 아니면 1), cna(날짜 모름/구간 다름), occ(그 후보의 확실 연결 전날이 다른 보이는 정답 기록),
    fp(0..H 운영 지문 거리 8변수), fpr(후보 중 fp 순위/10), ind(0..H 실내 3변수 거리).
  '없음' 특징: bias, n_free(cd1 & ~occ 후보 수, 4 자름), date_unk, min_fp, min_ind.
  학습 사례: 학습 확실 연결(truth cert, 두 기록 모두 학습) a→b 에서 대상 a, 정답 b (보이는 정답 = 학습 − 폴드 숨김 − a 자신); + b 를 숨긴 '없음' 사례(가중 .15/.85). 폴드마다 숨김 기록이 낀 연결은 학습에서 뺌.
결정: 1순위 실제 후보 p0 ≥ .8 이면 그 행 pred_h = clip(sg_h + .9·(B − sg 0..h 누적평균)), B = 그 후보 0시 EC. v7 앞 앵커 기록은 v7 그대로.
판정(1안, α .025, NONE·V7 각각 모두): NXA1 과 같음 — 12칸 2차 개선, DIAG10 시드별·EV P < .025, 상위 3 제외 ΔSSE < 0(DIAG10·EV), DIAG10 2차 바뀐 기록 ≥ 5.
서술: 다음날 앵커 기록 수·정확(진단 연결 다음날 기준, 서술만), NXA1 대비.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, IN_, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261103)
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
def run(Gv, y, sg, p2, hours, cm, recrows, v7val, labf, wf, tag, sd):
    base_v7 = sg.copy(); arm = sg.copy()
    for r, ix in recrows.items():
        if r in v7val:
            base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_); arm[ix] = base_v7[ix]; continue
        Bh, pk = next_anchor(r, labf(r), wf(r)); Bv = Bh[hours[ix]]; mk = np.isfinite(Bv)
        if mk.any():
            arm[ix[mk]] = np.clip(sg[ix[mk]] + .9 * (Bv[mk] - cm[ix[mk]]), lo_, hi_)
            if tag in ("DIAG10", "EV") and str(sd) in ("47", "7"):
                ch = {p for p in pk if p}; acc.append(dict(val=tag, rec=r, picks=",".join(sorted(ch)), true_next=",".join(sorted(NXT_TRUE.get(r, []))), ok=bool(ch & NXT_TRUE.get(r, set())), n_hours=int(mk.sum()), p2=int(r[4:7]) >= 179))
    preds = {"NONE": sg, "V7": base_v7, "NXL": arm}
    for bn in ("NONE", "V7"):
        b_, n_ = preds[bn], preds["NXL"]
        res.append(dict(val=tag, seed=str(sd), base=bn, all_pct=100 * (rm(n_ - y) / rm(b_ - y) - 1), p2_pct=100 * (rm((n_ - y)[p2]) / rm((b_ - y)[p2]) - 1)))
        ds = pd.Series(((n_ - y) ** 2 - (b_ - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
        if tag == "DIAG10": dsse[(bn, sd)] = ds
        if tag == "EV": evds.setdefault(bn, []).append(ds)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict()
    HM = {k: np.array([rid(i) in vv for i in range(N)]) for k, vv in hidden.items()}
    labf = lambda r: TRAIN & ~HM[fold_of[r]]; wf = lambda r: weights((v, fold_of[r]), HM[fold_of[r]])
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hidden[fold_of[r]]}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, labf, wf, v, sd); print("done", v, sd, flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
HME = {k: np.array([rid(i) in vv for i in range(N)]) for k, vv in hid.items()}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; ea = Gv.EA2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}; labf = lambda r: TRAIN & ~HME[fo[r]]; wf = lambda r: weights(("EV", fo[r]), HME[fo[r]])
    v7val = {r: float(ea[recrows[r][0]]) for r in set(Gv.rec[Gv.anchored.astype(bool)])}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, labf, wf, "EV", sd); print("done EV", sd, flush=True)
A = pd.DataFrame(acc); pd.set_option("display.width", 240)
for t in ("DIAG10", "EV"):
    S_ = A[A.val == t] if len(A) else A
    if len(S_): print(f"\n{t} next-day anchors: {len(S_)} records (pass-2 {S_.p2.sum()}) | pick = true next {S_.ok.mean():.3f} | pass-2 {S_[S_.p2].ok.mean() if S_.p2.any() else float('nan'):.3f}"); print(S_[S_.p2 | ~S_.ok].to_string(index=False))
S = pd.DataFrame(res)
print("\npass-2 RMSE % change:"); print(S.pivot_table(index="base", columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("\nall-rows % change (seed mean):"); print(S.pivot_table(index="base", columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
verdict = True
for bn in ("NONE", "V7"):
    g = S[S.base == bn]; c1 = bool((g.p2_pct < 0).all())
    Pd = [boot(dsse[(bn, sd)].values) for sd in (47, 1414, 6464)]; Dd = pd.concat([dsse[(bn, sd)] for sd in (47, 1414, 6464)], axis=1).mean(axis=1)
    De = pd.concat(evds[bn], axis=1).mean(axis=1); Pe = boot(De.values)
    t3d, t3e = Dd.sort_values().iloc[3:].sum(), De.sort_values().iloc[3:].sum(); nch = int((Dd.abs() > 1e-6).sum())
    ok = c1 and all(p < .025 for p in Pd) and Pe < .025 and t3d < 0 and t3e < 0 and nch >= 5; verdict &= ok
    print(f"\nNXL vs {bn}: all p2 better {c1} | DIAG10 P {np.round(Pd, 4).tolist()} | EV P {Pe:.4f} | drop-top3 DIAG10 {t3d:+.3f} EV {t3e:+.3f} | changed p2 DIAG10 {nch} EV {int((De.abs() > 1e-6).sum())} -> {'pass' if ok else 'fail'}")
    print("   top gains DIAG10:", (-Dd).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-Dd).sort_values().head(3).round(3).to_dict())
    print("   top gains EV:", (-De).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-De).sort_values().head(3).round(3).to_dict())
print("NXL1:", "ADOPT" if verdict else "FAIL")
