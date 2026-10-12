# -*- coding: utf-8 -*-
"""BRK1 앞뒤 끼움 추정기 (2026-10-13 집 클로드; 사용자: '실제 온실에서 이런 상황일 때도 쓸 수 있게 만들어줘').  사전 고정(이 파일 커밋 후 실행).  EC.
상황 가정(실제 온실 일반형): 여러 동의 시간별 기록이 날짜·동 이름 없이 섞여 있고, EC 는 일부 날에만 측정(정답).  같은 기상 센서를 공유.
절차(대상 기록 b, h 시 행 — 규정: b 의 0..h 입력, 학습 기록 전부(뒷날 포함), 다른 ID 평가 입력·같은 ID 뒷번호 입력 안 씀):
  1) 날짜: b 의 0시 외기 4칸과 같은 보이는 정답 기록들의 학습 전용 달력 DMT3e(E0) 날짜 최빈값 d·구간 s (없으면 '날짜 모름').
  2) 같은 동 찾기 — 방향 별(전날 B / 다음날 F) 조건부 로짓 + '없음' 선택지(v15·NXL1 방식, 시간 묶음 H ∈ {0,9,15,21}):
     후보 = 보이는 정답 기록; 날짜 알면 같은 구간 d−3..d+1(B) / d−1..d+3(F), 모르면 지문 최근접 30.
     특징: cd1(날짜가 정확히 d∓1), cdev, cna, occ(후보가 이미 다른 확실 연결을 가짐), fp(운영 지문 0..H), fpr(순위), ind(실내 0..H),
           jmp(B 만: 후보 23시 → 대상 0시 실내·외기 자정 이음), + 없음 특징(nb, nfree, dunk, minfp, minind).
     학습 = 학습 기록 사이 확실 연결(truth cert) — 숨김 기록 낀 연결 제외, 진짜 짝을 숨긴 '없음' 그룹 가중 PI/(1−PI), PI .15.
  3) 값: A = B 짝 23시 EC, Bv = F 짝 0시 EC.  σ(확신 구간) = 확신 구간 [0,.5),[.5,.8),[.8,.95),[.95,1] 별 '보이는 학습 기록'의 같은 방식 연결 오차 RMS
     (그 칸에서 숨김 아닌 기록만, 대상 자신 제외 — 누수 없음); 모델 몫 σ_m = 그 칸 다른 폴드 기록의 하루 수준 RMS(sg 누적평균 vs 실제 하루 평균).
     결합: 앞뒤 둘 다 있고 |A − Bv| > 2.5·sqrt(σa² + σb²) 이면 '어긋남' → 둘 다 버림(확인 필요 표시).
     L = (cm/σ_m² + A/σa² + Bv/σb²) / (1/σ_m² + 1/σa² + 1/σb²)  (없는 항은 뺌);  pred_h = clip(sg_h + (L − cm_h)), cm_h = sg 0..h 누적평균.
     불확실성 σ_L = 1/sqrt(Σ 1/σ²).
안 1개(BRK).  판정(기준선 NONE·V7 각각): 12칸(rv1 DIAG10·P2LOO·EL1 × 47/1414/6464 + EV × 7/101/2024) 모두 2차 행 RMSE 개선 AND
  (a) DIAG10·EV 2차 기록 부트스트랩 P(악화) < .025 (시드 평균) 또는 (b) 작은 표본 경로: 어느 칸에서도 2차 기록 ΔSSE > 1e-6 없음.  → ADOPT.
서술: 전체 행, 방향별 연결 정확도(깨끗한 진실 대비), σ 보정(실제 하루 수준 오차가 σ_L 1·2배 안에 드는 비율), 확인 필요 표시 수.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, IN_, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261113)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TLc = TL[TL.cert]
LINKS = [(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if TRAIN[int(a)] and TRAIN[int(b)]]
PRED, SUCC = {}, {}
for a_, b_ in LINKS: PRED.setdefault(b_, []).append(a_); SUCC.setdefault(a_, []).append(b_)
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)       # training-only scaling
INZ = (IN_ - np.nanmean(IN_[TRAIN], (0, 1))) / (np.nanstd(IN_[TRAIN], (0, 1)) + 1e-9)
sdO = np.nanstd(X[TRAIN].reshape(-1, 4), 0)
HB = [0, 9, 15, 21]; FC = ["cd1", "cdev", "cna", "occ", "fp", "fpr", "ind", "jmp"]; FN = ["nb", "nfree", "dunk", "minfp", "minind"]; FA = FC + FN; PI = .15
def tdate(b, lab):
    vi = np.where(lab)[0]; vi = vi[vi != b]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); m = vi[eq]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; return d, pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
def build(b, lab, H, td, dirn):
    labb = lab.copy(); labb[b] = False
    fpall = np.nanmean(np.minimum(np.abs(FPA[:, :H + 1] - FPA[b, :H + 1][None]), 4), axis=(1, 2))
    if td is not None:
        d, s = td; lo, hi = (d - 3, d + 1) if dirn < 0 else (d - 1, d + 3)
        cand = np.where(labb & (SEGT == s) & (DAYT >= lo) & (DAYT <= hi))[0]
    else:
        vi = np.where(labb)[0]; cand = vi[np.argsort(fpall[vi])[:30]]
    cand = cand[np.isfinite(EC[cand, 23 if dirn < 0 else 0])]
    if len(cand) == 0: return None, None
    fp = fpall[cand]; ind = np.nanmean(np.minimum(np.abs(INZ[cand, :H + 1] - INZ[b, :H + 1][None]), 4), axis=(1, 2))
    if td is not None:
        same = SEGT[cand] == td[1]; tgt = td[0] + dirn
        cd1 = (same & (DAYT[cand] == tgt)).astype(float); cdev = np.where(same, np.minimum(np.abs(DAYT[cand] - tgt), 6) / 6, 1.0); cna = (~same).astype(float)
    else: cd1 = np.zeros(len(cand)); cdev = np.ones(len(cand)); cna = np.ones(len(cand))
    rel = SUCC if dirn < 0 else PRED
    occ = np.array([float(any(labb[p] for p in rel.get(int(c), []))) for c in cand])
    if dirn < 0:
        jmp = np.nanmean(np.minimum(np.abs(INZ[cand, 23] - INZ[b, 0][None]), 4), axis=1) + np.sum(np.abs(X[cand, 23] - X[b, 0][None]) / sdO, axis=1) / 4
    else: jmp = np.zeros(len(cand))
    fpr = np.argsort(np.argsort(fp)) / 10
    Z = np.zeros((len(cand) + 1, len(FA))); Z[:-1, :len(FC)] = np.c_[cd1, cdev, cna, occ, fp, fpr, np.nan_to_num(ind, nan=2.0), np.nan_to_num(jmp, nan=3.0)]
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
TG = {(dn, H): [] for dn in (-1, 1) for H in HB}
for a, b in LINKS:
    for dn, tgt, tru in ((-1, b, a), (1, a, b)):
        lab = TRAIN.copy(); lab[tgt] = False; td = tdate(tgt, lab)
        for H in HB:
            Z, cand = build(tgt, lab, H, td, dn)
            if Z is None or tru not in set(cand.tolist()): continue
            TG[(dn, H)].append((tgt, tru, Z, int(np.where(cand == tru)[0][0]), 1.0))
            lab2 = lab.copy(); lab2[tru] = False; Z2, c2 = build(tgt, lab2, H, td, dn)
            if Z2 is not None: TG[(dn, H)].append((tgt, tru, Z2, len(c2), PI / (1 - PI)))
print("training groups:", {k: len(v) for k, v in TG.items()}, flush=True)
WC = {}
def weights(key, hid):
    if key not in WC: WC[key] = {k: fit_wt([(z, y, wt) for a, b, z, y, wt in g if not hid[a] and not hid[b]]) for k, g in TG.items()}
    return WC[key]
def link(b, lab, W):
    """per-hour (24) arrays: back value/prob/pick, fwd value/prob/pick"""
    td = tdate(b, lab); out = {dn: (np.full(24, np.nan), np.zeros(24), [None] * 24) for dn in (-1, 1)}
    for dn in (-1, 1):
        for H in HB:
            Z, cand = build(b, lab, H, td, dn)
            if Z is None: continue
            s = Z @ W[(dn, H)]; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p[:-1])); hi = HB[HB.index(H) + 1] if H != HB[-1] else 24
            out[dn][0][H:hi] = EC[cand[k], 23 if dn < 0 else 0]; out[dn][1][H:hi] = p[k]; out[dn][2][H:hi] = [int(cand[k])] * (hi - H)
    return out
PB = [0, .5, .8, .95, 1.0001]; pbin = lambda p: int(np.searchsorted(PB, p, side="right") - 1)
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEANP = {}; CLEANN = {}
for a_, b_ in zip(TL.a1, TL.b):
    if D11.get(rid(int(b_)), np.nan) - D11.get(rid(int(a_)), np.nan) == 1: CLEANP[int(b_)] = int(a_); CLEANN.setdefault(int(a_), int(b_))
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
LK = {}   # cache link results per (cellkey, rec)
res, rows, diag = [], [], []
def run(Gv, y, sg, p2, hours, cm, recrows, v7val, hidset, cellkey, tag, sd):
    base_v7 = sg.copy(); brk = sg.copy()
    # sigma tables from visible (non-hidden) training records of this cell: link each visible record with its own fold's weights? -> use the
    # record's own link computed with lab = visible minus itself (cheap approximation: same W as the cell's hidden fold set of that record)
    links = {}
    for r, ix in recrows.items():
        b = idx[r]; hs = hidset(r); hm = np.array([rid(i) in hs for i in range(N)]); lab = TRAIN & ~hm
        key = (cellkey, frozenset(hs))
        if (tag, r, key[1]) not in LK: LK[(tag, r, key[1])] = link(b, lab, weights(key, hm))
        links[r] = LK[(tag, r, key[1])]
    # sigma by p-bin from all records of this cell EXCEPT those in the same fold as the target (fold-wise, no self/fold leakage)
    err = {dn: [] for dn in (-1, 1)}
    for r, ix in recrows.items():
        b = idx[r]; ycur = y[ix]
        for dn in (-1, 1):
            v, p, _ = links[r][dn]; tv = ycur[0] if dn < 0 else ycur[-1]
            for H in HB:
                if np.isfinite(v[H]): err[dn].append((r, pbin(p[H]), v[H] - tv))
    E = {dn: pd.DataFrame(err[dn], columns=["rec", "bin", "e"]) for dn in (-1, 1)}
    daylev = pd.DataFrame({"rec": Gv.rec.values, "cm": cm, "y": y, "h": hours}).groupby("rec").apply(lambda g: g.cm.values[-1] - g.y.mean())
    for r, ix in recrows.items():
        if r in v7val: base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_)
        hs = hidset(r); others = [q for q in recrows if q not in hs]
        sm = max(rm(daylev[others].values), .05)
        sig = {}
        for dn in (-1, 1):
            e = E[dn][E[dn].rec.isin(others)]
            sig[dn] = {k: max(rm(g.e.values), .02) if len(g) >= 5 else .5 for k, g in e.groupby("bin")}
        vb, pb_, kb = links[r][-1]; vf, pf_, kf = links[r][1]; n_flag = 0
        for j, k in enumerate(ix):
            h = int(hours[k]); num = cm[k] / sm ** 2; den = 1 / sm ** 2
            sa = sig[-1].get(pbin(pb_[h]), .5) if np.isfinite(vb[h]) else None; sf = sig[1].get(pbin(pf_[h]), .5) if np.isfinite(vf[h]) else None
            if sa is not None and sf is not None and abs(vb[h] - vf[h]) > 2.5 * np.sqrt(sa ** 2 + sf ** 2): sa = sf = None; n_flag += 1
            if sa is not None: num += vb[h] / sa ** 2; den += 1 / sa ** 2
            if sf is not None: num += vf[h] / sf ** 2; den += 1 / sf ** 2
            L = num / den; brk[k] = np.clip(sg[k] + (L - cm[k]), lo_, hi_)
            if h == 23: diag.append(dict(val=tag, seed=str(sd), rec=r, p2=NUM[idx[r]] >= 179, sigL=1 / np.sqrt(den), errL=L - y[ix].mean(),
                                          back_ok=(kb[h] == CLEANP.get(idx[r])) if kb[h] is not None and idx[r] in CLEANP else None,
                                          fwd_ok=(kf[h] == CLEANN.get(idx[r])) if kf[h] is not None and idx[r] in CLEANN else None, flag=n_flag > 0))
    B = {"NONE": sg, "V7": base_v7}
    for bn, b_ in B.items():
        res.append(dict(val=tag, seed=str(sd), base=bn, all_pct=100 * (rm(brk - y) / rm(b_ - y) - 1), p2_pct=100 * (rm((brk - y)[p2]) / rm((b_ - y)[p2]) - 1)))
        ds = pd.Series(((brk - y) ** 2 - (b_ - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
        for r_, d in ds.items(): rows.append(dict(val=tag, seed=str(sd), base=bn, rec=r_, dsse=d))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict(); hs = lambda r: hidden.get(fold_of[r], set())
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hs(r)}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, hs, v, v, sd); print("done", v, sd, flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidE = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; ea = Gv.EA2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}
    v7val = {r: float(ea[recrows[r][0]]) for r in set(Gv.rec[Gv.anchored.astype(bool)])}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, lambda r: hidE[fo[r]], "EV", "EV", sd); print("done EV", sd, flush=True)
S = pd.DataFrame(res); D = pd.DataFrame(rows); G = pd.DataFrame(diag); pd.set_option("display.width", 240)
os.makedirs(os.path.join(env.LOCAL, "brk1"), exist_ok=True); G.to_csv(os.path.join(env.LOCAL, "brk1", "brk1_diag.csv"), index=False); D.to_csv(os.path.join(env.LOCAL, "brk1", "brk1_dsse.csv"), index=False)
print("pass-2 RMSE % change:"); print(S.pivot_table(index="base", columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("all-rows % change:"); print(S.pivot_table(index="base", columns=["val", "seed"], values="all_pct").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
verdict = True
for bn in ("NONE", "V7"):
    cells = S[S.base == bn]; c1 = bool((cells.p2_pct < 0).all()); dd = D[D.base == bn]
    Pd = boot(dd[dd.val == "DIAG10"].groupby("rec").dsse.mean().values); Pe = boot(dd[dd.val == "EV"].groupby("rec").dsse.mean().values)
    harm = dd[dd.dsse > 1e-6].rec.nunique(); ok = c1 and ((Pd < .025 and Pe < .025) or harm == 0); verdict &= ok
    g = dd.groupby("rec").dsse.mean()
    print(f"\nBRK vs {bn}: cells better {int((cells.p2_pct < 0).sum())}/12 | DIAG10 P {Pd:.4f} EV P {Pe:.4f} | harmed records {harm} | changed {int((g.abs() > 1e-6).sum())} -> {'pass' if ok else 'fail'}")
    print("   top gains:", (-g).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-g).sort_values().head(5).round(3).to_dict())
print("BRK1:", "ADOPT" if verdict else "FAIL")
print("\nlink accuracy vs clean truth (hour 23 decision):", {k: round(float(G[k].dropna().astype(bool).mean()), 3) for k in ("back_ok", "fwd_ok")}, "n", {k: int(G[k].notna().sum()) for k in ("back_ok", "fwd_ok")})
G["z"] = (G.errL.abs() / G.sigL)
print("sigma calibration (share |err| within 1/2 sigma_L):", {nm: (round(float((g.z <= 1).mean()), 3), round(float((g.z <= 2).mean()), 3)) for nm, g in G.groupby("p2")}, "| flagged", int(G.flag.sum()))
