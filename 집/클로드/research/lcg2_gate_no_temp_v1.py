# -*- coding: utf-8 -*-
"""LCG2 (2026-10-16 집 클로드; 사용자 '추운 거 빼고 실내 차이랑 확신만 넣은 게이트 해봐').  사전 고정(커밋 후 실행).  온도.
LCG1 방법 A 에서 0시 실내온도 특징만 뺌(평가 추운 기록에서 외삽해 게이트가 세지는 문제, 6.990): f = [시각, |A-BIAS-x0|, 연결 확신 p, 0시/23시 실내 차 dT].
나머지 LCG1 A 와 같음(후보 p >= .3, dT <= 3, PCH1 배치 사슬 행 적합, 판정 EXT10·EXT12·DIAG10 거울 표, 기준 M1).
판정(단일 안, alpha .025): 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .025.
한계(미리 적음): LCG1 결과를 보고 만든 안이고 판정 배치도 같음 - 독립 판정 아님(참고 수준).  서술: 평가 넓은 사슬 기록에서의 게이트 가중.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u lcg2_gate_no_temp_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "lcg2_gate_no_temp_v1.py")
import common, importlib.util as _iu
_s = _iu.spec_from_file_location("cv8", os.path.join(HERE, "cv8_cold_validator_v1.py")); C8 = _iu.module_from_spec(_s); _s.loader.exec_module(C8)
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
_wd = W.worlds(); WB = np.asarray(_wd[4], float); ELF = dict(_wd[7])["EL1"]; CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
FE = ["in_temp", "ph_in_temp_1", "ph_in_temp_3", "ph_in_temp_8", "ph_in_temp_24", "out_temp", "ph_rad_2", "act_heating", "act_thermal", "hr_sin", "hr_cos"]
L0 = lab.copy()
if "hr_sin" not in L0: L0["hr_sin"], L0["hr_cos"] = np.sin(2 * np.pi * L0.hour / 24), np.cos(2 * np.pi * L0.hour / 24)
es = lambda T: .6108 * np.exp(17.27 * T / (T + 237.3)); e_in = es(L0.in_temp) * L0.in_hum / 100
L0["ah"] = 2167 * e_in / (L0.in_temp + 273.15); w_ = .622 * e_in / (101.325 - e_in); L0["enth"] = 1.006 * L0.in_temp + w_ * (2501 + 1.86 * L0.in_temp)
L0["dewgap"] = L0.in_temp - (237.3 * np.log(e_in.clip(1e-3) / .6108) / (17.27 - np.log(e_in.clip(1e-3) / .6108)))
PSY = ["ah", "enth", "dewgap"]
def spl_fit_predict(tm, vm, feats):
    Xt = L0.loc[tm, feats].to_numpy(float); Xv = L0.loc[vm, feats].to_numpy(float); med = np.nanmedian(Xt, 0)
    Xt = np.where(np.isnan(Xt), med, Xt); Xv = np.where(np.isnan(Xv), med, Xv)
    m = make_pipeline(SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), StandardScaler(), Ridge(alpha=10.0)); m.fit(Xt, L0.sub_temp.values[tm], ridge__sample_weight=WB[tm])
    return pd.Series(m.predict(Xv), index=L0.row_id.values[vm])
def gate(M, lo_g, lo_h):
    g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - lo_g) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
    mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - lo_h) / 2, 0, 1)); return g_ * h_
def tx3(M, key, c, T0, e0):
    cm = np.nan_to_num(np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, M.hour.values)]), nan=0)
    return c * np.clip((cm - e0) / .4, 0, 1) * np.nan_to_num(np.clip((T0 - M.in_temp.values) / 2, 0, 1), nan=0.0)
from scipy.optimize import minimize
XR = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"]).rename(columns={"in_temp": "it_raw"}); XR["rec"] = XR.row_id.str[:7]; XR["h"] = XR.row_id.str[8:10].astype(int)
IT0 = XR[XR.h == 0].set_index("rec").it_raw.to_dict()
S1, TAU = .9, 8.0
def elig(LK, r, recs_pos, h, FOLDOF, labelled_ok, pmin, dmax):
    if r not in LK or not isinstance(LK[r][0], str) or LK[r][1] < pmin: return None
    p, q = LK[r]
    if p not in IDX: return None
    dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
    if not np.isfinite(dt) or dt > dmax: return None
    if labelled_ok(r, p): return ("K8", p, q, dt) if (q >= .8 and dt <= 1.5 and np.isfinite(ST[IDX[p], 23])) else None
    if p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in recs_pos and FOLDOF.get(p) == FOLDOF.get(r) and (h[recs_pos[p]] == 23).any(): return ("CH", p, q, dt)
    return None
def make_run(PL, pos, h, TXo, order):
    def run(x, gate, collect=False):
        out = x.copy(); rows = []
        for r, (kind, p, q, dt) in PL.items():
            if kind == "K8":
                ix = pos[r]; x0 = x[ix][h[ix] == 0]
                if len(x0): out[ix] += BETA[h[ix]] * np.clip(float(ST[IDX[p], 23]) - BIAS - x0[0], -2, 2)
        out = out + TXo
        for r in order:
            if r not in PL or PL[r][0] != "CH": continue
            kind, p, q, dt = PL[r]; ix = pos[r]; ip = pos[p]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
            if not (len(x0) and len(a)): continue
            raw = a[0] - BIAS - x0[0]; u = BETA[h[ix]] * np.clip(raw, -2, 2); hh = h[ix]
            F = dict(h=hh, absu=np.full(len(ix), abs(raw)), it0=np.full(len(ix), IT0.get(r, np.nan)), p=np.full(len(ix), q), dt=np.full(len(ix), dt))
            if collect: rows.append(pd.DataFrame(dict(ix=ix, u=u, **F)))
            if gate is not None: out[ix] += gate(F) * u
        return (out, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()) if collect else out
    return run
COLS_A = ["h", "absu", "p", "dt"]; COLS_B = ["h", "absu", "it0"]
def feats(F, cols):
    Z = np.column_stack([np.asarray(F[c], float) for c in cols]); Z = np.nan_to_num(Z, nan=15.0)
    sc = np.array([{"h": 23.0, "absu": 2.0, "it0": 20.0, "p": 1.0, "dt": 1.5}[c] for c in cols]); return Z / sc
def fit_gate(r, u, F, cols):
    Z = feats(F, cols); X = np.column_stack([np.ones(len(Z)), Z])
    def loss(th):
        w = 1.5 / (1 + np.exp(-(X @ th))); return float(((r - w * u) ** 2).sum()) + 1e-3 * float((th ** 2).sum())
    th = minimize(loss, np.zeros(X.shape[1]), method="L-BFGS-B").x
    return th, (lambda Fd, th=th: 1.5 / (1 + np.exp(-(np.column_stack([np.ones(len(Fd["h"])), feats(Fd, cols)]) @ th))))
# ---- method A: fit on PCH1 layouts (FL3 links, wide eligibility p >= .3, |dT| <= 3) ----
FITA = []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); TRS, VF = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
    PL = {r: e for r in recs for e in [elig(FLK, r, pos, h, VF, lambda r, p: r in TRS and p in TRS[r], .3, 3.0)] if e}
    run = make_run(PL, pos, h, TXo, sorted(recs, key=lambda r: (r[:3], int(r[4:7])))); O, RW = [], None
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); o, rw = run(q, None, collect=True); O.append(o); RW = rw if RW is None else RW
    base = np.mean(O, 0); RW = RW.assign(r=(y - base)[RW.ix.values], layout=L); FITA.append(RW); print(L, "fitA rows", len(RW), "chains", sum(v[0] == "CH" for v in PL.values()), flush=True)
FA = pd.concat(FITA, ignore_index=True)
thA, gA = fit_gate(FA.r.values, FA.u.values, {c: FA[c].values for c in COLS_A}, COLS_A)

print("\n[fit] A (F13/F47 PCH1 layouts, p>=.3, |dT|<=3) theta [b0, " + ", ".join(COLS_A) + "]:", np.round(thA, 3), "| rows", len(FA))

for nm, g_, cols in (("A2", gA, COLS_A),):
    for absu in (.25, 1.0):
        F = dict(h=np.arange(24), absu=np.full(24, absu), it0=np.full(24, 12.0), p=np.full(24, .9), dt=np.full(24, .5))
        print(f"  gate {nm} weight by hour (|A-BIAS-x0| {absu}, it0 12, p .9, dT .5):", " ".join(f"{x:.2f}" for x in g_(F)))
print("  fixed M1:", " ".join(f"{x:.2f}" for x in S1 * np.exp(-np.arange(24) / TAU)))
# ---- judge on EXT10, EXT12, DIAG10 (TV3 frames, FL5 mirror links) ----
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3lcg1"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); RES, CNT = {}, []
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LKM = {r: (p, q) for r, p, q in zip(MIR[MIR.layout == L].rec, MIR[MIR.layout == L].prev, MIR[MIR.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    lab_ok = lambda r, p: p not in HELD[FO[r]]
    PLn = {r: e for r in recs for e in [elig(LKM, r, pos, h, FO, lab_ok, .8, 1.5)] if e}                     # M1 / B eligibility (fixed thresholds)
    PLw = {r: e for r in recs for e in [elig(LKM, r, pos, h, FO, lab_ok, .3, 3.0)] if e}                     # A eligibility (wide; gate decides)
    runN = make_run(PLn, pos, h, TXo, order); runW = make_run(PLw, pos, h, TXo, order); m1 = lambda F: S1 * np.exp(-F["h"] / TAU)
    P = {k: [] for k in ("NONE", "M1", "A")}
    for b in bs:
        for fam in "AB":
            q = B2.w40(M, b, cx, fam); P["NONE"].append(runN(q, None)); P["M1"].append(runN(q, m1)); P["A"].append(runW(q, gA))
    CNT.append(dict(layout=L, chains_fixed=sum(v[0] == "CH" for v in PLn.values()), chains_wide=sum(v[0] == "CH" for v in PLw.values())))
    RES[L] = dict(P=P, y=y, bk=(M.farm + "_" + (M.day // 5).astype(str)).values); print(L, "done", flush=True)
pd.set_option("display.width", 220); print(pd.DataFrame(CNT).to_string(index=False))
def cellch(L, a, b):
    R_ = RES[L]; y = R_["y"]; return [rm(p - y) / rm(q - y) - 1 for p, q in zip(R_["P"][a], R_["P"][b])]
rng = np.random.default_rng(20270159)
for a, b, judged in (("A", "M1", True), ("A", "NONE", False), ("M1", "NONE", False)):
    for L in RES: print(f"  {a} vs {b} | {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, a, b))}")
    per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (q - RES[L]["y"]) ** 2 for p, q in zip(RES[L]["P"][a], RES[L]["P"][b])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
    U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean()); cells = sum((cellch(L, a, b) for L in RES), [])
    tag = (" -> " + ("PASS" if all(x < 0 for x in cells) and pw < .025 else "FAIL")) if judged else " (desc)"
    print(f"LCG2 {a} vs {b}: 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f}{tag}")

TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id", "in_temp"]); TX["rec"] = TX.row_id.str[:7]; TX["h"] = TX.row_id.str[8:10].astype(int)
EV = pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv")); same = np.array([isinstance(p, str) and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) for r, p in zip(EV.rec, EV.prev)])
E = EV[same & ~EV.prev_is_labelled & (EV.p >= .3) & (EV.dT_air.abs() <= 3)]
for hh in (0, 6):
    Fe = dict(h=np.full(len(E), hh), absu=np.full(len(E), 1.0), p=E.p.values, dt=E.dT_air.abs().values)
    print(f"  [desc] eval wide-chain records {len(E)}: gate weight at {hh}h (|u| 1.0) min {gA(Fe).min():.2f} median {np.median(gA(Fe)):.2f} max {gA(Fe).max():.2f} | M1 {S1 * np.exp(-hh / TAU):.2f}")
