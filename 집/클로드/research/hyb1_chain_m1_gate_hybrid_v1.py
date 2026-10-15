# -*- coding: utf-8 -*-
"""HYB1 (2026-10-16 집 클로드; 사용자 'M1 과 학습형 게이트(LCG2) 장점을 합칠 순 없나').  사전 고정(커밋 후 실행).  온도.
HYB: 확신 높은 연결(p >= .8, dT <= 1.5)은 M1(.9 exp(-h/8)) 그대로; M1 이 버리던 연결(p .3~.8 또는 dT 1.5~3)엔 학습형 게이트(LCG2 꼴 f = [h, |u|, p, dT])
     를 쓰되 **min(게이트, M1)** 로 M1 보다 세지 않게(평가 추운 기록 외삽 막기).
역할 뒤집기: 게이트는 EXT10·EXT12·DIAG10(TV3 틀, FL5 거울) 사슬 행으로 적합 -> 판정은 CV8·N·CK2(+EL1 안전), **FL7 거울 연결표**(이 배치용 새 표).
기준 = 27번 온도 셀 -> K8 -> TX3 (GRU 없음).  판정(단일 안, alpha .025): HYB vs M1 — CV8·N·CK2 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .025 & EL1 안전.
서술: GATE(상한 없는 게이트) vs M1, M1 vs 사슬 없음(이 배치·거울 표), 사슬 수.  한계: M1 의 s·tau 는 이 배치(FL3 표)에서 맞춘 값.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u hyb1_chain_m1_gate_hybrid_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "hyb1_chain_m1_gate_hybrid_v1.py")
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
            raw = a[0] - BIAS - x0[0]; u = BETA[h[ix]] * np.clip(raw, -2, 2)
            F = dict(h=h[ix], absu=np.full(len(ix), abs(raw)), p=np.full(len(ix), q), dt=np.full(len(ix), dt))
            if collect: rows.append(pd.DataFrame(dict(ix=ix, u=u, **F)))
            if gate is not None: out[ix] += gate(F) * u
        return (out, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()) if collect else out
    return run
COLS = ["h", "absu", "p", "dt"]; SC = {"h": 23.0, "absu": 2.0, "p": 1.0, "dt": 1.5}
def fit_gate(r, u, F):
    X = np.column_stack([np.ones(len(r))] + [np.asarray(F[c], float) / SC[c] for c in COLS])
    th = minimize(lambda th: float(((r - 1.5 / (1 + np.exp(-(X @ th))) * u) ** 2).sum()) + 1e-3 * float((th ** 2).sum()), np.zeros(X.shape[1]), method="L-BFGS-B").x
    return th, (lambda Fd, th=th: 1.5 / (1 + np.exp(-(np.column_stack([np.ones(len(Fd["h"]))] + [np.asarray(Fd[c], float) / SC[c] for c in COLS]) @ th))))
m1 = lambda F: S1 * np.exp(-np.asarray(F["h"], float) / TAU)
# ---- stage 1: fit gate on EXT10, EXT12, DIAG10 (TV3 frames, FL5 mirror) ----
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3hyb1"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR5 = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); FIT = []
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LK = {r: (p, q) for r, p, q in zip(MIR5[MIR5.layout == L].rec, MIR5[MIR5.layout == L].prev, MIR5[MIR5.layout == L].p)}
    PL = {r: e for r in recs for e in [elig(LK, r, pos, h, FO, lambda r, p: p not in HELD[FO[r]], .3, 3.0)] if e}
    run = make_run(PL, pos, h, TXo, sorted(recs, key=lambda r: (r[:3], int(r[4:7])))); O, RW = [], None
    for b in bs:
        for fam in "AB":
            o, rw = run(B2.w40(M, b, cx, fam), None, collect=True); O.append(o); RW = rw if RW is None else RW
    FIT.append(RW.assign(r=(y - np.mean(O, 0))[RW.ix.values])); print(L, "fit rows", len(RW), flush=True)
FT = pd.concat(FIT, ignore_index=True); th, gate = fit_gate(FT.r.values, FT.u.values, {c: FT[c].values for c in COLS})
print("[fit] gate theta [b0, h, absu, p, dt]:", np.round(th, 3), "| rows", len(FT))
F0 = dict(h=np.arange(24), absu=np.full(24, 1.0), p=np.full(24, .6), dt=np.full(24, .5)); print("  gate (p .6, dT .5, |u| 1):", " ".join(f"{x:.2f}" for x in gate(F0)))
def hyb(F):
    fixed = (np.asarray(F["p"]) >= .8) & (np.asarray(F["dt"]) <= 1.5); return np.where(fixed, m1(F), np.minimum(gate(F), m1(F)))
# ---- stage 2: judge on CV8, N, CK2 (+EL1 safety) with FL7 mirror ----
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); RES, CNT = {}, []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); TRS, VF = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
    LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}; lok = lambda r, p: r in TRS and p in TRS[r]
    order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    PLn = {r: e for r in recs for e in [elig(LK, r, pos, h, VF, lok, .8, 1.5)] if e}; PLw = {r: e for r in recs for e in [elig(LK, r, pos, h, VF, lok, .3, 3.0)] if e}
    runN = make_run(PLn, pos, h, TXo, order); runW = make_run(PLw, pos, h, TXo, order); P = {k: [] for k in ("NONE", "M1", "HYB", "GATE")}
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam)
            P["NONE"].append(runN(q, None)); P["M1"].append(runN(q, m1)); P["HYB"].append(runW(q, hyb)); P["GATE"].append(runW(q, gate))
    CNT.append(dict(layout=L, chains_fixed=sum(v[0] == "CH" for v in PLn.values()), chains_wide=sum(v[0] == "CH" for v in PLw.values())))
    RES[L] = dict(P=P, y=y, t=M.in_temp.values, bk=(M.farm + "_" + (M.day // 5).astype(str)).values); print(L, "judge done", flush=True)
pd.set_option("display.width", 220); print(pd.DataFrame(CNT).to_string(index=False))
def cellch(L, a, b):
    R_ = RES[L]; y = R_["y"]; return [rm(p - y) / rm(q - y) - 1 for p, q in zip(R_["P"][a], R_["P"][b])]
rng = np.random.default_rng(20270160)
for a, b in (("HYB", "M1"), ("GATE", "M1"), ("M1", "NONE"), ("HYB", "NONE")):
    for L in RES: print(f"  {a} vs {b} | {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, a, b))}")
    per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (q - RES[L]["y"]) ** 2 for p, q in zip(RES[L]["P"][a], RES[L]["P"][b])], 0)).groupby(RES[L]["bk"]).mean() for L in ("CV8", "N", "CK2")}
    U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean()); cells = sum((cellch(L, a, b) for L in ("CV8", "N", "CK2")), [])
    R_ = RES["EL1"]; c2 = cellch("EL1", a, b); d2 = pd.Series(np.mean([(p - R_["y"]) ** 2 - (q - R_["y"]) ** 2 for p, q in zip(R_["P"][a], R_["P"][b])], 0)).groupby(R_["bk"]).mean().values
    pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); safe = not (np.mean(c2) > 0 and pb < .025)
    tag = (" -> " + ("PASS" if all(x < 0 for x in cells) and pw < .025 and safe else "FAIL")) if (a, b) == ("HYB", "M1") else " (desc)"
    print(f"HYB1 {a} vs {b}: CV8/N/CK2 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f} | EL1 {100 * np.mean(c2):+.3f}% safe {safe}{tag}")
