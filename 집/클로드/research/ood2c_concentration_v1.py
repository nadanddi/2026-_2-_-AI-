# -*- coding: utf-8 -*-
"""OOD2C (2026-10-16 집 클로드; 서술 — OOD2 와 같은 계산 + 2차 행 기록별 dSSE 집중도).  원 OOD2 설명: (2026-10-16 집 클로드; TX3L 폐기(7.054) 뒤 OOD1 단독을 34 구조(33 + CHT, 손 TX3) 위에서 다시 판정 — 전체 행과 2차 행(평가 = 100% 2차) 두 범위).
사전 고정(커밋 후 실행).  온도.  OOD1 = 7.025 와 같은 식(학습 범위 밖 CODEX 이동, 곡선 블록 그룹 교차).
판정(각 범위, 12셀): 우월 = 12셀 모두 개선 & P(악화) < .025 & EXT <= 8도 <= +2%;  대체 = 셀 최대 <= +.5% & MSE 상한 <= +.5% & EXT <= 8도 <= +1%.
[한계: 평가의 깊은 추위(학습 최저보다 2~5℃ 낮음)는 검증 배치에 거의 없음 — 검증으로 확정 불가 영역]
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ood2_on34_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "ood2c_concentration_v1.py")
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
def link_plan(LK, recs, pos, h, FOLDOF, lab_ok):
    AF, CH = {}, {}
    for r in recs:
        if r not in LK or not isinstance(LK[r][0], str): continue
        p, q = LK[r]
        if p not in IDX: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
        if not np.isfinite(dt) or dt > 1.5: continue
        if lab_ok(r, p):
            if q >= .8 and np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
        elif q >= .5 and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and FOLDOF.get(p) == FOLDOF.get(r) and (h[pos[p]] == 23).any(): CH[r] = p
    return AF, CH
def stack(Xm, G, th, use):
    k = Xm.shape[1]; lg = G @ th.reshape(G.shape[1], k); lg = np.where(np.isfinite(Xm), lg, -1e9); lg[:, ~use] = -1e9
    lg -= lg.max(1, keepdims=True); w = np.exp(lg); w /= w.sum(1, keepdims=True); return (w * np.nan_to_num(Xm)).sum(1), w
def fit(Xm, G, y, use):
    f = lambda th: float(((stack(Xm, G, th, use)[0] - y) ** 2).mean()) + 1e-4 * float((th ** 2).sum())
    return minimize(f, np.zeros(G.shape[1] * Xm.shape[1]), method="L-BFGS-B", options=dict(maxiter=400)).x
def assemble(x, AF, CH, pos, h, TXo, order):
    out = x.copy()
    for r, a in AF.items():
        ix = pos[r]; x0 = x[ix][h[ix] == 0]
        if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
    out = out + TXo
    for r in order:
        if r not in CH: continue
        ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
        if len(x0) and len(a): out[ix] += S1 * np.exp(-h[ix] / TAU) * BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
    return out
_allbk = sorted(set(lab.farm + "_" + (lab.day // 5).astype(str))); BG = {x: i % 2 for i, x in enumerate(_allbk)}
LABR = lab.assign(rec=lab.farm + "_" + lab.day.map("{:03d}".format))
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3tx3l"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR5 = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); FR = {}
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    VFr = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    M = M.merge(VFr[["row_id", "V3LF"]], on="row_id", how="left")
    h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LK = {r: (p, q) for r, p, q in zip(MIR5[MIR5.layout == L].rec, MIR5[MIR5.layout == L].prev, MIR5[MIR5.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    AF, CH = link_plan(LK, recs, pos, h, FO, lambda r, p: p not in HELD[FO[r]]); linked = np.isin(key, list(AF) + list(CH))
    TMIN = {}
    for k, held in HELD.items():
        tr = LABR[~LABR.rec.isin(held)]
        for f in ("F13", "F47"): TMIN[(k, f)] = float(tr[tr.farm == f].in_temp.min())
    tmin = np.array([TMIN[(FO[r], r[:3])] for r in key]); t = np.nan_to_num(M.in_temp.values.astype(float), nan=15.0); cm = pd.Series(t).groupby(key).cummin().values
    G = np.column_stack([np.ones(len(M)), np.clip(t - tmin, -10, 20) / 2, np.clip(cm - tmin, -10, 20) / 2])           # S4: distance above farm training min (current, day-so-far min)
    GA = np.column_stack([np.ones(len(M)), np.clip(t, 0, 30) / 2, np.clip(cm, 0, 30) / 2])                              # S4b (desc): absolute in_temp
    miss = M.in_temp.isna().values
    bkv = (M.farm + "_" + (M.day // 5).astype(str)).values
    FR[L] = dict(GA=GA, miss=miss, grp=np.array([BG[x] for x in bkv]), M=M, bs=bs, cx=cx, G=G, AF=AF, CH=CH, pos=pos, h=h, TXo=TXo, order=order, y=M.sub_temp.values, farm=M.farm.values, t=M.in_temp.values, linked=linked,
                 bk=(M.farm + "_" + (M.day // 5).astype(str)).values, ood=(t < tmin + 2))
    print(L, "rows", len(M), "| out-of-range (below train min + 2C) rows", int((t < tmin + 2).sum()), flush=True)

def judge_stats(RES, a):
    def cellch(L, m=None):
        R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
        return [rm((p - y)[m]) / rm((q - y)[m]) - 1 for p, q in zip(R_["P"][a], R_["P"]["B"])] if m.any() else [np.nan]
    for L in RES:
        R_ = RES[L]; tt = np.nan_to_num(R_["t"], nan=99)
        print(f"  {a} vs R27+CH5 | {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L))} | <=6C {100 * np.nanmean(cellch(L, tt <= 6)):+.2f}% | <=8C {100 * np.nanmean(cellch(L, tt <= 8)):+.2f}% | out-of-range {100 * np.nanmean(cellch(L, R_['ood'])):+.2f}%")
    per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (q - RES[L]["y"]) ** 2 for p, q in zip(RES[L]["P"][a], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
    U = pd.concat(per, axis=1).mean(axis=1).values; idx = rng.integers(0, len(U), (20000, len(U))); pw = float((U[idx].mean(1) >= 0).mean()); cells = sum((cellch(L) for L in RES), [])
    base_mse = float(np.mean([np.mean(pd.Series(np.mean([(q - RES[L]["y"]) ** 2 for q in RES[L]["P"]["B"]], 0)).groupby(RES[L]["bk"]).mean()) for L in RES]))
    ub = (float(U.mean()) + 1.96 * float(np.std(U[idx].mean(1)))) / base_mse
    cold = np.nanmean([np.nanmean(cellch(L, np.nan_to_num(RES[L]["t"], nan=99) <= 8)) for L in ("EXT10", "EXT12")])
    sup = all(x < 0 for x in cells) and pw < .025 and cold <= .02; noninf = max(cells) <= .005 and ub <= .005 and cold <= .01
    print(f"{a}: 12 cells max {100 * max(cells):+.3f}% mean {100 * np.mean(cells):+.3f}% all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f} | MSE upper95 {100 * ub:+.3f}% | EXT <=8C {100 * cold:+.2f}% -> superiority {'PASS' if sup else 'FAIL'} | replacement(non-inferior) {'PASS' if noninf else 'FAIL'}")
mse = lambda e: float(np.nanmean(np.square(e)))
for L, F in FR.items():
    F["cellm"] = np.mean([B2.w40(F["M"], b, F["cx"], fam) for b in F["bs"] for fam in "AB"], 0); F["key"] = (F["M"].farm + "_" + F["M"].day.map("{:03d}".format)).values
rng = np.random.default_rng(20270171)
TXP = {0: (.5, 14.0, 1.2), 1: (.5, 14.0, .8)}   # TX3L block-group fits (7.026)
# ---- TAG2: field-type two-view agreement labelled anchors ----
F8 = pd.read_csv(os.path.join(env.LOCAL, "fl8_two_view_links.csv"))
TT = pd.read_pickle(os.path.join(env.LOCAL, "linker_v3", "truth.pkl")); TT = TT[TT.cert]
TRUEP = {f"{fb}_{int(nb):03d}": f"{fa}_{int(na):03d}" for fb, nb, fa, na in zip(TT.fb, TT.nb, TT.fa, TT.na)}

# ---- CHT: chain extension with field two-view agreement, on top of R27 + CH5 + TAG2 (= submission_33 structure) ----
TAGOPT = (1, 1.5, .5)
for L, F in FR.items():
    M = F["M"]; FO = dict(zip(F["key"], M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    sub = F8[F8.layout == L]; A = dict(F["AF"]); CHX = {}
    for r, p, q, po, pi in zip(sub.rec, sub.prev, sub.p, sub.prev_out, sub.prev_in):
        if r not in F["pos"] or not isinstance(p, str) or p not in IDX: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0]); dt = dt if np.isfinite(dt) else 99.0; ag = int(po == p) + int(pi == p)
        if r not in F["AF"] and p not in HELD[FO[r]] and np.isfinite(ST[IDX[p], 23]) and q >= .5 and q < .8 and dt <= 1.5 and ag >= 1: A[r] = float(ST[IDX[p], 23])
        elif p in HELD[FO[r]] and q < .5 and dt <= 1.5 and ag >= 1 and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in F["pos"] and (F["h"][F["pos"][p]] == 23).any(): CHX[r] = (p, q, ag)
    F["A_tag"] = A; F["CH_tag"] = {r: p for r, p in F["CH"].items() if r not in A}; F["CHX"] = CHX
OPTS = [(pm, k) for pm in (.3, 0.0) for k in (1, 2)]
for L, F in FR.items():
    for pm, k in OPTS:
        sel = [r for r, (p, q, ag) in F["CHX"].items() if q >= pm and ag >= k and r not in F["A_tag"]]; kn = [TRUEP.get(r) == F["CHX"][r][0] for r in sel if r in TRUEP]
        print(f"{L} chain-ext option p>={pm} agree>={k}: new chains {len(sel)} | true {sum(kn)}/{len(kn)} known", flush=True)
def ch_opt(F, opt):
    C = dict(F["CH_tag"])
    if opt is not None:
        pm, k = opt; C.update({r: p for r, (p, q, ag) in F["CHX"].items() if q >= pm and ag >= k and r not in F["A_tag"]})
    return C
def pick(sel):
    best = None
    for opt in [None] + OPTS:
        tot = n = 0
        for L in ("EXT12", "DIAG10"):
            F = FR[L]; out = assemble(F["cellm"], F["A_tag"], ch_opt(F, opt), F["pos"], F["h"], F["TXo"], F["order"]); m = sel(F) & np.isfinite(F["y"]); tot += float(((out - F["y"])[m] ** 2).sum()); n += int(m.sum())
        if best is None or tot / n < best[0] - 1e-12: best = (tot / n, opt)
    return best[1]
CHO = {g: pick(lambda F, g=g: F["grp"] == g) for g in (0, 1)}; CHO["all"] = pick(lambda F: np.ones(len(F["y"]), bool))
print("selected chain-extension option by block group:", CHO, flush=True)

# ---- B35: on 34b structure (A_tag + CH_tag + CHT ext + TX3L), data re-chooses chain (s, tau) and TAG2-added anchor strength {1, p} ----
def assemble2(x, AF, CH, pos, h, TXo, order, s1, tau, astr):
    out = x.copy()
    for r, a in AF.items():
        ix = pos[r]; x0 = x[ix][h[ix] == 0]
        if len(x0): out[ix] += astr.get(r, 1.0) * BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
    out = out + TXo
    for r in order:
        if r not in CH: continue
        ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
        if len(x0) and len(a): out[ix] += s1 * np.exp(-h[ix] / tau) * BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
    return out
for L, F in FR.items():
    sub = F8[F8.layout == L]; PQ = dict(zip(sub.rec, sub.p))
    F["astr_p"] = {r: float(PQ.get(r, 1.0)) for r in F["A_tag"] if r not in F["AF"]}
    F["CHfull"] = ch_opt(F, CHO["all"])
    F["TX_g"] = {g: tx3(F["M"], F["key"], *TXP[g]) for g in (0, 1)}



# ---- OOD2: 34 structure + OOD1 (shift toward CODEX below the farm training range; curve fitted per block group, cross-applied) ; judged on all rows AND pass-2 rows ----
def sfun(Z, th, miss): s = 1 / (1 + np.exp(-np.clip(Z @ th, -30, 30))); return np.where(miss, 0.0, s)
for L, F in FR.items(): F["Z"] = np.column_stack([np.ones(len(F["y"])), -F["G"][:, 1], -F["G"][:, 2]]); F["cxv"] = F["M"][F["cx"]].values
def run34(F, x, txo): return assemble2(x, F["A_tag"], F["CHfull"], F["pos"], F["h"], txo, F["order"], .9, 8.0, {})
def fit_ood(sel):
    Xc, Xx, Zs, Ys, Ms = [], [], [], [], []
    for L in ("EXT12", "DIAG10"):
        F = FR[L]; m = sel(F) & np.isfinite(F["y"]) & np.isfinite(F["cxv"]); Xc.append(F["cellm"][m]); Xx.append(F["cxv"][m]); Zs.append(F["Z"][m]); Ys.append(F["y"][m]); Ms.append(F["miss"][m])
    Xc, Xx, Zs, Ys, Ms = map(np.concatenate, (Xc, Xx, Zs, Ys, Ms))
    f = lambda th: float(((Xc + sfun(Zs, th, Ms) * (Xx - Xc) - Ys) ** 2).mean()) + 1e-4 * float((th[1:] ** 2).sum())
    best = None
    for th0 in ([-6, 1, 1], [-3, 1, 0], [-8, 2, 2], [0, 0, 0]):
        r = minimize(f, np.array(th0, float), method="L-BFGS-B", options=dict(maxiter=500))
        if best is None or r.fun < best.fun: best = r
    return best.x
TH = {g: fit_ood(lambda F, g=g: F["grp"] == g) for g in (0, 1)}
for scope in ("all", "pass2"):
    RES = {}
    for L, F in FR.items():
        sel = np.ones(len(F["y"]), bool) if scope == "all" else F["M"].day.values >= 179
        P = {"B": [], "OOD2": []}
        for b in F["bs"]:
            for fam in "AB":
                x = B2.w40(F["M"], b, F["cx"], fam); cxv = np.where(np.isfinite(F["cxv"]), F["cxv"], x); x2 = x.copy()
                for jg, fg in ((0, 1), (1, 0)):
                    m = F["grp"] == jg; x2[m] = x[m] + sfun(F["Z"][m], TH[fg], F["miss"][m]) * (cxv[m] - x[m])
                P["B"].append(run34(F, x, F["TXo"])[sel]); P["OOD2"].append(run34(F, x2, F["TXo"])[sel])
        RES[L] = dict(P=P, y=F["y"][sel], t=F["t"][sel], ood=F["ood"][sel], bk=F["bk"][sel])
    print("==== scope", scope, "====", flush=True); judge_stats(RES, "OOD2")
    if scope == "pass2":
        for L, F in FR.items():
            sel = F["M"].day.values >= 179; y = RES[L]["y"]; key = F["key"][sel]
            d = pd.Series(np.mean([(p - y) ** 2 - (q - y) ** 2 for p, q in zip(RES[L]["P"]["OOD2"], RES[L]["P"]["B"])], 0)).groupby(key).sum(); d = d[d.abs() > 1e-6].sort_values()
            tot = d.sum(); print(f"  {L}: changed records {len(d)} | helped {int((d < 0).sum())} harmed {int((d > 0).sum())} | total {tot:+.2f} | top-3 helpful share {d.head(3).sum() / d[d < 0].sum() if (d < 0).any() else float('nan'):.2f} | without top-3 {tot - d.head(3).sum():+.2f}")
            print("    top:", d.head(4).round(2).to_dict(), "| worst:", d.tail(3).round(2).to_dict())
