# -*- coding: utf-8 -*-
"""CLN6J (2026-10-16 집 클로드; 5점 정리판 최종 판정 — 모든 구성원을 같은 실행 재학습판으로 맞춤).  사전 고정(커밋 후 실행).  온도.
기준 B = NUM BASE(평가형 세계 번호 순, CLN4N) + codex_S(CLN5, second 포함) + pfn_S(CLN6, second 포함), 손 TX3, K8·CH5 조립.
안 CLEAN (5점 정리판) = RS BASE(FL 같은 동 어제 재연결, 번호 대체 없음, CLN2) + codex_NS + pfn_NS(second 제거) + TX3L(블록 그룹 교차), 같은 조립.
   CLEAN_handTX3 = CLEAN 이되 TX3 손 값(서술: TX3L 몫 분리).
판정(EXT10·EXT12·DIAG10 12셀): 우월 = 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .025 & EXT <= 8도 <= +2%;  대체(5점 정리판 채택) = 셀 최대 <= +.5% & MSE 95% 상한 <= +.5% & EXT <= 8도 <= +1%.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cln6j_five_point_clean_judge_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "cln6j_five_point_clean_judge_v1.py")
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
# ---- CLN1J: judge today-only BASE (5-point clean) on R27 + CH5 ----
CKC = os.path.join(env.LOCAL, "cln1_ckpt"); C1 = pd.concat([pd.read_csv(os.path.join(CKC, f)) for f in sorted(os.listdir(CKC))]).drop_duplicates("row_id")
CK2 = os.path.join(env.LOCAL, "cln2_ckpt"); C2 = pd.concat([pd.read_pickle(os.path.join(CK2, f)) for f in sorted(os.listdir(CK2)) if f.endswith(".pkl")]).drop_duplicates("row_id"); C1 = C1.merge(C2, on="row_id", how="left")
CK4 = os.path.join(env.LOCAL, "cln4n_ckpt"); C4 = pd.concat([pd.read_pickle(os.path.join(CK4, f)) for f in sorted(os.listdir(CK4)) if f.endswith(".pkl")]).drop_duplicates("row_id"); C1 = C1.merge(C4, on="row_id", how="left")
CK5 = os.path.join(env.LOCAL, "cln5_ckpt"); C5 = pd.concat([pd.read_csv(os.path.join(CK5, f)) for f in sorted(os.listdir(CK5))]).drop_duplicates("row_id"); C1 = C1.merge(C5, on="row_id", how="left")
CK6 = os.path.join(env.LOCAL, "cln6_ckpt"); C6 = pd.concat([pd.read_csv(os.path.join(CK6, f)) for f in sorted(os.listdir(CK6))]).drop_duplicates("row_id"); C1 = C1.merge(C6, on="row_id", how="left")
def w40c(M, bcol, cxcol, pcol):
    g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); return (.4 * M[bcol] + (.2 + .4 * (1 - g)) * M[cxcol] + .4 * g * M[pcol]).values
TXP = {0: (.5, 14.0, 1.2), 1: (.5, 14.0, .8)}           # TX3L (7.026) block-group fits; judged group uses the other group's constants
for L, F in FR.items():
    M = F["M"].merge(C1, on="row_id", how="left", suffixes=("", "_cln")); assert len(M) == len(F["M"]); F["M"] = M
    miss = int(M.base_RS_7.isna().sum()); dif = float(np.nanmax(np.abs(M["base_REF_7_cln"] - M["base_REF_7"]))) if "base_REF_7_cln" in M else float("nan")
    print(L, "rows", len(M), "| TOD missing", miss, "| REF retrain max abs diff vs stored %.4f" % dif, flush=True)
RES = {}
for L, F in FR.items():
    M = F["M"]; M = M.assign(**{c: M[c.replace("REF", "NUM")] for c in F["bs"]}); assert M[["pfn_S_A", "pfn_NS_A"]].notna().all().all()
    P = {"B": [], "CLEAN": [], "CLEAN_handTX3": []}
    for b in F["bs"]:
        rs = b.replace("REF", "RS")
        for fam in "AB":
            P["B"].append(assemble(w40c(M, b, "codex_S", "pfn_S_" + fam), F["AF"], F["CH"], F["pos"], F["h"], F["TXo"], F["order"]))
            xn = w40c(M, rs, "codex_NS", "pfn_NS_" + fam); P["CLEAN_handTX3"].append(assemble(xn, F["AF"], F["CH"], F["pos"], F["h"], F["TXo"], F["order"])); out = np.zeros(len(xn))
            for jg, fg in ((0, 1), (1, 0)):
                c, T0, e0 = TXP[fg]; o = assemble(xn, F["AF"], F["CH"], F["pos"], F["h"], tx3(M, F["key"], c, T0, e0), F["order"]); m = F["grp"] == jg; out[m] = o[m]
            P["CLEAN"].append(out)
    RES[L] = dict(P=P, y=F["y"], t=F["t"], ood=F["ood"], bk=F["bk"]); y = F["y"]
    print(f"{L} member RMSE | BASE NUM {np.mean([rm(M[b].values - y) for b in F['bs']]):.4f} RS {np.mean([rm(M[b.replace('REF', 'RS')].values - y) for b in F['bs']]):.4f} | CODEX S {rm(M.codex_S.values - y):.4f} NS {rm(M.codex_NS.values - y):.4f} | PFN S {rm(M.pfn_S_A.values - y):.4f} NS {rm(M.pfn_NS_A.values - y):.4f}", flush=True)
print('baseline B = all members retrained in the same runs: NUM BASE (eval-like world, number order) + codex_S + pfn_S (second kept), hand TX3')
for a in ("CLEAN", "CLEAN_handTX3"): judge_stats(RES, a)
