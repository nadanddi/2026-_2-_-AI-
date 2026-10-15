# -*- coding: utf-8 -*-
"""STK1 (2026-10-16 집 클로드; BMX1(7.013) 후보 '저녁 PFN 비중'·'어제 정보 없는 기록에 GRU30' -> 사용자 '둘 다 진행, 단 특정 시간을 사람이 정하지 말고 모델이 데이터로 정하게(5점)').
사전 고정(커밋 후 실행).  온도.  학습형 섞기 게이트: 행마다 구성원 비중 w = softmax(G theta), G = [1, sin h, cos h, 실내온도/10, 그날 0..h 최저 실내온도/10, 어제 정보 붙음 표시].
  S1: 구성원 BASE·CODEX·PFN (어제 정보 표시 0 으로 고정)  |  S2: + GRU(V3LF) 넷째 구성원, 어제 정보 표시 사용 (GRU 결측 행은 GRU 비중 0).
  적합: CV8·N·CK2·EL1 의 모든 행(BASE 2시드 평균·CODEX·PFN A/B 평균·V3LF), 손실 = 평균 제곱오차 + 1e-4 |theta|^2, L-BFGS.
  어제 정보 붙음 = K8(p >= .8, dT <= 1.5, 정답 전날) 또는 CH5 사슬(p >= .5, dT <= 1.5, 정답 없는 전날), FL7(적합)·FL5(판정) 거울 표.
판정: 기준 = R27 + CH5 (셀 -> K8 -> TX3 -> 사슬 M1 p >= .5);  안 = 셀 대신 학습형 섞기 -> 같은 K8·TX3·사슬.  EXT10·EXT12·DIAG10 (TV3 틀; BMX1 분해에 안 쓴 배치).
  기준(본페로니 2 -> alpha .0125): 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .0125.  서술: 학습된 비중(시각·온도별), 어제 정보 없는 행·추운 행·18~23시.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u stk1_learned_member_gate_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "stk1_learned_member_gate_v1.py")
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
def gfeat(M, key, linked):
    t = M.in_temp.values.astype(float); tt = np.nan_to_num(t, nan=15.0)
    cm = pd.Series(tt).groupby(key).cummin().values; h = M.hour.values
    return np.column_stack([np.ones(len(M)), np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.clip(tt, 0, 30) / 10, np.clip(cm, 0, 30) / 10, linked.astype(float)])
def stack(Xm, G, th, use):
    k = Xm.shape[1]; Wm = th.reshape(G.shape[1], k); lg = G @ Wm
    lg = np.where(np.isfinite(Xm), lg, -1e9); lg[:, ~use] = -1e9; lg -= lg.max(1, keepdims=True); w = np.exp(lg); w /= w.sum(1, keepdims=True)
    return (w * np.nan_to_num(Xm)).sum(1), w
def fit(Xm, G, y, use):
    k = Xm.shape[1]; th0 = np.zeros(G.shape[1] * k)
    f = lambda th: float(((stack(Xm, G, th, use)[0] - y) ** 2).mean()) + 1e-4 * float((th ** 2).sum())
    return minimize(f, th0, method="L-BFGS-B", options=dict(maxiter=300)).x
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
# ---- fit on CV8, N, CK2, EL1 (FL7 mirror) ----
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); XF, GF, YF = [], [], []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TRS, VF = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
    LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}
    AF, CH = link_plan(LK, recs, pos, h, VF, lambda r, p: r in TRS and p in TRS[r]); linked = np.isin(key, list(AF) + list(CH))
    Xm = np.column_stack([M[[c for c in M.columns if c.startswith("base_R_")]].mean(1).values, M.codex_R.values, (M.pfn_A.values + M.pfn_B.values) / 2, M.V3LF.values if "V3LF" in M else np.full(len(M), np.nan)])
    XF.append(Xm); GF.append(gfeat(M, key, linked)); YF.append(M.sub_temp.values); print(L, "fit rows", len(M), "linked share", round(linked.mean(), 3), flush=True)
XF, GF, YF = np.vstack(XF), np.vstack(GF), np.concatenate(YF)
U3 = np.array([True, True, True, False]); U4 = np.array([True, True, True, True])
G3 = GF.copy(); G3[:, 5] = 0.0
TH = {"S1": (fit(XF, G3, YF, U3), U3, False), "S2": (fit(XF, GF, YF, U4), U4, True)}
for nm, (th, use, _) in TH.items():
    _, w = stack(XF, GF if nm == "S2" else G3, th, use); print(f"[fit] {nm}: mean weights BASE/CODEX/PFN/GRU {np.round(w.mean(0), 3)}")
    for hh in (2, 9, 15, 21):
        for tv in (6, 18):
            g = np.array([[1, np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24), tv / 10, tv / 10, 0.0]]); xm = np.ones((1, 4))
            print(f"    h {hh:2d} in_temp {tv:2d}: w {np.round(stack(xm, g, th, use)[1][0], 2)}")
# ---- judge on EXT10, EXT12, DIAG10 (TV3 frames, FL5 mirror) ----
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3stk1"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR5 = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); RES = {}
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    VFr = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    M = M.merge(VFr[["row_id", "V3LF"]], on="row_id", how="left")
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LK = {r: (p, q) for r, p, q in zip(MIR5[MIR5.layout == L].rec, MIR5[MIR5.layout == L].prev, MIR5[MIR5.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    AF, CH = link_plan(LK, recs, pos, h, FO, lambda r, p: p not in HELD[FO[r]]); linked = np.isin(key, list(AF) + list(CH)); G = gfeat(M, key, linked); G3_ = G.copy(); G3_[:, 5] = 0.0
    P = {"B": [], "S1": [], "S2": []}
    for b in bs:
        for fam in "AB":
            P["B"].append(assemble(B2.w40(M, b, cx, fam), AF, CH, pos, h, TXo, order))
            Xm = np.column_stack([M[b].values, M[cx].values, M["pfn_" + fam].values, M.V3LF.values])
            P["S1"].append(assemble(stack(Xm, G3_, TH["S1"][0], U3)[0], AF, CH, pos, h, TXo, order)); P["S2"].append(assemble(stack(Xm, G, TH["S2"][0], U4)[0], AF, CH, pos, h, TXo, order))
    RES[L] = dict(P=P, y=y, t=M.in_temp.values, h=h, linked=linked, bk=(M.farm + "_" + (M.day // 5).astype(str)).values); print(L, "done", flush=True)
def cellch(L, a, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((q - y)[m]) - 1 for p, q in zip(R_["P"][a], R_["P"]["B"])]
rng = np.random.default_rng(20270167)
for a in ("S1", "S2"):
    for L in RES:
        R_ = RES[L]; tt = np.nan_to_num(R_["t"], nan=99)
        print(f"  {a} vs R27+CH5 | {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, a))} | unlinked {100 * np.mean(cellch(L, a, ~R_['linked'])):+.2f}% | <=8C {100 * np.mean(cellch(L, a, tt <= 8)):+.2f}% | 18-23h {100 * np.mean(cellch(L, a, R_['h'] >= 18)):+.2f}%")
    per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (q - RES[L]["y"]) ** 2 for p, q in zip(RES[L]["P"][a], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
    U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean()); cells = sum((cellch(L, a) for L in RES), [])
    print(f"STK1 {a}: 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f} (alpha .0125) -> {'PASS' if all(x < 0 for x in cells) and pw < .0125 else 'FAIL'}")
