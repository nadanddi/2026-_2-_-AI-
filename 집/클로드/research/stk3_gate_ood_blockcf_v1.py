# -*- coding: utf-8 -*-
"""STK3 (2026-10-16 집 클로드; STK2(7.015): 외삽 안전장치는 학습됐으나 농장 교차적합에서 농장 간 비중 차이로 실패 -> 같은 설계에서 교차적합 단위만 바꿈).  사전 고정(커밋 후 실행).  온도.
STK2 와 같음(게이트 입력: 시각·실내·그날 최저·어제 정보 붙음·학습 최저 + 2℃ 아래 정도(현재·그날 최저), 적합 = EXT12 + DIAG10, 구성원 BASE·CODEX·PFN·GRU) + 두 가지 변경:
  (1) 게이트 입력에 농장 표시(F47) 추가  (2) 교차적합 = 날짜 블록(농장 x 5일) 두 그룹(학습 자료 블록 목록 정렬 순 번갈아) — 판정 행은 다른 그룹 블록으로 적합한 게이트.
판정: 기준 R27 + CH5;  EXT10·EXT12·DIAG10.  기준(단일 안 S3, alpha .025): 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .025 & EXT10·EXT12 <= 8℃ 행 변화 평균 <= +2%.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u stk3_gate_ood_blockcf_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "stk3_gate_ood_blockcf_v1.py")
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
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3stk2"}
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
    G = np.column_stack([np.ones(len(M)), np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.clip(t, 0, 30) / 10, np.clip(cm, 0, 30) / 10, linked.astype(float),
                         np.clip(tmin + 2 - t, 0, 10) / 2, np.clip(tmin + 2 - cm, 0, 10) / 2, (M.farm.values == "F47").astype(float)])          # out-of-range: how far below (fold training min + 2C)
    bkv = (M.farm + "_" + (M.day // 5).astype(str)).values
    FR[L] = dict(grp=np.array([BG[x] for x in bkv]), M=M, bs=bs, cx=cx, G=G, AF=AF, CH=CH, pos=pos, h=h, TXo=TXo, order=order, y=M.sub_temp.values, farm=M.farm.values, t=M.in_temp.values, linked=linked,
                 bk=(M.farm + "_" + (M.day // 5).astype(str)).values, ood=(t < tmin + 2))
    print(L, "rows", len(M), "| out-of-range (below train min + 2C) rows", int((t < tmin + 2).sum()), flush=True)
U4 = np.array([True, True, True, True]); U3 = np.array([True, True, True, False])
def members(F, b, fam): M = F["M"]; return np.column_stack([M[b].values, M[F["cx"]].values, M["pfn_" + fam].values, M.V3LF.values])
TH = {}
for fit_farm in (0, 1):
    Xs, Gs, Ys = [], [], []
    for L in ("EXT12", "DIAG10"):
        F = FR[L]; m = F["grp"] == fit_farm
        Xm = np.mean([members(F, b, fam) for b in F["bs"] for fam in "AB"], 0); Xs.append(Xm[m]); Gs.append(F["G"][m]); Ys.append(F["y"][m])
    Xs, Gs, Ys = np.vstack(Xs), np.vstack(Gs), np.concatenate(Ys)
    for nm, use in (("S3", U4), ("S3b", U3)): TH[(nm, fit_farm)] = fit(Xs, Gs, Ys, use)
    print(f"[fit on block group {fit_farm}] rows {len(Ys)}", flush=True)
    for nm, use in (("S3", U4),):
        for desc, g in (("F13 warm in-range 15C h9", [1, np.sin(2 * np.pi * 9 / 24), np.cos(2 * np.pi * 9 / 24), 1.5, 1.5, 0, 0, 0, 0]), ("F47 warm in-range 15C h9", [1, np.sin(2 * np.pi * 9 / 24), np.cos(2 * np.pi * 9 / 24), 1.5, 1.5, 0, 0, 0, 1]),
                        ("F13 cold in-range 7C h3", [1, np.sin(2 * np.pi * 3 / 24), np.cos(2 * np.pi * 3 / 24), .7, .7, 0, 0, 0, 0]), ("F13 cold OUT-of-range h3", [1, np.sin(2 * np.pi * 3 / 24), np.cos(2 * np.pi * 3 / 24), .5, .5, 0, 2.5, 2.5, 0]),
                        ("F47 cold OUT-of-range h3", [1, np.sin(2 * np.pi * 3 / 24), np.cos(2 * np.pi * 3 / 24), .5, .5, 0, 2.5, 2.5, 1])):
            print(f"    {nm} {desc}: w BASE/CODEX/PFN/GRU {np.round(stack(np.ones((1, 4)), np.array([g]), TH[(nm, fit_farm)], use)[1][0], 2)}")
RES = {}
for L, F in FR.items():
    P = {"B": [], "S3": [], "S3b": []}
    for b in F["bs"]:
        for fam in "AB":
            Xm = members(F, b, fam); P["B"].append(assemble(B2.w40(F["M"], b, F["cx"], fam), F["AF"], F["CH"], F["pos"], F["h"], F["TXo"], F["order"]))
            for nm, use in (("S3", U4), ("S3b", U3)):
                x = np.zeros(len(Xm))
                for jg, fg in ((0, 1), (1, 0)):   # judged block group jg uses gate fitted on the other block group fg
                    m = F["grp"] == jg; x[m] = stack(Xm[m], F["G"][m], TH[(nm, fg)], use)[0]
                P[nm].append(assemble(x, F["AF"], F["CH"], F["pos"], F["h"], F["TXo"], F["order"]))
    RES[L] = dict(P=P, y=F["y"], t=F["t"], linked=F["linked"], ood=F["ood"], bk=F["bk"])
def cellch(L, a, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((q - y)[m]) - 1 for p, q in zip(R_["P"][a], R_["P"]["B"])] if m.any() else [np.nan]
rng = np.random.default_rng(20270169)
for a in ("S3", "S3b"):
    for L in RES:
        R_ = RES[L]; tt = np.nan_to_num(R_["t"], nan=99)
        print(f"  {a} vs R27+CH5 | {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, a))} | <=8C {100 * np.nanmean(cellch(L, a, tt <= 8)):+.2f}% | out-of-range {100 * np.nanmean(cellch(L, a, R_['ood'])):+.2f}% | unlinked {100 * np.mean(cellch(L, a, ~R_['linked'])):+.2f}%")
    per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (q - RES[L]["y"]) ** 2 for p, q in zip(RES[L]["P"][a], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
    U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean()); cells = sum((cellch(L, a) for L in RES), [])
    cold = np.nanmean([np.nanmean(cellch(L, a, np.nan_to_num(RES[L]["t"], nan=99) <= 8)) for L in ("EXT10", "EXT12")])
    tag = (" -> " + ("PASS" if all(x < 0 for x in cells) and pw < .025 and cold <= .02 else "FAIL")) if a == "S3" else " (desc)"
    print(f"STK3 {a}: 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f} | EXT <=8C mean change {100 * cold:+.2f}% (guard <= +2%){tag}")
