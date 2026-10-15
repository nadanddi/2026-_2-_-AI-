# -*- coding: utf-8 -*-
"""ANL2 (2026-10-16 집 클로드; ANL1 51개 온실 시험대 PASS(42/51) -> 19회차 구성에 유사 날 보조 예측 붙이기).  사전 고정(커밋 후 실행).  온도.
유사 날 예측 = 실내_h + gadj(r, h) (anl2_gadj_v1.csv: 0..h 공통 입력 궤적이 닮은 다른 49곳 날 30개의 '온실 수준 뺀 배지−실내' 평균) + 대상 농장 폴드 학습 기록의 시각별 중앙 (배지−실내).
기준 = R27 + 시간 감쇠 사슬 M1 (19회차 온도).  최종' = 최종 + beta * (유사 − 최종), beta = CV8·N·CK2·EL1 (FL7 거울) 행 최소제곱 [0, 1] (자동, 1개 값).
판정(단일 안, alpha .025): EXT10·EXT12·DIAG10 (FL5 거울) 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .025.  서술: 온도·시각대별.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u anl2_analog_on_r27ch_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "anl2_analog_on_r27ch_v1.py")
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
S1, TAU = .9, 8.0
GA = pd.read_csv(os.path.join(HERE, "anl2_gadj_v1.csv")); GAD = {(r, h): g for r, h, g in zip(GA.rec, GA.hour, GA.gadj)}
LABG = lab.assign(rec=lab.farm + "_" + lab.day.map("{:03d}".format), gap=lab.sub_temp - lab.in_temp)
def lvl_for(train_recs, farm):
    S = LABG[LABG.rec.isin(train_recs) & (LABG.farm == farm)]; return S.groupby("hour").gap.median()
def chain_run(x, y, h, key, recs, pos, TXo, LK, FO, lab_ok):
    AF, CH = {}, {}
    for r in recs:
        if r not in LK or not isinstance(LK[r][0], str): continue
        p, q = LK[r]
        if p not in IDX or q < .8: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
        if not np.isfinite(dt) or dt > 1.5: continue
        if lab_ok(r, p):
            if np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
        elif p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and FO.get(p) == FO.get(r) and (h[pos[p]] == 23).any(): CH[r] = p
    out = x.copy()
    for r, a in AF.items():
        ix = pos[r]; x0 = x[ix][h[ix] == 0]
        if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
    out = out + TXo
    for r in sorted(recs, key=lambda r: (r[:3], int(r[4:7]))):
        if r not in CH: continue
        ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
        if len(x0) and len(a): out[ix] += S1 * np.exp(-h[ix] / TAU) * BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
    return out
def analog(M, key, h, train_recs_of):
    ana = np.full(len(M), np.nan); cache = {}
    for i, (r, hh, t) in enumerate(zip(key, h, M.in_temp.values)):
        g = GAD.get((r, int(hh)))
        if g is None or not np.isfinite(t): continue
        trs = train_recs_of(r); ck = (id(trs), r[:3])
        if ck not in cache: cache[ck] = lvl_for(trs, r[:3])
        ana[i] = t + g + cache[ck].get(int(hh), 0.0)
    return ana
# ---- fit beta on CV8, N, CK2, EL1 (FL7 mirror) ----
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); XF, YF = [], []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); TRS, VF = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = frozenset(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
    LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}
    fin = np.mean([chain_run(B.w40(M, "base_R_%d" % s, "codex_R", fam), y, h, key, recs, pos, TXo, LK, VF, lambda r, p: r in TRS and p in TRS[r]) for s in O1.SEEDS for fam in "AB"], 0)
    ana = analog(M, key, h, lambda r: TRS[r]); ok = np.isfinite(ana)
    XF.append((ana - fin)[ok]); YF.append((y - fin)[ok]); print(L, "fit rows", int(ok.sum()), "/", len(M), flush=True)
xf, yf = np.concatenate(XF), np.concatenate(YF); BETA_A = float(np.clip((xf * yf).sum() / (xf * xf).sum(), 0, 1)); print(f"[fit] beta {BETA_A:.4f} (rows {len(xf)})")
# ---- judge on EXT10, EXT12, DIAG10 (FL5 mirror) ----
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3anl2"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR5 = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); ALLREC = set(LABG.rec); RES = {}
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    TRF = {k: frozenset(ALLREC - v) for k, v in HELD.items()}
    LK = {r: (p, q) for r, p, q in zip(MIR5[MIR5.layout == L].rec, MIR5[MIR5.layout == L].prev, MIR5[MIR5.layout == L].p)}
    ana = analog(M, key, h, lambda r: TRF[FO[r]]); ok = np.isfinite(ana); P = {"B": [], "A": []}
    for b in bs:
        for fam in "AB":
            f0 = chain_run(B2.w40(M, b, cx, fam), y, h, key, recs, pos, TXo, LK, FO, lambda r, p: p not in HELD[FO[r]])
            P["B"].append(f0); P["A"].append(np.where(ok, f0 + BETA_A * (np.nan_to_num(ana) - f0), f0))
    RES[L] = dict(P=P, y=y, t=M.in_temp.values, h=h, bk=(M.farm + "_" + (M.day // 5).astype(str)).values); print(L, "done | analog coverage", round(ok.mean(), 3), flush=True)
def cellch(L, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((q - y)[m]) - 1 for p, q in zip(R_["P"]["A"], R_["P"]["B"])]
for L in RES:
    tt = np.nan_to_num(RES[L]["t"], nan=99); hh = RES[L]["h"]
    print(f"  A vs R27+CH | {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L))} | <=10C {100 * np.mean(cellch(L, tt <= 10)):+.2f}% | >20C {100 * np.mean(cellch(L, tt > 20)):+.2f}% | 8-20h {100 * np.mean(cellch(L, (hh >= 8) & (hh <= 20))):+.2f}% | night {100 * np.mean(cellch(L, (hh < 8) | (hh > 20))):+.2f}%")
rng = np.random.default_rng(20270162)
per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (q - RES[L]["y"]) ** 2 for p, q in zip(RES[L]["P"]["A"], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean()); cells = sum((cellch(L) for L in RES), [])
print(f"ANL2 analog member on R27+CH: 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f} -> {'PASS' if all(x < 0 for x in cells) and pw < .025 else 'FAIL'}")
