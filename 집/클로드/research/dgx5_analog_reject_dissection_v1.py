# -*- coding: utf-8 -*-
"""DGX5 (2026-10-16 집 클로드; 사용자 '유사 날 보조 예측이 괜찮은데 우리 모델과 결합이 안 된 것 같다 - 왜 사슬과 붙이면 리젝되는지 분해').  서술 진단(판정 없음), 실행 전 커밋.  온도.
ANL2 적합 배치(CV8·N·CK2·EL1, FL7 거울)에서: [1] 단계별(셀 / +K8 / +TX3 / +사슬) 최적 beta·고침 상관 [2] 보조 예측 변화의 하루 수준 몫 vs 하루 안 모양 몫(각각 beta·상관)
[3] 기록 상태(K8·사슬·없음)·시각대·온도 구간별 beta(최종·셀 기준)·모양만 beta [4] 배치별.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dgx5_analog_reject_dissection_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "dgx5_analog_reject_dissection_v1.py")
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
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); ROWS = []
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
    AF, CH = {}, {}
    for r in recs:
        if r not in LK or not isinstance(LK[r][0], str): continue
        p, q = LK[r]
        if p not in IDX or q < .8: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
        if not np.isfinite(dt) or dt > 1.5: continue
        if r in TRS and p in TRS[r]:
            if np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
        elif p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and VF.get(p) == VF.get(r) and (h[pos[p]] == 23).any(): CH[r] = p
    st = np.array(["K8" if r in AF else ("CH" if r in CH else "none") for r in key])
    S0, S1_, S2, S3 = [], [], [], []
    for s in O1.SEEDS:
        for fam in "AB":
            x = B.w40(M, "base_R_%d" % s, "codex_R", fam); a = x.copy()
            for r, A in AF.items():
                ix = pos[r]; x0 = x[ix][h[ix] == 0]
                if len(x0): a[ix] += BETA[h[ix]] * np.clip(A - BIAS - x0[0], -2, 2)
            t3 = a + TXo; c = t3.copy()
            for r in sorted(recs, key=lambda r: (r[:3], int(r[4:7]))):
                if r not in CH: continue
                ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; A = c[ip][h[ip] == 23]
                if len(x0) and len(A): c[ix] += S1 * np.exp(-h[ix] / TAU) * BETA[h[ix]] * np.clip(A[0] - BIAS - x0[0], -2, 2)
            S0.append(x); S1_.append(a); S2.append(t3); S3.append(c)
    lv = {}
    ana = np.full(len(M), np.nan)
    for i, (r, hh, t) in enumerate(zip(key, h, M.in_temp.values)):
        g = GAD.get((r, int(hh)))
        if g is None or not np.isfinite(t): continue
        ck = (VF[r], r[:3])
        if ck not in lv: lv[ck] = LABG[LABG.rec.isin(TRS[r]) & (LABG.farm == r[:3])].groupby("hour").gap.median()
        ana[i] = t + g + lv[ck].get(int(hh), 0.0)
    ROWS.append(pd.DataFrame(dict(layout=L, rec=key, h=h, t=M.in_temp.values, y=y, st=st, ana=ana, cells=np.mean(S0, 0), k8=np.mean(S1_, 0), tx3=np.mean(S2, 0), fin=np.mean(S3, 0))))
    print(L, "done", flush=True)
D = pd.concat(ROWS, ignore_index=True); D = D[np.isfinite(D.ana)].reset_index(drop=True)
def beta(base, m=None):
    m = np.ones(len(D), bool) if m is None else m; x = (D.ana - D[base]).values[m]; r = (D.y - D[base]).values[m]
    return float((x * r).sum() / (x * x).sum()), float(np.corrcoef(x, r)[0, 1])
pd.set_option("display.width", 220)
print("\n[1] optimal beta (unclipped) and corr_fix of (analog - stage) vs (y - stage), by pipeline stage")
for stg in ("cells", "k8", "tx3", "fin"): b, c = beta(stg); print(f"  {stg:6s}: beta {b:+.4f} | corr {c:+.3f} | stage RMSE {np.sqrt(((D[stg] - D.y) ** 2).mean()):.4f} | analog RMSE {np.sqrt(((D.ana - D.y) ** 2).mean()):.4f}")
print("\n[2] level vs shape of the analog correction (vs final): split d = analog - fin into record-mean part and within-record part")
g = D.groupby(["layout", "rec"]); D["d"] = D.ana - D.fin; D["r"] = D.y - D.fin
D["d_lvl"] = g.d.transform("mean"); D["d_shp"] = D.d - D.d_lvl; D["r_lvl"] = g.r.transform("mean"); D["r_shp"] = D.r - D.r_lvl
for nm in ("d_lvl", "d_shp"):
    x = D[nm].values; r = D.r.values; print(f"  {nm}: beta {(x * r).sum() / (x * x).sum():+.4f} | corr with residual {np.corrcoef(x, r)[0, 1]:+.3f} | RMS {np.sqrt((x ** 2).mean()):.3f}")
print(f"  analog level vs residual level corr {np.corrcoef(D.d_lvl, D.r_lvl)[0, 1]:+.3f} | analog shape vs residual shape corr {np.corrcoef(D.d_shp, D.r_shp)[0, 1]:+.3f}")
print("\n[3] by record status / hour group / temp band: beta & corr (vs final), and vs cells")
D["hg"] = pd.cut(D.h, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]); D["band"] = pd.cut(D.t, [-99, 10, 15, 20, 99], labels=["<=10", "10-15", "15-20", ">20"])
for col in ("st", "hg", "band"):
    rows = []
    for v, S in D.groupby(col, observed=True):
        m = (D[col] == v).values; bf, cf = beta("fin", m); bc, cc = beta("cells", m); xs = D.d_shp.values[m]; rs = D.r.values[m]
        rows.append(dict(**{col: v}, rows=int(m.sum()), beta_fin=bf, corr_fin=cf, beta_cells=bc, corr_cells=cc, beta_shape_fin=(xs * rs).sum() / (xs * xs).sum()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\n[4] per layout: beta (fin), beta shape-only (fin)")
for L, S in D.groupby("layout"):
    x = (S.ana - S.fin).values; r = (S.y - S.fin).values; xs = S.d_shp.values
    print(f"  {L}: beta {(x * r).sum() / (x * x).sum():+.4f} | shape-only beta {(xs * r).sum() / (xs * xs).sum():+.4f} | corr shape {np.corrcoef(xs, r)[0, 1]:+.3f}")
D.to_csv(os.path.join(HERE, "dgx5_rows_v1.csv.gz"), index=False, compression="gzip")
