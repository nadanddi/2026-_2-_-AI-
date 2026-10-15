# -*- coding: utf-8 -*-
"""RTX1 (2026-10-16 집 클로드; LOGO2 6.974 'F13 추운 행은 선형만이 선형+잔차 트리보다 나음' -> 사용자 '진행해봐').  사전 고정(커밋 후 실행).  온도.
BASE = .65 res + .25 ridge + .10 nys,  res = 선형회귀(phc 물리 특징) + LightGBM 잔차.  폴드마다 같은 코드로 다시 적합해 구성원을 쪼갬(ckpt base_R 와 재현 차 출력).
[desc] 배치 x 온도 구간별 lin(res 의 선형 부분)·res·ridge·nys·BASE·CODEX·PFN RMSE·편향.
안 V: 추운 행에서만 BASE 의 res 를 선형 부분으로 바꿈 — BASE' = BASE + gc * (.65 lin + .25 ridge + .10 nys - BASE), gc = clip((10 - in_temp)/2, 0, 1) (결측 0).
기준 = 셀 -> V3LF 30% 8/10 -> FL K8 -> TX3(.25/10/.8) (CVJ 기준).  V 는 같은 사슬에서 셀의 BASE 만 BASE'.
판정(단일 안, alpha .025): CV8(전 폴드)·N·CK2 각 4셀(12셀) 모두 개선 & 세 배치 5일 블록 합친 P(악화) < .025 & EL1 안전(유의한 악화 아님).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u rtx1_cold_residual_tree_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "rtx1_cold_residual_tree_v1.py")
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
import screen_v6 as SV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
_lab, _pfn, CT, PHC = _wd[0], _wd[1], _wd[2], _wd[3]
def members(tr, va, w, s):
    W.cold_v5.SEED = s; SV.cold_v5.SEED = s if hasattr(SV, "cold_v5") else None
    imp = SimpleImputer(strategy="median").fit(tr[PHC]); b = SV.fit_w(LinearRegression(), imp.transform(tr[PHC]), tr.sub_temp.values, w, None)
    btr, bva = b.predict(imp.transform(tr[PHC])), b.predict(imp.transform(va[PHC])); y = tr.sub_temp.values
    r = SV.fit_w(SV.lgbh(), tr[CT], y - btr, w, None).predict(va[CT])
    rid = SV.fit_w(SV.ridge(), tr[CT], y, w, "ridge").predict(va[CT]); ny = SV.fit_w(SV.nys(), tr[CT], y, w, "ridge").predict(va[CT])
    return dict(lin=bva, res=bva + r, ridge=rid, nys=ny)
RES, TAB = {}, []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left")
    MEM = {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd)
        if not vm.sum(): continue
        for s in O1.SEEDS:
            mm = members(lab[tm], lab[vm].reset_index(drop=True), WB[tm], s)
            for nm, p in mm.items(): MEM.setdefault((nm, s), []).append(pd.Series(p, index=lab.row_id.values[vm]))
    for key, parts in MEM.items(): M["m_%s_%d" % key] = pd.concat(parts).groupby(level=0).first().reindex(M.row_id).values
    for s in O1.SEEDS:
        rb = .65 * M["m_res_%d" % s] + .25 * M["m_ridge_%d" % s] + .10 * M["m_nys_%d" % s]
        print(f"  {L} seed {s}: reproduced BASE vs ckpt base_R max|diff| {np.nanmax(np.abs(rb - M['base_R_%d' % s])):.4f}, rmse diff {np.sqrt(np.nanmean((rb - M['base_R_%d' % s]) ** 2)):.4f}", flush=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10); t = M.in_temp.values
    gc = np.nan_to_num(np.clip((10 - t) / 2, 0, 1), nan=0.0)
    TRS = {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
    AF = {r: float(ST[IDX[FLK[r][0]], 23]) for r in recs if r in FLK and r in TRS and FLK[r][1] >= .8 and FLK[r][0] in TRS[r] and np.isfinite(ST[IDX[FLK[r][0]], 23])
          and abs(IN_[IDX[r], 0, 0] - IN_[IDX[FLK[r][0]], 23, 0]) <= 1.5}
    def apply(x, AF=AF, pos=pos, h=h):
        out = x.copy()
        for r, a in AF.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        return out
    P = {"B": [], "V": []}
    for s in O1.SEEDS:
        lin_mix = .65 * M["m_lin_%d" % s].values + .25 * M["m_ridge_%d" % s].values + .10 * M["m_nys_%d" % s].values
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); M2 = M.copy(); M2["base_R_%d" % s] = M["base_R_%d" % s].values + gc * (lin_mix - M["base_R_%d" % s].values)
            qv = B.w40(M2, "base_R_%d" % s, "codex_R", fam)
            P["B"].append(apply(q + .30 * G8 * ok * (vv - q)) + TXo); P["V"].append(apply(qv + .30 * G8 * ok * (vv - qv)) + TXo)
    RES[L] = dict(P=P, y=y, t=t, bk=(L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values)
    cx = M.codex_R.values; pf = (M.pfn_A.values + M.pfn_B.values) / 2
    for nm, lo, hi in (("<=6", -99, 6), ("6-8", 6, 8), ("8-10", 8, 10), ("10-12", 10, 12), (">12", 12, 99)):
        m = (np.nan_to_num(t, nan=99) > lo) & (np.nan_to_num(t, nan=99) <= hi)
        if m.sum() < 10: continue
        row = dict(set=L, band=nm, rows=int(m.sum()))
        for nm2, arr in (("lin", np.mean([M["m_lin_%d" % s] for s in O1.SEEDS], 0)), ("res", np.mean([M["m_res_%d" % s] for s in O1.SEEDS], 0)), ("ridge", np.mean([M["m_ridge_%d" % s] for s in O1.SEEDS], 0)),
                         ("nys", np.mean([M["m_nys_%d" % s] for s in O1.SEEDS], 0)), ("BASE", np.mean([M["base_R_%d" % s] for s in O1.SEEDS], 0)), ("CODEX", cx), ("PFN", pf)):
            row[nm2] = rm((arr - y)[m]); row["b_" + nm2] = float((arr - y)[m].mean())
        TAB.append(row)
    print(L, "done", flush=True)
pd.set_option("display.width", 250); T_ = pd.DataFrame(TAB)
print("\n[desc] member RMSE by band"); print(T_[["set", "band", "rows", "lin", "res", "ridge", "nys", "BASE", "CODEX", "PFN"]].round(3).to_string(index=False))
print("\n[desc] member bias by band"); print(T_[["set", "band", "b_lin", "b_res", "b_ridge", "b_nys", "b_BASE", "b_CODEX", "b_PFN"]].round(3).to_string(index=False))
def cellch(L, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"]["V"], R_["P"]["B"])]
for L in RES:
    t = RES[L]["t"]; tt = np.nan_to_num(t, nan=99)
    print(f"  {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L))} | <=6 {100 * np.mean(cellch(L, tt <= 6)) if (tt <= 6).sum() >= 5 else float('nan'):+.2f}% | 6-8 {100 * np.mean(cellch(L, (tt > 6) & (tt <= 8))):+.2f}% | 8-10 {100 * np.mean(cellch(L, (tt > 8) & (tt <= 10))):+.2f}%")
cells = cellch("CV8") + cellch("N") + cellch("CK2"); rng = np.random.default_rng(20270149)
dd = pd.concat([pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for p, b in zip(RES[L]["P"]["V"], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in ("CV8", "N", "CK2")]).values
pw = float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) >= 0).mean())
R_ = RES["EL1"]; c2 = cellch("EL1"); d2 = pd.Series(np.mean([(p - R_["y"]) ** 2 - (b - R_["y"]) ** 2 for p, b in zip(R_["P"]["V"], R_["P"]["B"])], 0)).groupby(R_["bk"]).mean().values
pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); safe = not (np.mean(c2) > 0 and pb < .025)
print(f"RTX1 V(cold rows: BASE res -> linear only): 12 cells all better {all(c < 0 for c in cells)} | combined block P(worse) {pw:.4f} | EL1 {100 * np.mean(c2):+.3f}% safe {safe} -> {'PASS' if all(c < 0 for c in cells) and pw < .025 and safe else 'FAIL'}")
