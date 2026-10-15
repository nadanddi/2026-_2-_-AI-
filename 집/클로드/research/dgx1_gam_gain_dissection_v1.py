# -*- coding: utf-8 -*-
"""DGX1 (2026-10-15 집 클로드; 사용자 '아이디어는 좋아 보이는데, 내부 구조를 뜯어서 이득이 어디서·어떤 근거로 나오는지 철저히').  서술 진단(판정 없음), 실행 전 커밋.  온도.
GAM(서늘한 행 적합, 목표 A 배지−공기 / C 배지−24h 지연 공기, FE + PSY, 스플라인 직선 외삽) 을 기준 q(셀 + GRU30, 앵커·TX3 전) 에 .5 섞을 때, 추운 행(실내 <= 10도) 에서:
 [1] 기제: 배치·시기 x 온도 구간별 기본 편향, GAM 편향, 두 오차 상관, '고침 상관'(−기본오차 vs 보정량), 이득 합
 [2] 기록 단위 이득/손해 상위 + 맥락(하루 최저, 난방, 보온커튼, 외기, 습도, 학습 최저보다 낮은 행 수)
 [3] GAM 특징 항(가법 모형이라 항별 정확 분해) 평균: 이득 기록 vs 손해 기록
 [4] 같은 6~8도 행의 배치별 기본·GAM 편향(CV8 앞/뒤, N, CK2)
 [5] 학습 정답으로 본 실제 배지−공기 차이: 계절 구간 x 온도 구간, 난방 켬/끔
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dgx1_gam_gain_dissection_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "dgx1_gam_gain_dissection_v1.py")
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
REFC = {"A": "in_temp", "C": "ph_in_temp_24"}; FEAT_ALL = FE + PSY; NSPL = 7
def fit_contrib(tm, vm, tgk):
    a = tm & (L0.in_temp.values <= 12) & np.isfinite(L0.in_temp.values); tgv = L0.sub_temp.values - L0[REFC[tgk]].values; a = a & np.isfinite(tgv)
    Xt = L0.loc[a, FEAT_ALL].to_numpy(float); Xv = L0.loc[vm, FEAT_ALL].to_numpy(float); med = np.nanmedian(Xt, 0); Xt = np.where(np.isnan(Xt), med, Xt); Xv = np.where(np.isnan(Xv), med, Xv)
    sp = SplineTransformer(n_knots=5, degree=3, extrapolation="linear").fit(Xt); sc = StandardScaler().fit(sp.transform(Xt)); rg = Ridge(alpha=10.0).fit(sc.transform(sp.transform(Xt)), tgv[a], sample_weight=WB[a])
    Z = sc.transform(sp.transform(Xv)) * rg.coef_; contrib = np.stack([Z[:, j * NSPL:(j + 1) * NSPL].sum(1) for j in range(len(FEAT_ALL))], 1)
    pred = L0[REFC[tgk]].values[vm] + rg.intercept_ + contrib.sum(1)
    return pd.DataFrame(np.c_[pred, contrib], index=L0.row_id.values[vm], columns=["gam"] + ["c_" + f for f in FEAT_ALL])
ROWS = []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks), ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    if "fold" not in M: M["fold"] = -1
    G8 = gate(M, 8, 10); v = M.V3LF.values; miss = np.isnan(v)
    q = np.mean([B.w40(M, "base_R_%d" % s, "codex_R", fam) for s in O1.SEEDS for fam in "AB"], 0); q = q + .30 * G8 * (~miss) * (np.nan_to_num(v) - q)
    M["q"] = q; M["tmin_fold"] = np.nan
    for tgk in ("A", "C"):
        parts = []
        for k, fd in enumerate(folds):
            tm, vm = mk(lab, fd); parts.append(fit_contrib(tm, vm, tgk))
            for f in ("F13", "F47"):
                b = vm & (L0.farm == f).values
                if tgk == "A": M.loc[M.row_id.isin(L0.row_id.values[b]), "tmin_fold"] = np.nanmin(L0.in_temp.values[tm & (L0.farm == f).values])
        Cdf = pd.concat(parts).groupby(level=0).first().reindex(M.row_id); Cdf.columns = [f"{tgk}_{c}" for c in Cdf.columns]
        M = pd.concat([M.reset_index(drop=True), Cdf.reset_index(drop=True)], axis=1)
    X1 = L0.set_index("row_id")[["act_heating", "act_thermal", "out_temp", "in_hum", "ph_in_temp_24"]].reindex(M.row_id).reset_index(drop=True)
    M = pd.concat([M, X1.add_prefix("x_")], axis=1); M["layout"] = L; ROWS.append(M[M.in_temp <= 10].copy()); print(L, "done", flush=True)
D = pd.concat(ROWS, ignore_index=True); D["rec"] = D.farm + "_" + D.day.map("{:03d}".format)
D["eb"] = D.q - D.sub_temp
for tgk in ("A", "C"):
    D[f"{tgk}_eg"] = D[f"{tgk}_gam"] - D.sub_temp; D[f"{tgk}_delta"] = .5 * (D[f"{tgk}_gam"] - D.q); D[f"{tgk}_gain"] = (D.q + D[f"{tgk}_delta"] - D.sub_temp) ** 2 - D.eb ** 2
D["band"] = pd.cut(D.in_temp, [-99, 6, 8, 10], labels=["<=6", "6-8", "8-10"]); D["half"] = np.where(D.layout != "CV8", D.layout, np.where(D.fold <= 2, "CV8early", "CV8late"))
D["below_min"] = D.in_temp < D.tmin_fold
pd.set_option("display.width", 250)
print("\n[1] mechanism by layout-half x band (target C): base bias, GAM bias, corr(base err, GAM err), mean correction, sum gain (neg = good)")
T1 = D.groupby(["half", "band"], observed=True).apply(lambda g: pd.Series(dict(rows=len(g), base_bias=g.eb.mean(), gam_bias=g.C_eg.mean(), corr_err=np.corrcoef(g.eb, g.C_eg)[0, 1] if len(g) > 2 else np.nan,
    corr_fix=np.corrcoef(-g.eb, g.C_delta)[0, 1] if len(g) > 2 else np.nan, mean_delta=g.C_delta.mean(), gain=g.C_gain.sum(), base_rmse=np.sqrt((g.eb ** 2).mean()), gam_rmse=np.sqrt((g.C_eg ** 2).mean()))))
print(T1.round(3).to_string())
print("\n[2] records (target C, cold-gated rows <= 10C): top gains / losses with context")
R2 = D.groupby(["half", "rec"]).agg(gain=("C_gain", "sum"), rows=("C_gain", "size"), tmin=("in_temp", "min"), base_bias=("eb", "mean"), delta=("C_delta", "mean"), heat=("x_act_heating", "mean"),
                                  thermal=("x_act_thermal", "mean"), out=("x_out_temp", "mean"), hum=("x_in_hum", "mean"), below_min=("below_min", "sum")).reset_index()
print("gains:"); print(R2.nsmallest(10, "gain").round(2).to_string(index=False)); print("losses:"); print(R2.nlargest(10, "gain").round(2).to_string(index=False))
print("\n[3] feature-term contributions to GAM gap prediction (target C): mean term value, gain records vs loss records (records with |gain| > .05)")
cc = [c for c in D.columns if c.startswith("C_c_")]; RG = D.groupby(["half", "rec"])[cc + ["C_gain"]].mean().reset_index()
win, lose = RG[RG.C_gain < -.05 / 24], RG[RG.C_gain > .05 / 24]
print(pd.DataFrame({"term": [c[4:] for c in cc], "gain_recs": win[cc].mean().values, "loss_recs": lose[cc].mean().values, "diff": (win[cc].mean() - lose[cc].mean()).values}).round(3).sort_values("diff", key=abs, ascending=False).to_string(index=False))
print("  gain records", len(win), "loss records", len(lose))
print("\n[4] same 6-8C rows in different layouts: base bias and GAM bias by layout (target C)")
print(D[D.band == "6-8"].groupby("half").agg(rows=("eb", "size"), base_bias=("eb", "mean"), gam_bias=("C_eg", "mean"), base_rmse=("eb", lambda e: np.sqrt((e ** 2).mean())), gam_rmse=("C_eg", lambda e: np.sqrt((e ** 2).mean())), gain=("C_gain", "sum")).round(3).to_string())
print("\n[5] season: TRUE substrate - air gap (labels, training rows <= 10C) by season third and band; and by heating on/off")
T5 = L0[(L0.in_temp <= 10) & np.isfinite(L0.in_temp)].copy(); T5["gap"] = T5.sub_temp - T5.in_temp; T5["season"] = pd.cut(T5.day, [0, 110, 160, 179, 400], labels=["<=110", "111-160", "161-178", "pass-2"])
T5["band"] = pd.cut(T5.in_temp, [-99, 6, 8, 10], labels=["<=6", "6-8", "8-10"]); T5["heat"] = np.where(T5.act_heating > 0, "heat_on", "heat_off")
print(T5.groupby(["season", "band"], observed=True).gap.agg(["size", "mean", "std"]).round(2).to_string())
print(T5.groupby(["band", "heat"], observed=True).gap.agg(["size", "mean"]).round(2).to_string())
D.to_csv(os.path.join(HERE, "dgx1_rows_v1.csv.gz"), index=False, compression="gzip")
