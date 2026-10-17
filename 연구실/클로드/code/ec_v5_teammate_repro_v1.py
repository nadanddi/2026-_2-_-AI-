# -*- coding: utf-8 -*-
"""V5 — 팀원 EC 파이프라인(ec_v5.md) 재현 + DIAG10·A·B·EL1 판정 (계획 v1 고정, 2026-10-09 연구실 클로드)
계획: 연구실/클로드/문서/V5_팀원EC_재현검증_계획_v1.md
기준: local/ec_rw1_all_v1.csv 의 b_* (R3 .6ET+.3LGB+.1MLP, 시드 47/1414/6464), 같은 폴드·같은 학습 제외.
실행: cd 연구실/클로드 && PYTHONPATH="" python -u code/ec_v5_teammate_repro_v1.py  (체크포인트 local/v5_ckpt)
"""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
HR = os.path.join(HERE, "..", "..", "..", u"집", u"클로드", "research"); sys.path.insert(0, HR)
import env  # noqa
sys.path.append(os.path.join(HERE, "..", "local", "pylib_catboost"))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HR, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
p3 = wt0.p3
SEEDS = (42, 1004, 2026); BSEEDS = (47, 1414, 6464); LO, HI = 0.062, 3.46
OUT = os.path.join(HERE, ".."); CK = os.path.join(OUT, "local", "v5_ckpt")
DATA = os.path.join(OUT, "..", "..", u"공용", u"대회자료", u"정형데이터", u"참가자_배포")
INP = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "in_rad", "act_vent", "act_side", "act_shade",
       "act_thermal", "act_valve", "act_heating", "act_circfan", "act_co2", "act_fog", "act_cool", "act_pump"]
MODELS = ("et", "lgb", "cat")


def feats(df):
    """문서 3.1·3.2: 파싱 → 정렬 → 온실별 ffill → 0 대치 → VPD·evap_force·sin/cos·당일 누적."""
    d = df.copy(); p = d.row_id.str.split("_", expand=True)
    d["farm"], d["day"], d["hour"] = p[0], p[1].astype(int), p[2].astype(int)
    d = d.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    d[INP] = d.groupby("farm")[INP].ffill().fillna(0)
    d["gh"] = (d.farm == "F47").astype(int)
    es = 0.61078 * np.exp(17.27 * d.in_temp / (d.in_temp + 237.3))
    d["vpd"] = np.maximum(0, es - es * d.in_hum / 100); d["evap_force"] = d.vpd * d.in_rad
    d["sin_hour"], d["cos_hour"] = np.sin(2 * np.pi * d.hour / 24), np.cos(2 * np.pi * d.hour / 24)
    d["day_cum_rad"] = d.groupby(["farm", "day"]).in_rad.cumsum(); d["day_cum_pump"] = d.groupby(["farm", "day"]).act_pump.cumsum()
    return d


FEAT = INP + ["gh", "day", "hour", "vpd", "evap_force", "sin_hour", "cos_hour", "day_cum_rad", "day_cum_pump"]


def trend_fit(tr):
    out = {}
    for f, g in tr.groupby("farm"):
        x = (g.day + g.hour / 24).to_numpy()[:, None]; y = g.sub_ec.to_numpy()
        h = HuberRegressor(epsilon=1.35, max_iter=1000).fit(x, y); s = float(h.coef_[0]); b = float(h.intercept_)
        if abs(s) > 0.04: s = float(np.clip(s, -0.04, 0.04)); b = float(np.median(y - s * x[:, 0]))
        out[f] = (s, b)
    return out


def trend_apply(tf, d):
    return np.array([tf[f][0] * (dy + hr / 24) + tf[f][1] for f, dy, hr in zip(d.farm, d.day, d.hour)])


def make(nm, s):
    if nm == "et": return ExtraTreesRegressor(n_estimators=300, n_jobs=6, random_state=s)
    if nm == "lgb": return LGBMRegressor(n_estimators=600, learning_rate=.03, num_leaves=31, subsample=.8, subsample_freq=1,
                                         colsample_bytree=.8, random_state=s, verbose=-1, n_jobs=6)
    return CatBoostRegressor(iterations=800, learning_rate=.05, depth=6, random_seed=s, verbose=0, thread_count=6)


def fit_predict(tr, va):
    tf = trend_fit(tr); ttr, tva = trend_apply(tf, tr), trend_apply(tf, va); res = tr.sub_ec.to_numpy() - ttr
    o = {"trend": tva, "slope_F13": tf.get("F13", (np.nan,))[0], "slope_F47": tf.get("F47", (np.nan,))[0]}
    for s in SEEDS:
        for nm in MODELS:
            o["%s_%d" % (nm, s)] = make(nm, s).fit(tr[FEAT], res).predict(va[FEAT])
    return o


def load():
    X = pd.read_csv(os.path.join(DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])]
    T = pd.read_csv(os.path.join(DATA, "test_X.csv"))
    nan_tr = {c for c in INP if X[c].isna().all()}; nan_te = {c for c in INP if T[c].isna().all()}
    assert nan_te <= nan_tr, ("MASK: 평가 결측 열이 학습에 있음", nan_te - nan_tr)
    for c in INP:  # 평가에 부분 결측 열이 있으면 학습 마스크 재현이 필요 — 없음을 확인
        assert T[c].isna().all() or T[c].notna().all(), ("평가 부분 결측", c)
    print("전체 결측 열(학습/평가):", sorted(nan_tr), sorted(nan_te), flush=True)
    Y = pd.read_csv(os.path.join(DATA, "train_y.csv"))[["row_id", "sub_ec"]]
    return feats(X), feats(T), Y


def run_folds():
    os.makedirs(CK, exist_ok=True)
    trf, tef, Y = load()
    raw, full, lab, lock, signatures, fds = p3.prepare()
    L = lab[["row_id", "farm", "day", "hour", "sub_ec"]].merge(trf[["row_id"] + FEAT], on="row_id", validate="one_to_one", suffixes=("", "_x"))
    assert len(L) == 8640 and (L.day == L.day_x).all() and (L.hour == L.hour_x).all(); L = L.drop(columns=["day_x", "hour_x"])
    folds = [x for x in fds if x[0] in ("DIAG10", "A", "B")]
    for f in ("F13", "F47"):
        ds = sorted(L[(L.farm == f) & (L.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for name, i, vd in folds:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path): continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(L.farm, L.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(L.farm, L.day)])
        tr, va = L[tr_m], L[va_m]
        o = fit_predict(tr, va)
        fr = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy(); fr["validator"], fr["validation_fold"] = name, i
        for k, v in o.items(): fr[k] = v
        fr.to_csv(path, index=False); print("%s/%d done (train %d, val %d)" % (name, i, len(tr), len(va)), flush=True)
    # 참고: 팀원 방식 GroupKFold(day, 5), 학습 제외 없음
    path = os.path.join(CK, "GKF_all.csv")
    if not os.path.exists(path):
        from sklearn.model_selection import GroupKFold
        parts = []
        for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(L, groups=L.day)):
            o = fit_predict(L.iloc[a], L.iloc[b]); fr = L.iloc[b][["row_id", "farm", "day", "hour", "sub_ec"]].copy()
            fr["validator"], fr["validation_fold"] = "GKF", k
            for kk, v in o.items(): fr[kk] = v
            parts.append(fr); print("GKF/%d done" % k, flush=True)
        pd.concat(parts).to_csv(path, index=False)
    # 재현 확인: 전체 학습(잠금 포함 9600행) → test
    path = os.path.join(OUT, "local", "ec_v5_repro_testpred_v1.csv")
    if not os.path.exists(path):
        A = trf.merge(Y, on="row_id", validate="one_to_one"); assert len(A) == 9600
        o = fit_predict(A, tef); fr = tef[["row_id", "farm", "day", "hour"]].copy()
        for k, v in o.items(): fr[k] = v
        fr.to_csv(path, index=False); print("test 예측 저장(제출 파일 아님)", flush=True)


r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def blend(G, w, s=None):
    if s is None:
        R = np.column_stack([np.mean([G["%s_%d" % (nm, x)] for x in SEEDS], axis=0) for nm in MODELS])
    else:
        R = np.column_stack([G["%s_%d" % (nm, s)] for nm in MODELS])
    return np.clip(R @ w + G.trend.to_numpy(), LO, HI)


def slsqp(G):
    R = np.column_stack([np.mean([G["%s_%d" % (nm, x)] for x in SEEDS], axis=0) for nm in MODELS]); y = G.sub_ec.to_numpy(); t = G.trend.to_numpy()
    f = lambda w: r(np.clip(R @ w + t, LO, HI) - y)
    return minimize(f, np.ones(3) / 3, method="SLSQP", bounds=[(0, 1)] * 3, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}]).x


def evaluate():
    V = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f != "GKF_all.csv"], ignore_index=True)
    B = pd.read_csv(os.path.join(OUT, "local", "ec_rw1_all_v1.csv"))
    keep = ["row_id", "validator", "validation_fold", "lo", "hi"] + [c for c in B.columns if c.startswith("b_")]
    M = V.merge(B[keep], on=["row_id", "validator", "validation_fold"], how="outer", indicator=True, validate="one_to_one")
    assert (M._merge == "both").all(), M._merge.value_counts()
    M = M.drop(columns="_merge"); M.to_csv(os.path.join(OUT, "local", "ec_v5_all_v1.csv"), index=False)
    bpred = lambda G, s: np.clip(.6 * G["b_et_%d" % s] + .3 * G["b_lgb_%d" % s] + .1 * G["b_mlp_%d" % s], G.lo, G.hi)
    w = slsqp(M[M.validator == "DIAG10"]); w3 = np.ones(3) / 3
    print("SLSQP 가중치(DIAG10 OOF에서 적합, A·B·EL1에 고정 적용) ET/LGB/CAT =", np.round(w, 4))
    print("기울기(폴드별) F13: %.4f~%.4f, F47: %.4f~%.4f, 잘림 비율 F13 %.2f F47 %.2f" % (
        M.groupby(["validator", "validation_fold"]).slope_F13.first().min(), M.groupby(["validator", "validation_fold"]).slope_F13.first().max(),
        M.groupby(["validator", "validation_fold"]).slope_F47.first().min(), M.groupby(["validator", "validation_fold"]).slope_F47.first().max(),
        (M.groupby(["validator", "validation_fold"]).slope_F13.first().abs() >= .04 - 1e-12).mean(),
        (M.groupby(["validator", "validation_fold"]).slope_F47.first().abs() >= .04 - 1e-12).mean()))

    def pboot(T, wt, seed=20261009):
        T = T.copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
        bm = np.mean([bpred(T, s) for s in BSEEDS], axis=0); cm = blend(T, wt)
        dd = pd.Series((cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2, index=T.index).groupby(T.cl).agg(["sum", "count"])
        sm, n = dd["sum"].values, dd["count"].values; idx = np.random.default_rng(seed).integers(0, len(sm), (20000, len(sm)))
        return r(bm - T.sub_ec), r(cm - T.sub_ec), float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())

    ok = True
    for lbl, wt in (("SLSQP(판정)", w), ("1/3 가중(서술)", w3)):
        print("\n=== V5 %s vs 기준 R3 ===" % lbl)
        for v in ("DIAG10", "A", "B", "EL1"):
            G = M[M.validator == v]; cells = []
            for cs, bs in zip(SEEDS, BSEEDS):
                a, b = r(bpred(G, bs) - G.sub_ec), r(blend(G, wt, cs) - G.sub_ec)
                if lbl.startswith("SLSQP") and v != "EL1": ok &= b < a
                cells.append("s%d↔%d %.4f→%.4f (%+.1f%%)" % (cs, bs, a, b, 100 * (b / a - 1)))
            bm, cm, p = pboot(G, wt)
            print(f"  {v:6s} 날 {G.groupby(['farm','day']).ngroups:3d} | " + "  ".join(cells) + f" | 시드평균 {bm:.4f}→{cm:.4f} ({100*(cm/bm-1):+.1f}%) P(worse) {p:.4f}")
    _, _, pd10 = pboot(M[M.validator == "DIAG10"], w)
    guards = []
    for v, H in (("EL1", M[M.validator == "EL1"]), ("DIAG10 2차", M[(M.validator == "DIAG10") & (M.day >= 179)])):
        bm, cm, _ = pboot(H, w); guards.append(100 * (cm / bm - 1)); print(f"  안전장치 {v}: {bm:.4f}→{cm:.4f} ({guards[-1]:+.1f}%)")
    passed = ok and pd10 < .025
    print(f"\nV5 판정: 전 시드×DIAG10·A·B 개선 {ok}, DIAG10 P(worse) {pd10:.4f} → {'PASS' if passed else 'FAIL'}{' (안전장치 보류)' if passed and max(guards) >= 2 else ''}")
    print("\n=== 서술: 구성원·추세선 단독 (시드평균 RMSE) ===")
    for v in ("DIAG10", "A", "B", "EL1"):
        G = M[M.validator == v]; y = G.sub_ec
        parts = {"추세선만": r(np.clip(G.trend, LO, HI) - y)}
        for k, nm in enumerate(MODELS): parts[nm] = r(blend(G, np.eye(3)[k]) - y)
        parts["기준ET"] = r(np.clip(np.mean([G["b_et_%d" % s] for s in BSEEDS], axis=0), G.lo, G.hi) - y)
        print(f"  {v:6s} " + "  ".join(f"{k} {x:.4f}" for k, x in parts.items()))
    WT = pd.read_csv(os.path.join(OUT, "local", "drive_copy", "ec3_WT1_all.csv"))[["row_id", "validator", "pfn"]]
    for v in ("DIAG10", "EL1"):
        G = M[M.validator == v].merge(WT[WT.validator == v], on=["row_id", "validator"], how="left")
        if G.pfn.isna().any(): print(f"  PFN 혼합 {v}: PFN 결측 {G.pfn.isna().sum()}행 — 생략"); continue
        bm = np.mean([np.clip(.8 * bpred(G, s) + .2 * G.pfn, G.lo, G.hi) for s in BSEEDS], axis=0); cm = np.clip(.8 * blend(G, w) + .2 * G.pfn, LO, HI)
        print(f"  PFN 혼합(0.8·X+0.2·PFN, 서술) {v}: 기준 {r(bm - G.sub_ec):.4f}  V5 {r(cm - G.sub_ec):.4f}")
    K = pd.read_csv(os.path.join(CK, "GKF_all.csv")); D = M[M.validator == "DIAG10"]
    print(f"\n참고: 팀원 방식 GroupKFold(day,5) OOF RMSE {r(blend(K, w) - K.sub_ec):.4f} (1/3 {r(blend(K, w3) - K.sub_ec):.4f})  vs 우리 DIAG10 {r(blend(D, w) - D.sub_ec):.4f}")
    P = pd.read_csv(os.path.join(OUT, "local", "ec_v5_repro_testpred_v1.csv")); P["v5"] = blend(P, w)
    S = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else os.path.join(OUT, "local", "teammate_v5", "submission_v5.csv"))
    J = P.merge(S, on="row_id", validate="one_to_one")
    print("재현 확인: 우리 재현 test 예측 vs 팀원 submission_v5 sub_ec — 상관 %.4f, RMS 차이 %.4f, 평균 %.4f vs %.4f" % (
        J.v5.corr(J.sub_ec), r(J.v5 - J.sub_ec), J.v5.mean(), J.sub_ec.mean()))
    for f, g in J.groupby("farm"): print("   %s 상관 %.4f 평균 %.4f vs %.4f" % (f, g.v5.corr(g.sub_ec), g.v5.mean(), g.sub_ec.mean()))


if __name__ == "__main__":
    run_folds()
    evaluate()
