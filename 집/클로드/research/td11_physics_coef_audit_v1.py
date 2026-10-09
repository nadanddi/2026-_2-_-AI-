# -*- coding: utf-8 -*-
"""TD11: how much weight do the LINEAR physics parts of the 9th-round temperature members give to air/outside/heating
history, how stable are those weights, and on what basis are they set?  (2026-10-10 집 클로드, user: "그런 피처들의 비율이
얼마나 정확하게 들어있는지, 어떤 근거로 그 비율대로 들어있는지 확인해봐")  Descriptive, no judgement.
 BASE physics  = LinearRegression (no penalty) on phc (median-imputed), weights wb  (screen_v6.temp_members).
 CODEX physics = StandardScaler + Ridge(alpha 100) on PHYSICS_COLUMNS, weights wp (temp_mask_v1.codex_fit_predict).
Fitted on all 400 days and on each DIAG10 training part (10 fits).  Report coefficients in C per input unit, fold sd,
sign flips; family sums (air, outside, heating); implied response of the linear part to a 5 C air step (air history
family only) vs today's observed follow rate (TD10, time constant ~6 h).  Also condition number of the air family.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td11_physics_coef_audit_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def fit_base(tr, w, phc):
    imp = SimpleImputer(strategy="median").fit(tr[phc])
    m = LinearRegression().fit(imp.transform(tr[phc]), tr.sub_temp.values, sample_weight=w)
    return pd.Series(m.coef_, index=phc)


def fit_codex(tr, w, pc):
    p = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    p.fit(tr[pc], tr.sub_temp.values, ridge__sample_weight=w)
    return pd.Series(p[-1].coef_ / p[1].scale_, index=pc)


def family(c):
    if "in_temp" in c:
        return "실내온도"
    if "out_temp" in c:
        return "외기온도"
    if "heat" in c:
        return "난방"
    if "rad" in c:
        return "일사"
    return "기타"


def step(coefs, hl_of):
    """response at hour t of the air-family linear part to a +1 C step in in_temp at t=0 (ewm halflife h)."""
    out = []
    for t in (0, 1, 3, 6, 12, 24):
        s = 0.0
        for c, b in coefs.items():
            h = hl_of(c)
            if h is None:
                continue
            s += b * (1.0 if h == 0 else 1 - 0.5 ** ((t + 1) / h))
        out.append(s)
    return out


def main():
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    PC = list(W.TM.PHYSICS_COLUMNS)
    fd = dict(sets)["DIAG10"]
    for nm, D, w, cols, fit in (("BASE", lab, np.asarray(wb, float), phc, fit_base), ("CODEX", pfn, np.asarray(wp, float), PC, fit_codex)):
        full = fit(D, w, cols)
        fo = pd.DataFrame([fit(D[common.split_mask(lab, f)[0]], w[common.split_mask(lab, f)[0]], cols) for f in fd])
        T = pd.DataFrame({"전체": full, "폴드평균": fo.mean(), "폴드sd": fo.std(), "부호바뀜": (np.sign(fo) != np.sign(full)).sum()})
        T["입력sd"] = D[cols].std().values
        T["1sd당℃"] = T["전체"] * T["입력sd"]
        print("\n################ %s 선형 물리 부분 (%s)" % (nm, "LinearRegression, 벌점 없음" if nm == "BASE" else "Ridge alpha 100, 표준화"))
        pd.set_option("display.width", 220)
        print(T.round(3).to_string())
        fam = T.assign(f=[family(c) for c in T.index]).groupby("f")
        print("  계열 합(℃/단위, 폴드 sd):", {k: "%.3f (%.3f)" % (g["전체"].sum(), fo[g.index].sum(1).std()) for k, g in fam})
        air = [c for c in cols if "in_temp" in c]
        X = SimpleImputer(strategy="median").fit_transform(D[air]); X = (X - X.mean(0)) / X.std(0)
        print("  실내온도 계열 %d개 상관 조건수 %.0f (100 넘으면 비율 개별 해석 불안정)" % (len(air), np.linalg.cond(np.corrcoef(X.T))))
        if nm == "BASE":
            hl = lambda c: (0 if c.endswith("None") else int(c.split("_")[-1])) if c.startswith("ph_in_temp") else None
        else:
            hl = lambda c: 0 if c == "in_temp" else (int(c.split("reset")[-1]) if c.startswith("in_temp_reset") else None)
        print("  실내 +1℃ 계단 변화 후 선형 부분 예측 변화 (0,1,3,6,12,24시간 뒤):", np.round(step(full[[c for c in air]], hl), 2),
              " | 관측 1차 추종(시정수 6h) 기대:", np.round([1 - np.exp(-(t + 1) / 6) for t in (0, 1, 3, 6, 12, 24)], 2))


if __name__ == "__main__":
    main()
