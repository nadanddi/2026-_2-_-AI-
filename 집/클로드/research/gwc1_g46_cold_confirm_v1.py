# -*- coding: utf-8 -*-
"""GWC1 (2026-10-16 집 클로드).  사전 고정(커밋 후 실행).  온도.  GW1(6.971) 에서 고른 G46(V3LF 30% 게이트를 8/10 -> 4/6 으로 낮춤) 1안을 고정하고,
GW1 에서 쓰지 않은 추운 외삽 배치 EXT10·EXT12 로 새로 확인 (GW1 은 12셀 모두 개선이나 P .037 로 FAIL - 이 확인으로 GW1 판정을 바꾸지 않음, 별도 가설검정).
틀: CB3-T 의 EXT 틀(TV3 W40G-S 셀, v3lf_ckpt V3LF, 앵커·TX3 없음).  기준 = 30% 게이트 8/10.
판정(α .025): EXT10·EXT12 8셀 모두 개선 & 합친 5일 블록 P(악화) < .025 & DIAG10 안전(유의한 악화 아님).  서술: 6~10도·10도 초과 행.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u gwc1_g46_cold_confirm_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "gwc1_g46_cold_confirm_v1.py")
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
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3gwc"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
RES = {}
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values
    VF = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    v = M[["row_id"]].merge(VF[["row_id", "V3LF"]], on="row_id", how="left").V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); t = M.in_temp.values
    P = {"B": [], "G46": []}
    for b in bs:
        for fam in "AB":
            q = B2.w40(M, b, cx, fam); P["B"].append(q + .30 * gate(M, 8, 10) * ok * (vv - q)); P["G46"].append(q + .30 * gate(M, 4, 6) * ok * (vv - q))
    RES[L] = dict(P=P, y=y, t=t, bk=(L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values)
    print(f"{L} done | rows {len(y)} | V3LF coverage {ok.mean():.2f} | rows <= 8C {np.mean(np.nan_to_num(t, nan=99) <= 8):.3f}", flush=True)
def cellch(L, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"]["G46"], R_["P"]["B"])]
for L in RES:
    t = RES[L]["t"]; print(f"  {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L))} | 6-10C rows {100 * np.mean(cellch(L, (t > 6) & (t <= 10))):+.2f}% | >10C {100 * np.mean(cellch(L, t > 10)):+.2f}%")
rng = np.random.default_rng(20270148); cells = cellch("EXT10") + cellch("EXT12")
dd = pd.concat([pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for p, b in zip(RES[L]["P"]["G46"], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in ("EXT10", "EXT12")]).values
pw = float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) >= 0).mean())
R_ = RES["DIAG10"]; c2 = cellch("DIAG10"); d2 = pd.Series(np.mean([(p - R_["y"]) ** 2 - (b - R_["y"]) ** 2 for p, b in zip(R_["P"]["G46"], R_["P"]["B"])], 0)).groupby(R_["bk"]).mean().values
pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); safe = not (np.mean(c2) > 0 and pb < .025)
print(f"GWC1 G46: EXT10+EXT12 cells all better {all(c < 0 for c in cells)} | combined block P(worse) {pw:.4f} | DIAG10 {100 * np.mean(c2):+.3f}% safe {safe} -> {'PASS' if all(c < 0 for c in cells) and pw < .025 and safe else 'FAIL'}")
