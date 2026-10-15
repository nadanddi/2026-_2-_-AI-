# -*- coding: utf-8 -*-
"""EB1 (2026-10-16 집 클로드; 사용자 '온도 더 개선할 부분 찾아봐').  서술 진단(판정 없음), 실행 전 커밋.  온도.
19회차 구성(R27 + 시간 감쇠 사슬 M1) 의 남은 오차 예산: EXT10·EXT12·DIAG10(TV3 틀, FL5 거울 표), 기록 상태(K8 정답 앵커 / 사슬 / 약한 연결 / 연결 없음),
온도 구간, 시각대, 난방, 기록 수준 몫, 상위 기록 집중도, 평가 기록 구성(링크만).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u eb1_error_budget_r27ch_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "eb1_error_budget_r27ch_v1.py")
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
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3eb1"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); XR = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "act_heating"])
ROWS = []
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LK = {r: (p, q) for r, p, q in zip(MIR[MIR.layout == L].rec, MIR[MIR.layout == L].prev, MIR[MIR.layout == L].p)}
    status = {}; AF = {}; CH = {}
    for r in recs:
        st = "none(no link)"
        if r in LK and isinstance(LK[r][0], str):
            p, q = LK[r]; dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0]) if p in IDX else 9
            lab_ = p not in HELD[FO[r]]
            if q >= .8 and dt <= 1.5 and lab_ and np.isfinite(ST[IDX[p], 23]): st = "K8"; AF[r] = float(ST[IDX[p], 23])
            elif q >= .8 and dt <= 1.5 and not lab_ and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos: st = "CH"; CH[r] = p
            elif lab_: st = "weak link->labelled"
            else: st = "weak link->unlabelled"
        status[r] = st
    order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    P = []
    for b in bs:
        for fam in "AB":
            x = B2.w40(M, b, cx, fam); out = x.copy()
            for r, a in AF.items():
                ix = pos[r]; x0 = x[ix][h[ix] == 0]
                if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
            out = out + TXo
            for r in order:
                if r not in CH: continue
                ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
                if len(x0) and len(a): out[ix] += S1 * np.exp(-h[ix] / TAU) * BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
            P.append(out)
    e = np.mean(P, 0) - y
    F = pd.DataFrame(dict(layout=L, rec=key, h=h, t=M.in_temp.values, e=e, st=[status[r] for r in key], row_id=M.row_id.values)); ROWS.append(F); print(L, "done", flush=True)
D = pd.concat(ROWS, ignore_index=True).merge(XR, on="row_id", how="left")
D["band"] = pd.cut(D.t, [-99, 8, 10, 12, 15, 20, 99], labels=["<=8", "8-10", "10-12", "12-15", "15-20", ">20"]); D["hg"] = pd.cut(D.h, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])
D["heat"] = np.where(D.act_heating > 0, "on", "off"); D["se"] = D.e ** 2
pd.set_option("display.width", 220)
def tab(col):
    G = D.groupby(["layout", col], observed=True).agg(rows=("se", "size"), rmse=("se", lambda s: np.sqrt(s.mean())), bias=("e", "mean"), sse=("se", "sum"))
    G["share"] = G.sse / G.groupby(level=0).sse.transform("sum"); return G.round(3)
for col in ("st", "band", "hg", "heat"): print(f"\n[{col}] error budget (R27 + chain M1)"); print(tab(col).to_string())
R = D.groupby(["layout", "rec"]).agg(sse=("se", "sum"), mean_e=("e", "mean"), st=("st", "first"))
R["level_part"] = D.groupby(["layout", "rec"]).e.mean() ** 2 * D.groupby(["layout", "rec"]).size();
print("\n[level] share of SSE that is record-level offset:", (R.groupby(level=0).level_part.sum() / R.groupby(level=0).sse.sum()).round(3).to_dict())
for L in ("EXT10", "EXT12", "DIAG10"):
    s = R.loc[L].sse.sort_values(ascending=False); print(f"  {L}: top 10% records share {s.head(max(1, len(s) // 10)).sum() / s.sum():.2f} | top records:", s.head(6).round(1).to_dict())
print("\n[status x level] mean |record offset| by status:"); print(R.assign(absoff=R.mean_e.abs()).groupby(["st"]).absoff.describe()[["count", "mean", "50%"]].round(3).to_string())
TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id", "in_temp"]); TX["rec"] = TX.row_id.str[:7]
EV = pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv")); same = np.array([isinstance(p, str) and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) for r, p in zip(EV.rec, EV.prev)])
k8 = (EV.p >= .8) & EV.prev_is_labelled & (EV.dT_air.abs() <= 1.5); ch = same & ~EV.prev_is_labelled & (EV.p >= .8) & (EV.dT_air.abs() <= 1.5)
print(f"\n[eval composition, links only] K8 {int(k8.sum())} | chain {int(ch.sum())} | other {int((~k8 & ~ch).sum())} (records)")
