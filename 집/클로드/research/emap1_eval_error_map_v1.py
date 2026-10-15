# -*- coding: utf-8 -*-
"""EMAP1 (2026-10-16 집 클로드; 사용자 '꼭 그 정보만이 유일한 방법이냐' -> 평가 구성으로 다시 가중한 오차 지도).  서술(판정 없음), 실행 전 커밋.  온도.
EXT10·EXT12·DIAG10 (TV3 틀, 기준 R27 + CH5, FL5 거울 연결) 기록마다: 연결 상태(K8/사슬/없음) x 그날 실내 최저 구간(<= 6 / 6~10 / > 10도) x SSE(하루 수준 / 하루 안 모양).
평가 구성: test_X 의 F13·F47 기록 실내 최저 구간 + 29회차 평가 연결 표(fl_eval_links.csv)에 같은 규칙(K8 = 정답 전날 p >= .8 dT <= 1.5, 사슬 = 정답 없는 같은 농장 앞번호 p >= .5 dT <= 1.5).
칸별 '기록당 평균 SSE'(배치 평균) x 평가 기록 수 -> 평가에서 예상되는 SSE 몫(칸에 검증 기록 없으면 같은 온도 구간 평균으로 대체, 표시).
출력: 칸별 몫(수준·모양), 구간·상태별 합, 추운 기록 치우침, 큰 오차 기록.  목적: RMSE -11%(SSE 약 -21%)가 어느 칸 조합에서 나올 수 있는지.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u emap1_eval_error_map_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "emap1_eval_error_map_v1.py")
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
BANDS = [(-99, 6, "min<=6"), (6, 10, "6<min<=10"), (10, 99, "min>10")]
def band(v): return next(n for lo, hi, n in BANDS if lo < v <= hi) if np.isfinite(v) else "min>10"
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3emap1"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR5 = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); REC = []
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LK = {r: (p, q) for r, p, q in zip(MIR5[MIR5.layout == L].rec, MIR5[MIR5.layout == L].prev, MIR5[MIR5.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    AF, CH = link_plan(LK, recs, pos, h, FO, lambda r, p: p not in HELD[FO[r]])
    P = np.mean([assemble(B2.w40(M, b, cx, fam), AF, CH, pos, h, TXo, order) for b in bs for fam in "AB"], 0); e = P - y; t = M.in_temp.values
    for r in recs:
        ix = pos[r]; er = e[ix]; er = er[np.isfinite(er)]
        if not len(er): continue
        lv = er.mean(); REC.append(dict(layout=L, rec=r, status="K8" if r in AF else ("chain" if r in CH else "none"), band=band(np.nanmin(t[ix])), n=len(er),
                                        sse=float((er ** 2).sum()), sse_level=float(len(er) * lv ** 2), sse_shape=float(((er - lv) ** 2).sum()), level=lv, warm_bias=float(er[t[ix][np.isfinite(e[ix])] <= 6].mean()) if (t[ix] <= 6).any() else np.nan))
D = pd.DataFrame(REC); pd.set_option("display.width", 220)
# eval composition (test inputs + submission-29 eval link table; same rules: K8 = labelled prev p >= .8 dT <= 1.5, chain = unlabelled earlier same-farm prev p >= .5 dT <= 1.5)
TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TE = TE[TE.row_id.str[:3].isin(["F13", "F47"])]; TE["rec"] = TE.row_id.str[:7]
EL = pd.read_csv(os.path.join(HERE, "..", "..", "..", "제출", "미제출", "submission_29_27온도사슬M1_EC앵커확률비례_가설검정", "팜모니_정형데이터_재현패키지_29", "output_29", "fl_eval_links.csv"))
def ev_status(r):
    if r not in set(EL.rec): return "none"
    z = EL[EL.rec == r].iloc[0]; p = z.prev0; q = z.p0; dt = abs(z.dT_air) if np.isfinite(z.dT_air) else 99
    if not isinstance(p, str) or dt > 1.5: return "none"
    if bool(z.prev_is_labelled): return "K8" if q >= .8 else "none"
    return "chain" if (q >= .5 and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7])) else "none"
E = TE.groupby("rec").in_temp.min().reset_index(); E["band"] = E.in_temp.map(band); E["status"] = E.rec.map(ev_status)
print("eval records:", len(E)); print(pd.crosstab(E.band, E.status, margins=True).to_string())
print("\nvalidator records:"); print(pd.crosstab([D.layout, D.band], D.status, margins=True).to_string())
# per-cell mean per-record SSE (pooled over 3 layouts, each layout weighted equally)
g = D.groupby(["layout", "band", "status"]).agg(nrec=("rec", "size"), sse=("sse", "mean"), lev=("sse_level", "mean"), shp=("sse_shape", "mean")).reset_index()
C = g.groupby(["band", "status"]).apply(lambda z: pd.Series(dict(nrec=z.nrec.sum(), n_layouts=len(z), sse=np.average(z.sse), lev=np.average(z.lev), shp=np.average(z.shp)))).reset_index()
C["rec_rmse"] = np.sqrt(C.sse / 24); C["level_share"] = C.lev / C.sse
print("\nper-record error by cell (mean over layouts):"); print(C.round(3).to_string(index=False))
Bd = D.groupby("band").agg(sse=("sse", "mean"), lev=("sse_level", "mean"), shp=("sse_shape", "mean")).reset_index()
ec = E.groupby(["band", "status"]).size().rename("n_eval").reset_index().merge(C, on=["band", "status"], how="left")
miss = ec.sse.isna()
for i in ec.index[miss]:
    b = Bd[Bd.band == ec.at[i, "band"]].iloc[0]; ec.loc[i, ["sse", "lev", "shp"]] = [b.sse, b.lev, b.shp]
ec["fallback"] = miss; ec["eval_sse"] = ec.n_eval * ec.sse; ec["eval_lev"] = ec.n_eval * ec.lev; ec["eval_shp"] = ec.n_eval * ec.shp
tot = ec.eval_sse.sum(); ec["share"] = ec.eval_sse / tot; ec["share_level"] = ec.eval_lev / tot; ec["share_shape"] = ec.eval_shp / tot
print("\n== eval-reweighted error map (expected SSE share by cell; -11% RMSE needs ~-21% SSE) ==")
print(ec[["band", "status", "n_eval", "fallback", "rec_rmse", "share", "share_level", "share_shape"]].sort_values("share", ascending=False).round(3).to_string(index=False))
print(f"eval-reweighted RMSE estimate {np.sqrt(tot / (24 * ec.n_eval.sum())):.3f} vs unweighted validator {np.sqrt(D.sse.sum() / D.n.sum()):.3f} | level share {ec.eval_lev.sum() / tot:.2f}")
print("\nby band (eval-reweighted):"); print(ec.groupby("band")[["n_eval", "share", "share_level", "share_shape"]].sum().round(3).to_string())
print("\nby status (eval-reweighted):"); print(ec.groupby("status")[["n_eval", "share", "share_level", "share_shape"]].sum().round(3).to_string())
print("\ncold record bias (mean error, + = too warm), validators:"); print(D.groupby("band").agg(mean_level=("level", "mean"), med_level=("level", "median"), le6_row_bias=("warm_bias", "mean"), n=("rec", "size")).round(3).to_string())
D = D.sort_values("sse", ascending=False); D["cum"] = D.groupby("layout").sse.cumsum() / D.groupby("layout").sse.transform("sum")
print("\ntop records (validators):"); print(D.groupby("layout").head(5)[["layout", "rec", "status", "band", "sse", "level", "cum"]].round(3).to_string(index=False))
D.to_csv(os.path.join(HERE, "emap1_records_v1.csv"), index=False); ec.to_csv(os.path.join(HERE, "emap1_eval_map_v1.csv"), index=False)
