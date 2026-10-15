# -*- coding: utf-8 -*-
"""DGX2 (2026-10-15 집 클로드; 사용자 '27번 + GRU 30% 도 DGX1 처럼 세세하게').  서술 진단(판정 없음), 실행 전 커밋.  온도.
GRU30: q1 = q0 + .30 * gate(8/10) * (V3LF - q0), q0 = W40G-S 셀(2시드 x A/B 4셀).  V3LF = FL 링크(p >= .5) 전날·전전날 + 오늘 0..h 입력 48시간 창 GRU(정답 없음).
배치 CV8(앞/뒤)·N·CK2·EL1, 이득 = 4셀 평균 SSE 변화(음수 = 좋음), 체인 이득 = 앵커(FL K8) + TX3 까지 붙인 뒤의 변화.
 [1] 배치별 합계·상위 3기록 몫·빼면 남는 이득·좋아진/나빠진 기록 비율
 [2] GRU 단독 vs 셀(활성 행): RMSE, 편향, 오차 상관, 고침 상관
 [3] 하루 수준(기록 평균 이동) 몫 vs 하루 안 모양 몫
 [4] 시각대 [5] 기록의 링크 종류(진짜 전날 / 가까운 날 / 같은 농장 먼 날 / 다른 농장 / 없음) [6] 실내온도 구간·계절·난방
 [7] 상위 이득/손해 기록, 같은 기록의 배치 간 부호 뒤집힘 [8] 앵커가 GRU 이득을 흡수하는지 [9] 평가 입력의 게이트 적용 비율(입력만)
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dgx2_gru30_gain_dissection_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "dgx2_gru30_gain_dissection_v1.py")
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
FL3 = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); LP = {r: (p, q) for r, p, q in zip(FL3.rec, FL3.prev, FL3.p)}
def ltype(r):
    if r not in LP or not isinstance(LP[r][0], str) or LP[r][1] < .5 or LP[r][0] == r: return "none(p<.5)"
    p = LP[r][0]
    if p[:3] != r[:3]: return "cross-farm"
    dd = int(p[4:7]) - int(r[4:7])
    return "true prev(d-1)" if dd == -1 else ("near(|d|<=3)" if abs(dd) <= 3 else "same-farm far")
FRAMES = []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    if "fold" not in M: M["fold"] = -1
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; miss = np.isnan(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10)
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
    Q0, Q1, A0, A1 = [], [], [], []
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); q1 = q + .30 * G8 * (~miss) * (np.nan_to_num(v) - q)
            Q0.append(q); Q1.append(q1); A0.append(apply(q) + TXo); A1.append(apply(q1) + TXo)
    F = pd.DataFrame(dict(layout=L, fold=M.fold.values, rec=key, farm=M.farm.values, day=M.day.values, hour=h, y=y, t=M.in_temp.values, g=G8, v=v,
                          q0=np.mean(Q0, 0), q1=np.mean(Q1, 0), a0=np.mean(A0, 0), a1=np.mean(A1, 0), anch=np.isin(key, list(AF))))
    F["gain"] = np.mean([(a - y) ** 2 - (b - y) ** 2 for a, b in zip(Q1, Q0)], 0); F["gain_chain"] = np.mean([(a - y) ** 2 - (b - y) ** 2 for a, b in zip(A1, A0)], 0)
    F["rel"] = 100 * np.mean([rm(a - y) / rm(b - y) - 1 for a, b in zip(Q1, Q0)]); F["rel_chain"] = 100 * np.mean([rm(a - y) / rm(b - y) - 1 for a, b in zip(A1, A0)])
    FRAMES.append(F); print(L, "done", flush=True)
D = pd.concat(FRAMES, ignore_index=True)
X1 = lab.assign(rec=lab.farm + "_" + lab.day.map("{:03d}".format))[["rec", "hour", "act_heating"]]; D = D.merge(X1, on=["rec", "hour"], how="left")
D["half"] = np.where(D.layout != "CV8", D.layout, np.where(D.fold <= 2, "CV8early", "CV8late")); D["act"] = (D.g > 0) & D.v.notna()
D["ltype"] = D.rec.map(ltype); D["hgrp"] = pd.cut(D.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])
D["band"] = pd.cut(D.t, [-99, 10, 12, 15, 20, 99], labels=["<=10", "10-12", "12-15", "15-20", ">20"]); D["season"] = pd.cut(D.day, [0, 110, 160, 179, 400], labels=["<=110", "111-160", "161-178", "pass-2"])
D["heat"] = np.where(D.act_heating > 0, "on", "off"); D["e0"] = D.q0 - D.y; D["d"] = D.q1 - D.q0
pd.set_option("display.width", 250)
print("\n[1] totals by layout (gain = mean over 4 cells of SSE change, neg = good)")
def tot(g):
    R = g.groupby("rec").gain.sum(); top = R.abs().nlargest(3).index
    return pd.Series(dict(rows=len(g), active=int(g.act.sum()), rel=g.rel.iloc[0], rel_chain=g.rel_chain.iloc[0], gain=g.gain.sum(), gain_chain=g.gain_chain.sum(),
                          top3_share=R.abs().nlargest(3).sum() / R.abs().sum(), without_top3=R.drop(top).sum(), recs_better=(R < -1e-9).mean(), recs_worse=(R > 1e-9).mean()))
print(D.groupby("half").apply(tot).round(3).to_string())
print("\n[2] GRU alone vs cells on active rows: RMSE, bias, corr(errors), corr_fix(-e0 vs d)")
A = D[D.act]
print(A.groupby("half").apply(lambda g: pd.Series(dict(rows=len(g), cells_rmse=rm(g.e0), gru_rmse=rm(g.v - g.y), cells_bias=g.e0.mean(), gru_bias=(g.v - g.y).mean(),
      corr_err=np.corrcoef(g.e0, g.v - g.y)[0, 1], corr_fix=np.corrcoef(-g.e0, g.d)[0, 1]))).round(3).to_string())
print("\n[3] level vs shape: gain = record-mean (day offset) part + within-day part")
def lvl(g):
    n = g.groupby("rec").size(); e = g.groupby("rec").e0.mean(); dm = g.groupby("rec").d.mean()
    level = float((n * ((e + dm) ** 2 - e ** 2)).sum()); return pd.Series(dict(total=g.gain.sum(), level_part=level, shape_part=g.gain.sum() - level))
print("  (cell-mean prediction; total here = SSE change of averaged q, close to [1])")
D["gain_avg"] = (D.q1 - D.y) ** 2 - (D.q0 - D.y) ** 2
def lvl2(g):
    n = g.groupby("rec").size(); e = g.groupby("rec").e0.mean(); dm = g.groupby("rec").d.mean(); level = float((n * ((e + dm) ** 2 - e ** 2)).sum())
    return pd.Series(dict(total=g.gain_avg.sum(), level_part=level, shape_part=g.gain_avg.sum() - level))
print(D.groupby("half").apply(lvl2).round(2).to_string())
for col, nm in (("hgrp", "[4] by hour group"), ("ltype", "[5] by link type of the record (training FL link)"), ("band", "[6a] by in_temp band"), ("season", "[6b] by season"), ("heat", "[6c] by heating")):
    print(f"\n{nm}: active rows, gain per active row (x1000), sum gain")
    T = D.groupby(["half", col], observed=True).apply(lambda g: pd.Series(dict(rows=len(g), active=int(g.act.sum()), per_row_x1000=1000 * g.gain.sum() / max(g.act.sum(), 1), gain=g.gain.sum())))
    print(T.round(2).unstack(0).to_string())
print("\n[7] records: top gains / losses (all layouts) and sign flips of same record across layouts")
R = D.groupby(["half", "rec"]).agg(gain=("gain", "sum"), active=("act", "sum"), tmean=("t", "mean"), e0=("e0", "mean"), d=("d", "mean"), lt=("ltype", "first")).reset_index()
print("gains:"); print(R.nsmallest(8, "gain").round(2).to_string(index=False)); print("losses:"); print(R.nlargest(8, "gain").round(2).to_string(index=False))
Pv = R[R.half.isin(["N", "CK2", "EL1"])].pivot(index="rec", columns="half", values="gain").dropna()
big = Pv[Pv.abs().max(1) > .5]; flip = ((big > 0).any(1) & (big < 0).any(1))
print(f"  records in N,CK2,EL1 with max |gain| > .5: {len(big)} | sign flips across layouts: {int(flip.sum())} | consistent better: {int((big < 0).all(1).sum())} | consistent worse: {int((big > 0).all(1).sum())}")
print("  pairwise corr of record gains:", Pv.corr().round(2).to_dict())
print("\n[8] does the anchor absorb the GRU gain? (gain vs gain_chain, anchored records vs not)")
print(D.groupby(["half", "anch"]).agg(rows=("gain", "size"), gain=("gain", "sum"), gain_chain=("gain_chain", "sum")).round(2).to_string())
print("\n[9] eval coverage: share of rows with gate > 0 (eval inputs only) vs training; eval link p>=.5 share")
TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TX = TX[TX.row_id.str[:3].isin(["F13", "F47"])].copy(); TX["farm"] = TX.row_id.str[:3]; TX["day"] = TX.row_id.str[4:7].astype(int); TX["hour"] = TX.row_id.str[8:10].astype(int)
TX = TX.reset_index(drop=True); ge = gate(TX, 8, 10); gl = gate(lab.reset_index(drop=True), 8, 10)
print(f"  eval rows {len(TX)}: gate>0 {np.mean(ge > 0):.2f}, gate=1 {np.mean(ge >= 1):.2f}, mean gate {ge.mean():.2f} | training rows {len(lab)}: gate>0 {np.mean(gl > 0):.2f}, gate=1 {np.mean(gl >= 1):.2f}, mean {gl.mean():.2f}")
EV = pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv")); print(f"  eval records with FL link p>=.5: {(EV.p >= .5).mean():.2f} (training {(FL3.p >= .5).mean():.2f})")
D.to_csv(os.path.join(HERE, "dgx2_rows_v1.csv.gz"), index=False, compression="gzip")
