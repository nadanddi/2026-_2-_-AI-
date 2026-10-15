# -*- coding: utf-8 -*-
"""ASM3 (2026-10-16 집 클로드; 사용자 '27번 구성에 붙여서 조립 검증해봐').  사전 고정(커밋 후 실행).  온도.
R27 = 27번 온도 구성 = 셀(W40G-S 2시드 x A/B; 27 의 NX 와 근사) -> FL K8 정답 앵커 -> TX3(.25/10/.8).  붙일 부품:
  CH = 시간 감쇠 예측 사슬(PCH3 M1: .9 exp(-h/8), BETA ta8, BIAS .36; 6.982·6.984), G = V3LF 30% 게이트 8/10 (6.954·6.970).
안: R27_CH, R27_G_CH (판정, 본페로니 2 -> alpha .0125);  R27_G 는 서술.
판정 배치: EXT10·EXT12·DIAG10 (TV3 틀), **평가 조건 거울 연결표(FL5)** 로 앵커·사슬 모두; FL3 표 결과는 서술.
판정: 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .0125.  서술: <= 8도·> 10도 행, 배치별 RMSE.
한계: 사슬 매개변수는 PCH1 배치에서 맞춤·판정 배치는 PCH3·PCH5 에서 본 배치(같은 370기록).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u asm3_r27_chain_gru_assembly_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "asm3_r27_chain_gru_assembly_v1.py")
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
XR = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "out_rad"]).rename(columns={"in_temp": "it_raw"})
S1, TAU = .9, 8.0
MIR = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); FLK3 = dict(FLK)
def chains(LK, recs, pos, h, FOLDOF, labelled_ok):
    AF, CH = {}, {}
    for r in recs:
        if r not in LK or not isinstance(LK[r][0], str) or LK[r][1] < .8: continue
        p = LK[r][0]
        if p not in IDX or abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0]) > 1.5: continue
        if labelled_ok(r, p):
            if np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
        elif p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and FOLDOF.get(p) == FOLDOF.get(r) and (h[pos[p]] == 23).any():
            CH[r] = p
    return AF, CH
def make_run(AF, CH, pos, h, TXo, order):
    def run(x, wfun):
        out = x.copy()
        for r, a in AF.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        out = out + TXo
        if wfun is not None:
            for r in order:
                if r not in CH: continue
                ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
                if len(x0) and len(a): out[ix] += wfun(ix) * BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
        return out
    return run
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3pch5"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
RES, CNT = {}, []
OPTS = ("R27", "R27_CH", "R27_G", "R27_G_CH")
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    VFr = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    M = M.merge(VFr[["row_id", "V3LF"]], on="row_id", how="left")
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10); t = M.in_temp.values
    FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LKM = {r: (p, q) for r, p, q in zip(MIR[MIR.layout == L].rec, MIR[MIR.layout == L].prev, MIR[MIR.layout == L].p)}
    order = sorted(recs, key=lambda r: (r[:3], int(r[4:7]))); m1 = lambda ix: S1 * np.exp(-h[ix] / TAU)
    dmin = pd.Series(t).groupby(key).transform("min").values; P = {}
    for tab, LK in (("MIRROR", LKM), ("FL3", FLK3)):
        AF, CH = chains(LK, recs, pos, h, FO, lambda r, p: p not in HELD[FO[r]]); run = make_run(AF, CH, pos, h, TXo, order)
        for o in OPTS: P[tab + "_" + o] = []
        for b in bs:
            for fam in "AB":
                q = B2.w40(M, b, cx, fam); qg = q + .30 * G8 * ok * (vv - q)
                P[tab + "_R27"].append(run(q, None)); P[tab + "_R27_CH"].append(run(q, m1)); P[tab + "_R27_G"].append(run(qg, None)); P[tab + "_R27_G_CH"].append(run(qg, m1))
        CNT.append(dict(layout=L, table=tab, labelled_anchor=len(AF), chain=len(CH)))
    RES[L] = dict(P=P, y=y, t=t, bk=(M.farm + "_" + (M.day // 5).astype(str)).values); print(L, "done", flush=True)
pd.set_option("display.width", 220); print(pd.DataFrame(CNT).to_string(index=False))
def cellch(L, a, b, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((q - y)[m]) - 1 for p, q in zip(R_["P"][a], R_["P"][b])]
rng = np.random.default_rng(20270156)
for tab in ("MIRROR", "FL3"):
    print(f"\n== link table {tab}" + (" (judged)" if tab == "MIRROR" else " (desc)"))
    for o in OPTS[1:]:
        a, b = f"{tab}_{o}", f"{tab}_R27"
        for L in RES:
            tt = np.nan_to_num(RES[L]["t"], nan=99)
            print(f"  {o} vs R27 | {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, a, b))} | <=8C rows {100 * np.mean(cellch(L, a, b, tt <= 8)):+.2f}% | >10C {100 * np.mean(cellch(L, a, b, tt > 10)):+.2f}%")
        per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (q - RES[L]["y"]) ** 2 for p, q in zip(RES[L]["P"][a], RES[L]["P"][b])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
        U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean()); cells = sum((cellch(L, a, b) for L in RES), [])
        judged = tab == "MIRROR" and o in ("R27_CH", "R27_G_CH")
        tag = (" -> " + ("PASS" if all(x < 0 for x in cells) and pw < .0125 else "FAIL")) if judged else " (desc)"
        print(f"ASM3 {tab} {o}: 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f}{tag}")
    for L in RES:
        R_ = RES[L]; y = R_["y"]; print(f"  RMSE {L}: " + " | ".join(f"{o} {np.mean([rm(p - y) for p in R_['P'][tab + '_' + o]]):.4f}" for o in OPTS))
