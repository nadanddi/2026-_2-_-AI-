# -*- coding: utf-8 -*-
"""PCH2 (2026-10-16 집 클로드; PCH1(6.979) 에서 사슬 기록 0~5시 −14~−36% 이나 뒤 시각·추운 행 악화 -> '새벽 한정 사슬' 가설, 사용자 '진행해').
사전 고정(커밋 후 실행).  온도.  PCH1 결과를 보고 만든 가설이므로 PCH1 에 쓴 배치(CV8·N·CK2·EL1)는 쓰지 않고 **새 배치 EXT10·EXT12·DIAG10**(TV3 틀 셀 2시드 x A/B) 로만 판정.
기준 = 셀 -> V3LF 30% 8/10(v3lf_ckpt) -> FL K8(전날이 그 폴드 가림 밖 = 정답 있음, p >= .8, |dT| <= 1.5) -> TX3(.25/10/.8).
DAWN: K8 없는 기록 r 중 FL 전날 p 가 같은 폴드 가림 안(정답 없음)·같은 농장·앞 날짜·p >= .8·|dT| <= 1.5 이면 0~5시에만 BETA[h] * clip(A − BIAS − x0, ±2) (A = p 최종 예측 23시), 6시 이후 0.
판정(단일 안, alpha .025): 세 배치 12셀 모두 개선 & 합친 5일 블록 P(악화) < .025.  서술: CH1·CH5(PCH1 형태)도 같은 배치에 표시(판정 아님), 사슬 기록 수·추운 사슬 행.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u pch2_dawn_chain_fresh_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "pch2_dawn_chain_fresh_v1.py")
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
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3pch2"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
BD = np.where(np.arange(24) <= 5, BETA, 0.0)
OPTS = {"DAWN": (1.0, BD), "CH1_desc": (1.0, BETA), "CH5_desc": (.5, BETA)}
RES, CNT = {}, []
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    VF = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    M = M.merge(VF[["row_id", "V3LF"]], on="row_id", how="left")
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10); t = M.in_temp.values
    FOLD = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FOLD.items(): HELD.setdefault(k, set()).add(r)
    AF, CH = {}, {}
    for r in recs:
        if r not in FLK or not isinstance(FLK[r][0], str) or FLK[r][1] < .8: continue
        p = FLK[r][0]
        if p not in IDX or abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0]) > 1.5: continue
        if p not in HELD[FOLD[r]]:
            if np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
        elif p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and (h[pos[p]] == 23).any():
            CH[r] = p
    order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    def run(x, strength, bet):
        out = x.copy()
        for r, a in AF.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        out = out + TXo
        if strength > 0:
            for r in order:
                if r not in CH: continue
                ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
                if len(x0) and len(a): out[ix] += strength * bet[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
        return out
    P = {"B": []}; P.update({o: [] for o in OPTS})
    for b in bs:
        for fam in "AB":
            q = B2.w40(M, b, cx, fam); q = q + .30 * G8 * ok * (vv - q); P["B"].append(run(q, 0, BETA)); [P[o].append(run(q, st, bt)) for o, (st, bt) in OPTS.items()]
    chm = np.isin(key, list(CH)); dmin = pd.Series(t).groupby(key).transform("min").values
    RES[L] = dict(P=P, y=y, t=t, ch=chm, h=h, bk=(L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values)
    CNT.append(dict(layout=L, records=len(recs), labelled_anchor=len(AF), chain=len(CH), chain_cold_recs=int(len({r for r in CH if (dmin[pos[r]] <= 8).any()})), chain_rows=int(chm.sum())))
    print(L, "done", flush=True)
pd.set_option("display.width", 220); print(pd.DataFrame(CNT).to_string(index=False))
def cellch(L, c, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"][c], R_["P"]["B"])] if m.any() else [np.nan]
for c in OPTS:
    for L in RES:
        R_ = RES[L]; tt = np.nan_to_num(R_["t"], nan=99); chm = R_["ch"]
        print(f"  {c} {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, c))} | chained rows {100 * np.mean(cellch(L, c, chm)):+.2f}% | chained <=8C {100 * np.mean(cellch(L, c, chm & (tt <= 8))):+.2f}% | chained 0-5h {100 * np.mean(cellch(L, c, chm & (R_['h'] <= 5))):+.2f}%")
rng = np.random.default_rng(20270152); c = "DAWN"
cells = cellch("EXT10", c) + cellch("EXT12", c) + cellch("DIAG10", c)
dd = pd.concat([pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for p, b in zip(RES[L]["P"][c], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES]).values
pw = float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) >= 0).mean())
print(f"PCH2 DAWN: 12 cells all better {all(x < 0 for x in cells)} | combined block P(worse) {pw:.4f} (alpha .025) -> {'PASS' if all(x < 0 for x in cells) and pw < .025 else 'FAIL'}")
