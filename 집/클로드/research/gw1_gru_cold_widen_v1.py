# -*- coding: utf-8 -*-
"""GW1 (2026-10-16 집 클로드; DGX2 6.970 '평가에서 GRU 적용 행 37%' -> 사용자 '시도해봐').  사전 고정(커밋 후 실행).  온도.
GRU(V3LF) 를 추운 행까지 넓히기.  기준 B = 셀 -> V3LF 30% 게이트 8/10 -> FL K8 -> TX3(.25/10/.8) (CVJ 기준과 같음).
안 5개: C10/C20/C30 = 게이트 밖(추운) 행에 V3LF 10/20/30% 추가 (가중 = .30*g + rc*(1-g));  G68/G46 = 게이트 자체를 6/8, 4/6 으로 낮춤(30%).
고르기: CV8 앞 폴드 0~2 평균 RMSE 최소 1안.  판정(α .025, 고르기와 판정 분리): CV8 뒤 폴드 3~5 · N · CK2 각 4셀(12셀) 모두 개선 & 세 배치 5일 블록 합친 P(악화) < .025 & EL1 안전(유의한 악화 아님).
서술: GRU 단독 vs 셀 온도 구간별, 안별 배치·구간 표, 평가 입력의 GRU 가중 평균(입력만).
주의: N·CK2 는 CV8 과 같은 학습 기록을 다른 폴드로 다시 예측한 것(기록 겹침) - 독립 표본 아님.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u gw1_gru_cold_widen_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "gw1_gru_cold_widen_v1.py")
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
OPTS = ["B", "C10", "C20", "C30", "G68", "G46"]
RES = {}
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); TXo = tx3(M, key, .25, 10, .8); t = M.in_temp.values
    GT = {"B": gate(M, 8, 10), "G68": gate(M, 6, 8), "G46": gate(M, 4, 6)}
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
    P = {o: [] for o in OPTS}; Q = []
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); Q.append(q); G8 = GT["B"]
            P["B"].append(apply(q + .30 * G8 * ok * (vv - q)) + TXo)
            for rc, nm in ((.1, "C10"), (.2, "C20"), (.3, "C30")): P[nm].append(apply(q + (.30 * G8 + rc * (1 - G8)) * ok * (vv - q)) + TXo)
            for nm in ("G68", "G46"): P[nm].append(apply(q + .30 * GT[nm] * ok * (vv - q)) + TXo)
    RES[L] = dict(P=P, y=y, t=t, bk=(M.farm + "_" + (M.day // 5).astype(str)).values, fold=M.fold.values if "fold" in M else np.zeros(len(M), int), v=v, q=np.mean(Q, 0), ok=ok)
    print(L, "done", flush=True)
def rel(L, c, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return 100 * np.mean([rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"][c], R_["P"]["B"])]) if m.any() else np.nan
pd.set_option("display.width", 240)
print("\n[desc] GRU alone vs cells by in_temp band (rows with V3LF)")
for L in RES:
    R_ = RES[L]; y = R_["y"]; t = R_["t"]
    for nm, lo, hi in (("<=6", -99, 6), ("6-8", 6, 8), ("8-10", 8, 10), (">10", 10, 99)):
        m = R_["ok"] & (t > lo) & (t <= hi)
        if m.sum() >= 5: print(f"  {L:4s} {nm:5s} rows {int(m.sum()):5d} | cells rmse {rm((R_['q'] - y)[m]):.3f} bias {(R_['q'] - y)[m].mean():+.3f} | GRU rmse {rm((R_['v'] - y)[m]):.3f} bias {(R_['v'] - y)[m].mean():+.3f}")
C8R = RES["CV8"]; early = C8R["fold"] <= 2; late = ~early; t8 = C8R["t"]
T_ = pd.DataFrame([dict(opt=n, CV8early=rel("CV8", n, early), CV8late=rel("CV8", n, late), late_le6=rel("CV8", n, late & (t8 <= 6)), late_6_8=rel("CV8", n, late & (t8 > 6) & (t8 <= 8)),
                        N=rel("N", n), N_le10=rel("N", n, RES["N"]["t"] <= 10), CK2=rel("CK2", n), CK2_le10=rel("CK2", n, RES["CK2"]["t"] <= 10), EL1=rel("EL1", n)) for n in OPTS[1:]])
print(T_.round(3).to_string(index=False))
score = {n: np.mean([rm((p - C8R["y"])[early]) for p in C8R["P"][n]]) for n in OPTS[1:]}; best = min(OPTS[1:], key=lambda n: (round(score[n], 6), OPTS.index(n)))
print("selected on CV8 early folds:", best)
rng = np.random.default_rng(20270147); cells, dds = [], []
for L, m in (("CV8", late), ("N", None), ("CK2", None)):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    ch = [rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"][best], R_["P"]["B"])]; cells += ch
    dd = pd.Series(np.mean([((p - y) ** 2 - (b - y) ** 2)[m] for p, b in zip(R_["P"][best], R_["P"]["B"])], 0)).groupby(L + R_["bk"][m]).mean(); dds.append(dd)
    print(f"  {L}: 4 cells {' '.join('%+.3f' % (100 * x) for x in ch)}")
xb = pd.concat(dds).values; pw = float((xb[rng.integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean())
R_ = RES["EL1"]; y2 = R_["y"]; c2 = [rm(p - y2) / rm(b - y2) - 1 for p, b in zip(R_["P"][best], R_["P"]["B"])]
d2 = pd.Series(np.mean([(p - y2) ** 2 - (b - y2) ** 2 for p, b in zip(R_["P"][best], R_["P"]["B"])], 0)).groupby(R_["bk"]).mean().values
pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); safe = not (np.mean(c2) > 0 and pb < .025)
print(f"GW1 {best}: 12 cells all better {all(c < 0 for c in cells)} | combined block P(worse) {pw:.4f} | EL1 {100 * np.mean(c2):+.3f}% safe {safe} -> {'PASS' if all(c < 0 for c in cells) and pw < .025 and safe else 'FAIL'}")
TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TX = TX[TX.row_id.str[:3].isin(["F13", "F47"])].copy(); TX["farm"] = TX.row_id.str[:3]; TX["day"] = TX.row_id.str[4:7].astype(int); TX["hour"] = TX.row_id.str[8:10].astype(int); TX = TX.reset_index(drop=True)
gb = gate(TX, 8, 10)
for nm, w in (("B", .3 * gb), ("C10", .3 * gb + .1 * (1 - gb)), ("C20", .3 * gb + .2 * (1 - gb)), ("C30", .3 + 0 * gb), ("G68", .3 * gate(TX, 6, 8)), ("G46", .3 * gate(TX, 4, 6))):
    print(f"  eval inputs {nm}: mean GRU weight {w.mean():.3f}, rows with weight > 0 {np.mean(w > 0):.2f}")
