# -*- coding: utf-8 -*-
"""CVJ2 (2026-10-15 집 클로드; 사용자 목표 온도 .44~.45 — CVJ1 6.963 습공기 가법 스플라인(SPLPSY) 다듬기).  사전 고정(커밋 후 실행).  온도.
기준 B = W40G-S 셀 → GRU 30%(8/10) → FL K8 앵커 → TX3 27번 값(.25/10/.8)  (CVJ1: GRU30 은 CV8 포함 모든 판 개선, FLN 보충·TX3 새 값은 추운 영역 이득 없음).
후보: B 의 GRU 뒤 셀에 SPLPSY 섞기 q' = q + λ·c_h·(SPLPSY − q), c_h = clip((T − in_temp)/2, 0, 1)(결측 0), 이후 앵커·TX3.
  격자 λ {.2, .4, .6, .8, 1.0} x 게이트 T {8, 10, 12} x TX3 {27번 값, 끔} = 30.
고르기/판정 나눔(같은 추운 150행으로 고르고 판정하지 않도록): CV8 폴드 0~2(앞쪽 날짜의 추운 기록)로 고르기(4셀 평균 RMSE 최소), 폴드 3~5(뒤쪽 날짜)로 판정.
판정(단일 안 α .025): 선택 vs B — CV8 폴드 3~5 행 4셀 모두 개선 & 블록(농장 x 5일) P(악화) < .025 & N·CK2·EL1 안전(셀 평균 악화 & P(나음) < .025 → FAIL).
서술: 30조합 표(앞/뒤 CV8·N·CK2·EL1, 뒤쪽 ≤ 8℃·≤ 6℃), 선택 λ·T 에서 습공기 특징 하나씩만(AH / 엔탈피 / 이슬점차) 넣은 판.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cvj2_spl_shaping_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "cvj2_spl_shaping_v1.py")
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
LAMS = (.2, .4, .6, .8, 1.0); TS = (8, 10, 12); TXS = ("old", "off"); COMBOS = list(itertools.product(LAMS, TS, TXS)); ABL = {"AH": ["ah"], "ENTH": ["enth"], "DEW": ["dewgap"]}
RES = {}
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; miss = np.isnan(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10); t = M.in_temp.values
    TRS = {}; SP = {n: [] for n in ["SPLPSY"] + list(ABL)}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
        SP["SPLPSY"].append(spl_fit_predict(tm, vm, FE + PSY))
        for n, extra in ABL.items(): SP[n].append(spl_fit_predict(tm, vm, FE + extra))
    SPV = {n: pd.concat(s_).groupby(level=0).first().reindex(M.row_id).values for n, s_ in SP.items()}
    AF = {r: float(ST[IDX[FLK[r][0]], 23]) for r in recs if r in FLK and r in TRS and FLK[r][1] >= .8 and FLK[r][0] in TRS[r] and np.isfinite(ST[IDX[FLK[r][0]], 23])
          and abs(IN_[IDX[r], 0, 0] - IN_[IDX[FLK[r][0]], 23, 0]) <= 1.5}
    def apply(x):
        out = x.copy()
        for r, a in AF.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        return out
    G = [(q := B.w40(M, "base_R_%d" % s, "codex_R", fam)) + .30 * G8 * (~miss) * (np.nan_to_num(v) - q) for s in O1.SEEDS for fam in "AB"]
    P = {"B": [apply(g) + TXo for g in G]}
    for lam, T_, tx in COMBOS:
        cgT = np.nan_to_num(np.clip((T_ - t) / 2, 0, 1), nan=0.0); sv = SPV["SPLPSY"]; ok = ~np.isnan(sv)
        P[(lam, T_, tx)] = [apply(g + lam * cgT * ok * (np.nan_to_num(sv) - g)) + (TXo if tx == "old" else 0.0) for g in G]
    RES[L] = dict(P=P, y=y, t=t, bk=(M.farm + "_" + (M.day // 5).astype(str)).values, fold=M.fold.values if "fold" in M else np.zeros(len(M), int), SPV=SPV, G=G, apply=apply)
    print(L, "done", flush=True)
def rel(L, c, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return 100 * np.mean([rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"][c], R_["P"]["B"])]) if m.any() else np.nan
C8R = RES["CV8"]; early = C8R["fold"] <= 2; late = ~early
rows = [dict(lam=c[0], T=c[1], tx3=c[2], CV8_early=rel("CV8", c, early), CV8_late=rel("CV8", c, late), late_le8=rel("CV8", c, late & (C8R["t"] <= 8)), late_le6=rel("CV8", c, late & (C8R["t"] <= 6)),
             N=rel("N", c), CK2=rel("CK2", c), EL1=rel("EL1", c)) for c in COMBOS]
T_ = pd.DataFrame(rows); pd.set_option("display.width", 220); print(T_.round(3).to_string(index=False))
score = {c: np.mean([rm((p - C8R["y"])[early]) for p in C8R["P"][c]]) for c in COMBOS}; best = min(COMBOS, key=lambda c: (round(score[c], 6), COMBOS.index(c)))
print("selected on CV8 early folds:", best)
rng = np.random.default_rng(20270143); y = C8R["y"]
ch = [rm((p - y)[late]) / rm((b - y)[late]) - 1 for p, b in zip(C8R["P"][best], C8R["P"]["B"])]
dd = pd.Series(np.mean([((p - y) ** 2 - (b - y) ** 2)[late] for p, b in zip(C8R["P"][best], C8R["P"]["B"])], 0)).groupby(C8R["bk"][late]).mean().values
pw = float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) >= 0).mean()); ok = all(x < 0 for x in ch) and pw < .025; safe = {}
for L in ("N", "CK2", "EL1"):
    R_ = RES[L]; y2 = R_["y"]; c2 = [rm(p - y2) / rm(b - y2) - 1 for p, b in zip(R_["P"][best], R_["P"]["B"])]
    d2 = pd.Series(np.mean([(p - y2) ** 2 - (b - y2) ** 2 for p, b in zip(R_["P"][best], R_["P"]["B"])], 0)).groupby(R_["bk"]).mean().values
    pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); safe[L] = not (np.mean(c2) > 0 and pb < .025)
print(f"CVJ2 {best}: CV8 late 4 cells {' '.join('%+.3f' % (100 * x) for x in ch)} | P(worse) {pw:.4f} | safety {safe} -> {'PASS' if ok and all(safe.values()) else 'FAIL'}")
lam, T_b, tx = best
for n in ABL:
    vals = []
    for L in ("CV8",):
        R_ = RES[L]; t = R_["t"]; cgT = np.nan_to_num(np.clip((T_b - t) / 2, 0, 1), nan=0.0); sv = R_["SPV"][n]; okk = ~np.isnan(sv)
        TXo_ = 0.0
        Pn = [R_["apply"](g + lam * cgT * okk * (np.nan_to_num(sv) - g)) for g in R_["G"]]; Pb = [R_["apply"](g + lam * cgT * (~np.isnan(R_["SPV"]["SPLPSY"])) * (np.nan_to_num(R_["SPV"]["SPLPSY"]) - g)) for g in R_["G"]]
        m6 = t <= 8; vals.append(100 * np.mean([rm((a - R_["y"])[m6]) / rm((b - R_["y"])[m6]) - 1 for a, b in zip(Pn, Pb)]))
    print(f"  feature ablation (only {n} as psychrometric; vs full SPLPSY, CV8 <=8C rows): {vals[0]:+.2f}%")
