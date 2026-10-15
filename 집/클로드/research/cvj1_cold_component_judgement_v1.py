# -*- coding: utf-8 -*-
"""CVJ1 (2026-10-15 집 클로드; 사용자 '다양한 방향 + 부품 단위 + 데이터로 검증, 5점 규정' — DV1 6.960 후속).  사전 고정(커밋 후 실행).  온도.
CV8(평가형 추운 배치: 하루 최저 ≤ 8℃ 학습 42기록 6폴드, cv8_cold_validator_v1.py) 에서 부품 단위 판정 + 새 구성원(가법 스플라인) 시험.
배치: CV8(주) | N·CK2(OS1 v2)·EL1(YL1 기준) — 서술/안전.  셀 = W40G-S(BASE 1616/1717 x PFN A/B).  앵커는 그 폴드 학습 쪽 전날만.
(1) 27 vs 28 구성 및 한 부품씩 바꾸기(서술; 27 = 셀 → FL K8 → TX3(.25/10/.8)):
   R27 | R28(GRU 30%·8/10 → UNI → TX3 .40/12/.7) | R27+GRU30 | R27+UNI | R27 TX3→새 값 | R27 TX3 끔 | R27+GRU10 엄격(10/12)
(2) 새 구성원 SPL(가법 스플라인, GAM 꼴): 특징 = in_temp, ph_in_temp_1·3·8·24, out_temp, ph_rad_2, act_heating, act_thermal, hr_sin, hr_cos (+ PSY: 실내 절대습도·엔탈피·이슬점차)
   각 특징 SplineTransformer(knots 5, degree 3, extrapolation 'linear') → StandardScaler → Ridge(alpha 10), 폴드 학습 행만으로 적합(가중 = 학습 가중).
   섞기: q' = q + λ·c_h·(SPL − q), c_h = clip((10 − in_temp)/2, 0, 1)(추운 행만), λ ∈ {.2, .4} (미리 고정 2개) → 이후 R27 의 앵커·TX3 그대로.
판정(새 구성원, 본페로니 k 4 = {SPL, SPL+PSY} x λ{.2, .4}, α .00625): CV8 4셀 모두 R27 보다 개선 & CV8 블록(농장 x 5일) P(악화) < .00625 & N·EL1 안전(셀 평균 악화 & P(나음) < .025 → FAIL).
서술: 추운 행(≤ 8℃)·6~8℃·8~12℃ 구간별, SPL 단독 RMSE.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cvj1_cold_component_judgement_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "cvj1_cold_component_judgement_v1.py")
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
CF = ["R27", "R28", "R27+GRU30", "R27+UNI", "R27+TX3new", "R27-TX3", "R27+GRU10s"] + [f"{n}_{l}" for n in ("SPL", "SPLPSY") for l in (.2, .4)]
RES = {}
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; miss = np.isnan(v); TXo, TXn = tx3(M, key, .25, 10, .8), tx3(M, key, .40, 12, .7); G8, G10 = gate(M, 8, 10), gate(M, 10, 12)
    cg = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0)
    TRS = {}; SP = {"SPL": [], "SPLPSY": []}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
        SP["SPL"].append(spl_fit_predict(tm, vm, FE)); SP["SPLPSY"].append(spl_fit_predict(tm, vm, FE + PSY))
    SPV = {n: pd.concat(s).groupby(level=0).first().reindex(M.row_id).values for n, s in SP.items()}
    def anc(LK, pmin):
        return {r: float(ST[IDX[LK[r][0]], 23]) for r in recs if r in LK and r in TRS and LK[r][1] >= pmin and LK[r][0] in TRS[r] and np.isfinite(ST[IDX[LK[r][0]], 23])
                and abs(IN_[IDX[r], 0, 0] - IN_[IDX[LK[r][0]], 23, 0]) <= 1.5}
    AF = anc(FLK, .8); UNI = {**anc(LINK, .5), **AF}
    def apply(x, A):
        out = x.copy()
        for r, a in A.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        return out
    P = {c: [] for c in CF}
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); g30 = q + .30 * G8 * (~miss) * (np.nan_to_num(v) - q); g10 = q + .10 * G10 * (~miss) * (np.nan_to_num(v) - q)
            P["R27"].append(apply(q, AF) + TXo); P["R28"].append(apply(g30, UNI) + TXn); P["R27+GRU30"].append(apply(g30, AF) + TXo); P["R27+UNI"].append(apply(q, UNI) + TXo)
            P["R27+TX3new"].append(apply(q, AF) + TXn); P["R27-TX3"].append(apply(q, AF)); P["R27+GRU10s"].append(apply(g10, AF) + TXo)
            for n in ("SPL", "SPLPSY"):
                for l in (.2, .4):
                    sv = SPV[n]; ok = ~np.isnan(sv); qq = q + l * cg * ok * (np.nan_to_num(sv) - q); P[f"{n}_{l}"].append(apply(qq, AF) + TXo)
    t = M.in_temp.values
    RES[L] = dict(P=P, y=y, bk=(M.farm + "_" + (M.day // 5).astype(str)).values, seg={"all": np.ones(len(y), bool), "<=8": t <= 8, "6-8": (t > 6) & (t <= 8), "<=6": t <= 6, "8-12": (t > 8) & (t < 12), ">=12": t >= 12},
                  spl={n: rm((SPV[n] - y)[~np.isnan(SPV[n])]) for n in SPV}, base=np.mean([rm(p - y) for p in P["R27"]]))
    print(L, "done | rows", len(y), "| <=8 rows", int((t <= 8).sum()), "| SPL alone", {n: round(x, 3) for n, x in RES[L]["spl"].items()}, "| R27 RMSE %.3f" % RES[L]["base"], flush=True)
def rel(L, a, b="R27", seg="all"):
    R_ = RES[L]; m = R_["seg"][seg]
    return 100 * np.mean([rm((p - R_["y"])[m]) / rm((q - R_["y"])[m]) - 1 for p, q in zip(R_["P"][a], R_["P"][b])]) if m.any() else np.nan
pd.set_option("display.width", 220)
rows = [dict(config=c, **{f"{L}": rel(L, c) for L in RES}, **{f"CV8_{sg}": rel("CV8", c, seg=sg) for sg in ("<=8", "<=6", "8-12")}) for c in CF[1:]]
print("\n(1)+(2) vs R27 (%):"); print(pd.DataFrame(rows).round(3).to_string(index=False))
rng = np.random.default_rng(20270141); verdict = {}
for c in CF[7:]:
    R_ = RES["CV8"]; y = R_["y"]; ch = [rm(p - y) / rm(q - y) - 1 for p, q in zip(R_["P"][c], R_["P"]["R27"])]
    dd = pd.Series(np.mean([(p - y) ** 2 - (q - y) ** 2 for p, q in zip(R_["P"][c], R_["P"]["R27"])], 0)).groupby(R_["bk"]).mean().values
    pw = float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) >= 0).mean()); ok = all(x < 0 for x in ch) and pw < .00625
    for L in ("N", "EL1"):
        R2 = RES[L]; y2 = R2["y"]; ch2 = [rm(p - y2) / rm(q - y2) - 1 for p, q in zip(R2["P"][c], R2["P"]["R27"])]
        d2 = pd.Series(np.mean([(p - y2) ** 2 - (q - y2) ** 2 for p, q in zip(R2["P"][c], R2["P"]["R27"])], 0)).groupby(R2["bk"]).mean().values
        pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); ok &= not (np.mean(ch2) > 0 and pb < .025)
    verdict[c] = ok; print(f"  {c}: CV8 cells {' '.join('%+.3f' % (100 * x) for x in ch)} P(worse) {pw:.4f} -> {'ADOPT' if ok else 'FAIL'}")
print("CVJ1 new-member verdicts:", verdict)
