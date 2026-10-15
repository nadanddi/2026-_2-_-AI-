# -*- coding: utf-8 -*-
"""CVJ4 (2026-10-15 집 클로드; 사용자 '배지-실내공기만 하지 말고 다른 것들도').  사전 고정(커밋 후 실행).  온도.
CVJ3(6.966): 배지-공기 차이 GAM - 6~8도 는 일관 개선, <= 6도 는 시기 따라 뒤집힘.  기준선(목표)을 넓혀 비교:
  A 배지 - in_temp | B 배지 - ph_in_temp_3(3시간 지연 평균, 열관성) | C 배지 - ph_in_temp_24(하루 지연, 큰 열용량)
  D 위치 비율 rho = (배지 - 외기)/(실내 - 외기) (|실내 - 외기| < 1도 행은 학습·적용 제외), 예측 = 외기 + rho_hat*(실내 - 외기)
  E 배지 - 오늘 0..h 실내 평균
  모두 서늘한 학습 행(실내 <= 12도, CVJ3 G2 와 같음)으로 GAM(FE + PSY, 스플라인 직선 외삽, Ridge 10) 적합, lambda .5.
게이트 2종: T10 = clip((10 - in_temp)/2, 0, 1) | T10m = T10 이되 그 폴드 같은 농장 학습 실내온도 최저보다 낮으면 0(학습 범위 밖 외삽 금지, 원칙형 경계).
10안(5 x 2). 기준 B = 셀 -> GRU30 -> FL K8 -> TX3 27번 값.
고르기 CV8 폴드 0~2, 판정 폴드 3~5(alpha .025): 4셀 모두 개선 & 블록 P < .025 & 뒤쪽 <= 6도 악화 아님 & N·CK2·EL1 안전.  서술: 10안 표.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cvj4_reference_targets_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "cvj4_reference_targets_v1.py")
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
TG = ("A", "B", "C", "D", "E"); GATES = ("T10", "T10m"); OPTS = [f"{a}_{g}" for a in TG for g in GATES]
L0 = L0.sort_values(["farm", "day", "hour"]); L0["cm_in"] = L0.groupby(["farm", "day"]).in_temp.transform(lambda z: z.expanding().mean()); L0 = L0.sort_index()
REFC = {"A": "in_temp", "B": "ph_in_temp_3", "C": "ph_in_temp_24", "E": "cm_in"}
def tgt_fit_predict(tm, vm, tgk):
    feats = FE + PSY; a = tm & (L0.in_temp.values <= 12) & np.isfinite(L0.in_temp.values); out = pd.Series(np.nan, index=L0.row_id.values[vm])
    if tgk == "D":
        den = (L0.in_temp - L0.out_temp).values; a = a & (np.abs(den) >= 1); tgv = (L0.sub_temp.values - L0.out_temp.values) / np.where(np.abs(den) >= 1, den, np.nan)
    else:
        ref = L0[REFC[tgk]].values; tgv = L0.sub_temp.values - ref
    a = a & np.isfinite(tgv)
    Xt = L0.loc[a, feats].to_numpy(float); Xv = L0.loc[vm, feats].to_numpy(float); med = np.nanmedian(Xt, 0); Xt = np.where(np.isnan(Xt), med, Xt); Xv = np.where(np.isnan(Xv), med, Xv)
    m = make_pipeline(SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), StandardScaler(), Ridge(alpha=10.0)); m.fit(Xt, tgv[a], ridge__sample_weight=WB[a]); pr = m.predict(Xv)
    if tgk == "D":
        den_v = (L0.in_temp - L0.out_temp).values[vm]; val = np.where(np.abs(den_v) >= 1, L0.out_temp.values[vm] + pr * den_v, np.nan)
    else:
        val = L0[REFC[tgk]].values[vm] + pr
    out.loc[L0.row_id.values[vm]] = val; return out
RES = {}
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; miss = np.isnan(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10); t = M.in_temp.values
    TRS = {}; SP = {a: [] for a in TG}; MINS = []
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
        for a in TG: SP[a].append(tgt_fit_predict(tm, vm, a))
        mn = pd.Series(np.nan, index=L0.row_id.values[vm])
        for f in ("F13", "F47"):
            b = vm & (L0.farm == f).values; mn.loc[L0.row_id.values[b]] = np.nanmin(L0.in_temp.values[tm & (L0.farm == f).values])
        MINS.append(mn)
    SPV = {a: pd.concat(s_).groupby(level=0).first().reindex(M.row_id).values for a, s_ in SP.items()}; TMIN = pd.concat(MINS).groupby(level=0).first().reindex(M.row_id).values
    AF = {r: float(ST[IDX[FLK[r][0]], 23]) for r in recs if r in FLK and r in TRS and FLK[r][1] >= .8 and FLK[r][0] in TRS[r] and np.isfinite(ST[IDX[FLK[r][0]], 23])
          and abs(IN_[IDX[r], 0, 0] - IN_[IDX[FLK[r][0]], 23, 0]) <= 1.5}
    def apply(x, AF=AF, pos=pos, h=h):
        out = x.copy()
        for r, a in AF.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        return out
    G = []
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); G.append(q + .30 * G8 * (~miss) * (np.nan_to_num(v) - q))
    t10 = np.nan_to_num(np.clip((10 - t) / 2, 0, 1), nan=0.0); GT = {"T10": t10, "T10m": np.where(np.nan_to_num(t, nan=99) < TMIN, 0.0, t10)}
    P = {"B": [apply(g) + TXo for g in G]}
    for a in TG:
        for gn in GATES:
            sv = SPV[a]; ok = ~np.isnan(sv); P[f"{a}_{gn}"] = [apply(g + .5 * GT[gn] * ok * (np.nan_to_num(sv) - g)) + TXo for g in G]
    RES[L] = dict(P=P, y=y, t=t, bk=(M.farm + "_" + (M.day // 5).astype(str)).values, fold=M.fold.values if "fold" in M else np.zeros(len(M), int))
    print(L, "done | rows below fold training min:", int((np.nan_to_num(t, nan=99) < TMIN).sum()), flush=True)
def rel(L, c, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return 100 * np.mean([rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"][c], R_["P"]["B"])]) if m.any() else np.nan
C8R = RES["CV8"]; early = C8R["fold"] <= 2; late = ~early; t8 = C8R["t"]
T_ = pd.DataFrame([dict(opt=n, early=rel("CV8", n, early), late=rel("CV8", n, late), late_le6=rel("CV8", n, late & (t8 <= 6)), late_6_8=rel("CV8", n, late & (t8 > 6) & (t8 <= 8)),
                        early_le6=rel("CV8", n, early & (t8 <= 6)), N=rel("N", n), CK2=rel("CK2", n), EL1=rel("EL1", n)) for n in OPTS])
pd.set_option("display.width", 220); print(T_.round(3).to_string(index=False))
score = {n: np.mean([rm((p - C8R["y"])[early]) for p in C8R["P"][n]]) for n in OPTS}; best = min(OPTS, key=lambda n: (round(score[n], 6), OPTS.index(n)))
print("selected on CV8 early folds:", best)
rng = np.random.default_rng(20270145); y = C8R["y"]
ch = [rm((p - y)[late]) / rm((b - y)[late]) - 1 for p, b in zip(C8R["P"][best], C8R["P"]["B"])]
dd = pd.Series(np.mean([((p - y) ** 2 - (b - y) ** 2)[late] for p, b in zip(C8R["P"][best], C8R["P"]["B"])], 0)).groupby(C8R["bk"][late]).mean().values
pw = float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) >= 0).mean()); le6 = rel("CV8", best, late & (t8 <= 6)); ok = all(x < 0 for x in ch) and pw < .025 and not (le6 > 0); safe = {}
for L in ("N", "CK2", "EL1"):
    R_ = RES[L]; y2 = R_["y"]; c2 = [rm(p - y2) / rm(b - y2) - 1 for p, b in zip(R_["P"][best], R_["P"]["B"])]
    d2 = pd.Series(np.mean([(p - y2) ** 2 - (b - y2) ** 2 for p, b in zip(R_["P"][best], R_["P"]["B"])], 0)).groupby(R_["bk"]).mean().values
    pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); safe[L] = not (np.mean(c2) > 0 and pb < .025)
print(f"CVJ4 {best}: CV8 late 4 cells {' '.join('%+.3f' % (100 * x) for x in ch)} | P(worse) {pw:.4f} | late <=6 {le6:+.2f}% | safety {safe} -> {'PASS' if ok and all(safe.values()) else 'FAIL'}")
