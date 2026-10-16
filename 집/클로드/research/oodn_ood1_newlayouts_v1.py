# -*- coding: utf-8 -*-
"""OODN (2026-10-16 집 클로드; 사용자 '효과 제대로 검증했나, 아까처럼 바뀌는 거 아니냐' -> 35 의 OOD1 을 판정에 안 쓴 배치 N·CK2·CV8(평가형 추운 블록)·EL1 에서 확인).
사전 고정(커밋 후 실행).  온도.  CVJ 틀(저장 CVJ 체크포인트, FL7 거울 K8·CH5, 손 TX3), OOD1 곡선은 TV3 에서 학습된 값 그대로(-1.32, .29, 1.50), 학습 최저 = 그 폴드 학습 기록.
범위: 전체 행 / 2차 행(일 >= 179).  확인 규칙(고정): 2차 범위 16셀 중 14셀 이상 개선 & 셀 최대 <= +.5% & P(악화) < .05 & CV8 2차 셀 모두 개선 -> 확인; 아니면 35 대신 34.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u oodn_ood1_newlayouts_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "oodn_ood1_newlayouts_v1.py")
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

def tx3x(M, key, c, T0, e0, cold_hand):
    a = tx3(M, key, c, T0, e0)
    if cold_hand: a = np.where(np.nan_to_num(M.in_temp.values, nan=99) > 8, a, tx3(M, key, .25, 10, .8))
    return a

TH = np.array([-1.32, .29, 1.50])     # OOD1 curve fitted on TV3 EXT12 + DIAG10 (7.025 'all'), applied unchanged
def sfun(Z, miss): s = 1 / (1 + np.exp(-np.clip(Z @ TH, -30, 30))); return np.where(miss, 0.0, s)
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); RES = {"all": {}, "p2": {}}
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TRS, VF, TMIN = {}, {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        mins = {f: float(np.nanmin(lab.in_temp.values[tm & (lab.farm == f).values])) for f in ("F13", "F47")}
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k; TMIN[r] = mins[r[:3]]
    LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    AF, CH = link_plan(LK, recs, pos, h, VF, lambda r, p: r in TRS and p in TRS[r])
    t = M.in_temp.values.astype(float); tt = np.nan_to_num(t, nan=15.0); cm = pd.Series(tt).groupby(key).cummin().values; tmin = np.array([TMIN[r] for r in key]); miss = np.isnan(t)
    Z = np.column_stack([np.ones(len(M)), np.clip(tmin - tt, -20, 10) / 2, np.clip(tmin - cm, -20, 10) / 2]); s = sfun(Z, miss)
    T0h = tx3(M, key, .25, 10, .8); P = {"B": [], "OOD": []}
    for sd in O1.SEEDS:
        for fam in "AB":
            x = B.w40(M, "base_R_%d" % sd, "codex_R", fam); cx = np.where(np.isfinite(M.codex_R.values), M.codex_R.values, x)
            P["B"].append(assemble(x, AF, CH, pos, h, T0h, order)); P["OOD"].append(assemble(x + s * (cx - x), AF, CH, pos, h, T0h, order))
    for sc, sel in (("all", np.ones(len(M), bool)), ("p2", M.day.values >= 179)):
        RES[sc][L] = dict(P={k: [v[sel] for v in vv] for k, vv in P.items()}, y=y[sel], t=t[sel], bk=(M.farm + "_" + (M.day // 5).astype(str)).values[sel])
    print(L, "rows", len(M), "| rows s > .1:", int((s > .1).sum()), "| pass-2 rows", int((M.day >= 179).sum()), "| pass-2 rows s > .1", int(((s > .1) & (M.day.values >= 179)).sum()), flush=True)
rng = np.random.default_rng(20270220)
for sc in ("all", "p2"):
    cells = []
    for L, R_ in RES[sc].items():
        y = R_["y"]; c = [rm(p - y) / rm(q - y) - 1 for p, q in zip(R_["P"]["OOD"], R_["P"]["B"])]; cells += c
        tt = np.nan_to_num(R_["t"], nan=99); m8 = tt <= 8
        c8 = np.mean([rm((p - y)[m8]) / rm((q - y)[m8]) - 1 for p, q in zip(R_["P"]["OOD"], R_["P"]["B"])]) if m8.any() else np.nan
        print(f"  [{sc}] OOD1 vs base | {L}: cells {' '.join('%+.3f' % (100 * x) for x in c)} | <=8C {100 * c8:+.2f}%")
    per = {L: pd.Series(np.mean([(p - R_["y"]) ** 2 - (q - R_["y"]) ** 2 for p, q in zip(R_["P"]["OOD"], R_["P"]["B"])], 0)).groupby(R_["bk"]).mean() for L, R_ in RES[sc].items()}
    U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean())
    print(f"OODN [{sc}]: 16 cells better {sum(x < 0 for x in cells)}/16 (mean {100 * np.mean(cells):+.3f}%, max {100 * max(cells):+.3f}%) | dedup P(worse) {pw:.4f}")
print("pre-registered rule: CONFIRM if pass-2 scope has >= 14/16 cells better, max cell <= +.5%, P < .05 AND CV8 pass-2 cells all better; else 35 NOT submitted (34 instead)")
