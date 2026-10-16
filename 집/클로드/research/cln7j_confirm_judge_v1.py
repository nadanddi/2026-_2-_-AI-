# -*- coding: utf-8 -*-
"""CLN7J (2026-10-16 집 클로드; 5점 정리판 핵심(BASE 같은 동 어제 재연결) 확인 시험 — 사용자 '최소 유의 수준은 맞춰야').  사전 고정(커밋 후 실행).  온도.
새 배치 N·CK2·CV8·EL1 (CLN1~6 판정·선택에 안 쓴 배치; CVJ 틀, FL7 거울 연결로 K8 앵커·CH5 사슬, TX3 손 값).
기준 = 셀 BASE 를 NUM(평가형 세계 번호 순, CLN7) 으로; 안 = RS(같은 세계, FL 같은 동 어제 재연결, CLN7).  CODEX·PFN 은 두 쪽 모두 저장 CVJ 체크포인트 그대로(재연결 효과만 분리).
확인 PASS 기준(단일 안): 16셀(4배치 x 시드 2 x PFN 2) 모두 개선 & 4배치 합친 같은 날 중복 제거 블록 P(악화) < .025 & CV8·CK2 실내 <= 8도 행 변화 평균 <= +2%.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cln7j_confirm_judge_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "cln7j_confirm_judge_v1.py")
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
CK7 = os.path.join(env.LOCAL, "cln7_ckpt"); C7 = pd.concat([pd.read_pickle(os.path.join(CK7, f)) for f in sorted(os.listdir(CK7)) if f.endswith(".pkl")]).drop_duplicates("row_id")
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); RES = {}
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    M = M.merge(C7, on="row_id", how="left"); miss = int(M.base_RS_7.isna().sum()); M = M[M.base_RS_7.notna() & M.base_NUM_7.notna()].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); TRS, VF = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
    LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    AF, CH = link_plan(LK, recs, pos, h, VF, lambda r, p: r in TRS and p in TRS[r])
    P = {"B": [], "RS": []}
    for s in (7, 101):
        for fam in "AB":
            P["B"].append(assemble(B.w40(M, "base_NUM_%d" % s, "codex_R", fam), AF, CH, pos, h, TXo, order))
            P["RS"].append(assemble(B.w40(M, "base_RS_%d" % s, "codex_R", fam), AF, CH, pos, h, TXo, order))
    RES[L] = dict(P=P, y=y, t=M.in_temp.values, bk=(M.farm + "_" + (M.day // 5).astype(str)).values)
    print(f"{L}: rows {len(M)} (missing CLN7 {miss}) | BASE alone NUM {np.mean([rm(M['base_NUM_%d' % s].values - y) for s in (7, 101)]):.4f} RS {np.mean([rm(M['base_RS_%d' % s].values - y) for s in (7, 101)]):.4f}", flush=True)
def cellch(L, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((q - y)[m]) - 1 for p, q in zip(R_["P"]["RS"], R_["P"]["B"])] if m.any() else [np.nan]
rng = np.random.default_rng(20270190); cells = []
for L in RES:
    tt = np.nan_to_num(RES[L]["t"], nan=99); c = cellch(L); cells += c
    print(f"  RS vs NUM | {L}: cells {' '.join('%+.3f' % (100 * x) for x in c)} | <=6C {100 * np.nanmean(cellch(L, tt <= 6)):+.2f}% | <=8C {100 * np.nanmean(cellch(L, tt <= 8)):+.2f}%")
per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (q - RES[L]["y"]) ** 2 for p, q in zip(RES[L]["P"]["RS"], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean())
cold = np.nanmean([np.nanmean(cellch(L, np.nan_to_num(RES[L]["t"], nan=99) <= 8)) for L in ("CV8", "CK2")])
ok = all(x < 0 for x in cells) and pw < .025 and cold <= .02
print(f"CLN7J confirmation: 16 cells all better {all(x < 0 for x in cells)} (mean {100 * np.mean(cells):+.3f}%, max {100 * max(cells):+.3f}%) | dedup P(worse) {pw:.4f} | CV8+CK2 <=8C {100 * cold:+.2f}% (guard <= +2%) -> {'PASS' if ok else 'FAIL'}")
