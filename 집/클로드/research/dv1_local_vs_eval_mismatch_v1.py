# -*- coding: utf-8 -*-
"""DV1 (2026-10-15 집 클로드; 사용자 '왜 로컬 검증과 평가가 반대로 나왔는지 분석').  서술 진단(판정 없음), 실행 전 커밋.  온도.
17회차(로컬 +2~3% 예상 → 평가 −0.9%), 18회차(로컬 −1.9~−3% 예상 → 평가 +2.2%) — 리더보드 값으로 평가 정답을 역산하지 않음(rule5).
(A) 표본 잡음: ASM2 의 NEW vs R27(= 28 vs 27 온도) 행 예측을 N·CK2·EL1·EXT10 에서 모아, 평가처럼 60기록(농장당 30)을 2만 번 뽑아
    RMSE 변화율 분포 → 평균·95% 구간·P(변화 ≥ +2.18%).  2차 기록만 뽑는 판도 같이.
(B) 부품 작동 조건: 평가 입력(test_X)과 각 배치에서 GRU 게이트(8/10) > 0 행 비율·평균 게이트, TX3 새 값 > 0 행 비율·평균 더함, 실내 ≤ 8 / 8~12℃ 비율,
    2차 비율, 하루 실내 최저 분포.
(C) 어제 연결: 평가 기록 FL/FLN 링크 확률 분포 vs 학습표(검증에 쓴) 확률 분포, 링크 있는 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dv1_local_vs_eval_mismatch_v1.py
"""

import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])
__file__ = os.path.join(HERE, "dv1_local_vs_eval_mismatch_v1.py")
import common
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
ELF = dict(W.worlds()[7])["EL1"]; CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
def gate(M, lo_g, lo_h):
    g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - lo_g) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
    mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - lo_h) / 2, 0, 1)); return g_ * h_
def tx3(M, key, c, T0, e0):
    cm = np.nan_to_num(np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, M.hour.values)]), nan=0)
    return c * np.clip((cm - e0) / .4, 0, 1) * np.nan_to_num(np.clip((T0 - M.in_temp.values) / 2, 0, 1), nan=0.0)
RES = {}
for L, ckdir, folds, mk in (("N", O1.CKB, LF["N"], O1.masks), ("CK2", O1.CKB, LF["CK2"], O1.masks), ("EL1", CKR, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckdir, f)) for f in sorted(os.listdir(ckdir)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(O1.CKG, f)) for f in sorted(os.listdir(O1.CKG)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; miss = np.isnan(v); TX_old, TX_new = tx3(M, key, .25, 10, .8), tx3(M, key, .40, 12, .7); G8 = gate(M, 8, 10)
    TRS = {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
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
    P = {"R27": [], "NEW": [], "NEW_oldTX3": []}
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); x = q + .30 * G8 * (~miss) * (np.nan_to_num(v) - q)
            P["R27"].append(apply(q, AF) + TX_old); P["NEW"].append(apply(x, UNI) + TX_new); P["NEW_oldTX3"].append(apply(x, UNI) + TX_old)
    RES[L] = (P, y, (L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values, (M.in_temp <= 8).values, M.day.values >= 179); print(L, "done", flush=True)
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3asm2"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
for L in ("EXT10", "EXT12"):
    d, pre_, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values
    VF = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    v = M[["row_id"]].merge(VF[["row_id", "V3LF"]], on="row_id", how="left").V3LF.values; miss = np.isnan(v); G8 = gate(M, 8, 10); TX_old, TX_new = tx3(M, key, .25, 10, .8), tx3(M, key, .40, 12, .7)
    P = {"R27": [], "NEW": [], "NEW_oldTX3": []}
    for b in bs:
        for fam in "AB":
            q = B2.w40(M, b, cx, fam); x = q + .30 * G8 * (~miss) * (np.nan_to_num(v) - q); P["R27"].append(q + TX_old); P["NEW"].append(x + TX_new); P["NEW_oldTX3"].append(x + TX_old)
    RES[L] = (P, y, (M.farm + "_" + (M.day // 5).astype(str)).values, (M.in_temp <= 8).values, M.day.values >= 179); print(L, "done", flush=True)
rng = np.random.default_rng(20270140); POOL = []
for L in ("N", "CK2", "EL1", "EXT10"):
    P, y, bk, cold, p2 = RES[L]; Pn = np.mean(P["NEW"], 0); Pr = np.mean(P["R27"], 0)
    F = pd.DataFrame({"L": L, "bk": bk, "y": y, "new": Pn, "ref": Pr, "p2": p2, "cold": cold}); F["rec"] = [L + "|" + b for b in bk]
    POOL.append(F)
POOL = pd.concat(POOL, ignore_index=True)
key = pd.Series(range(len(POOL))); POOL["r"] = POOL.groupby(["L", "bk"]).ngroup()
def boot(sub, n=60, B=20000):
    g = sub.groupby("r").agg(sn=("new", lambda x: 0), farm=("bk", "first")); recs = sub.r.unique()
    SN = sub.assign(en=(sub.new - sub.y) ** 2, er=(sub.ref - sub.y) ** 2).groupby("r")[["en", "er"]].sum(); cnt = sub.groupby("r").size()
    idx = rng.integers(0, len(SN), (B, n)); en = SN.en.values[idx].sum(1); er = SN.er.values[idx].sum(1)
    return 100 * (np.sqrt(en / er) - 1)
for nm, sub in (("all layouts", POOL), ("pass-2 rows only", POOL[POOL.p2]), ("EL1+CK2 (eval-like)", POOL[POOL.L.isin(["EL1", "CK2"])])):
    d = boot(sub); print(f"(A) {nm}: 60-record samples (of 5-day blocks as units): mean {d.mean():+.2f}%, 95% [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}], P(>= +2.18%) {np.mean(d >= 2.18):.3f}, P(> 0) {np.mean(d > 0):.3f}")
# (B) component action conditions
def cond(M, cm_key):
    t = M.in_temp.values; key2 = (M.farm + "_" + M.day.map("{:03d}".format)).values
    cm = np.nan_to_num(np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key2, M.hour.values)]), nan=0) if cm_key is None else cm_key
    G8 = gate(M, 8, 10); TX = .40 * np.clip((cm - .7) / .4, 0, 1) * np.nan_to_num(np.clip((12 - t) / 2, 0, 1), nan=0.0)
    mm = M.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"]).in_temp.cummin().reindex(M.index)
    return dict(rows=len(M), gru_on=float((G8 > 0).mean()), gru_mean=float(G8.mean()), tx3_on=float((TX > 1e-9).mean()), tx3_mean=float(TX.mean()),
                cold8=float((t <= 8).mean()), band8_12=float(((t > 8) & (t < 12)).mean()), pass2=float((M.day >= 179).mean()), daymin_med=float(mm.median()), nh_cov=float((cm > 0).mean()))
rowsB = []
for L, ckdir in (("N", O1.CKB), ("CK2", O1.CKB), ("EL1", CKR)):
    M = pd.concat([pd.read_csv(os.path.join(ckdir, f)) for f in sorted(os.listdir(ckdir)) if f.startswith(L + "_")], ignore_index=True)
    rowsB.append(dict(set=L, **cond(M, None)))
X = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
E27 = pd.read_csv(os.path.join(env.ROOT, u"제출", u"미제출", u"submission_28_5점모델v2_GRU확대_TX3다듬기_EC약한앵커_가설검정", u"팜모니_정형데이터_재현패키지_28", "EC", "data", "test_X.csv")) if False else None
NHE = pd.read_csv(os.path.join(env.ROOT, u"집", u"클로드", "research", "local", "dv1_ec_nh_eval.csv")) if os.path.exists(os.path.join(env.LOCAL, "dv1_ec_nh_eval.csv")) else None
X = X.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
if NHE is not None:
    NHE = NHE.set_index("row_id").sub_ec.reindex(X.row_id).values; cmE = pd.Series(NHE).groupby((X.farm + X.day.astype(str)).values).transform(lambda s: s.expanding().mean()).values
else: cmE = None
rowsB.append(dict(set="EVAL", **cond(X, cmE if cmE is not None else np.zeros(len(X)))))
pd.set_option("display.width", 220); print("(B) component action conditions:"); print(pd.DataFrame(rowsB).round(3).to_string(index=False))
# (C) link probabilities
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLE = pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv"))
FNT = pd.read_csv(os.path.join(env.LOCAL, "fln1_train_links.csv"))
print("(C) FL link p: train(validation) median %.2f, share p>=.8 %.2f, p>=.5 %.2f | eval median %.2f, p>=.8 %.2f, p>=.5 %.2f" % (FLT.p.median(), (FLT.p >= .8).mean(), (FLT.p >= .5).mean(), FLE.p.median(), (FLE.p >= .8).mean(), (FLE.p >= .5).mean()))
print("    eval FL prev is labelled training: %.2f | prev is earlier eval record: %.2f | cross-farm: %.2f" % (FLE.prev_is_labelled.mean(), (~FLE.prev_is_labelled.astype(bool)).mean(), FLE.xf.mean()))
print("    FLN train p>=.5 %.2f (p_none median %.2f)" % ((FNT.p >= .5).mean(), FNT.p_none.median()))
