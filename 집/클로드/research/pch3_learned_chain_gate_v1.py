# -*- coding: utf-8 -*-
"""PCH3 (2026-10-16 집 클로드; 사용자 '새벽 몇 시까지·왜 5시인지 기준이 필요 -> 모델이 게이트처럼 데이터를 보고 정하게').  사전 고정(커밋 후 실행).  온도.
예측 사슬(PCH1·2 와 같은 사슬 조건·앵커 식)에서 시각별 가중 w 를 데이터가 정함:
  M1 시간 감쇠: w(h) = s * exp(-h / tau)                         (s, tau)
  M2 상황 게이트: w = s * exp(-z / kappa) * (해 뜬 뒤면 rho, 전이면 1)   z = |그 시각 실내 - 0시 실내|, 해 = 0..h 외부 일사 최대 > 0 (그 기록 현재·이전 입력만)
적합(A단계): PCH1 배치 CV8·N·CK2·EL1 의 사슬 기록 행에서 r = y - 기준 최종 예측, u = BETA[h] * clip(A - BIAS - x0, +-2) (4셀 평균) 로 sum (r - w u)^2 최소 격자 탐색
  (s .1~1.5, tau .5~24·무한, kappa .25~5·무한, rho 0~1).
판정(B단계): PCH2 배치 EXT10·EXT12·DIAG10 (TV3 틀) — M1·M2 각각 vs 사슬 없음: 12셀 모두 개선 & 같은 날 중복 제거(농장 x 5일 블록, 배치 평균) P(악화) < .0125 (본페로니 2).
  한계(미리 적음): 판정 배치는 PCH2 에서 이미 결과를 본 배치이고 같은 370 학습 기록을 다르게 나눈 것 - 완전 독립 아님.  서술: DAWN(0~5시 고정)·추운 사슬 기록 도움/해.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u pch3_learned_chain_gate_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "pch3_learned_chain_gate_v1.py")
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
def signals(M, key, h):
    S = pd.DataFrame({"row_id": M.row_id.values, "rec": key, "h": h}).merge(XR, on="row_id", how="left")
    S = S.sort_values(["rec", "h"]); i0 = S.groupby("rec").it_raw.transform(lambda s: s.iloc[0] if len(s) else np.nan)
    S["z"] = (S.it_raw - i0).abs().fillna(0.0); S["sun"] = (S.groupby("rec").out_rad.cummax().fillna(0) > 0).astype(float)
    S = S.set_index("row_id").reindex(M.row_id); return S.z.values, S.sun.values
def chains(recs, pos, h, FOLDOF, labelled_ok):
    AF, CH = {}, {}
    for r in recs:
        if r not in FLK or not isinstance(FLK[r][0], str) or FLK[r][1] < .8: continue
        p = FLK[r][0]
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
        out = out + TXo; U = np.zeros(len(x))
        for r in order:
            if r not in CH: continue
            ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
            if len(x0) and len(a):
                U[ix] = BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
                if wfun is not None: out[ix] += wfun(ix) * U[ix]
        return out, U
    return run
# ---- stage A: fit gate parameters on PCH1 layouts (CV8, N, CK2, EL1) ----
FIT = []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10)
    TRS, VF = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
    AF, CH = chains(recs, pos, h, VF, lambda r, p: r in TRS and p in TRS[r])
    run = make_run(AF, CH, pos, h, TXo, sorted(recs, key=lambda r: (r[:3], int(r[4:7])))); z, sun = signals(M, key, h)
    O, U = [], []
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); q = q + .30 * G8 * ok * (vv - q); o, u = run(q, None); O.append(o); U.append(u)
    m = np.isin(key, list(CH)); FIT.append(pd.DataFrame(dict(layout=L, r=(y - np.mean(O, 0))[m], u=np.mean(U, 0)[m], h=h[m], z=z[m], sun=sun[m])))
    print(L, "fit rows", int(m.sum()), "chains", len(CH), flush=True)
F = pd.concat(FIT, ignore_index=True)
def sse(w): return float(((F.r - w * F.u) ** 2).sum())
S_GRID = np.round(np.arange(.1, 1.51, .1), 2)
g1 = [(s, tau, sse(s * np.exp(-F.h / tau))) for s in S_GRID for tau in (.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 1e9)]
S1, TAU, e1 = min(g1, key=lambda x: x[2])
g2 = [(s, kap, rho, sse(s * np.exp(-F.z / kap) * np.where(F.sun > 0, rho, 1.0))) for s in S_GRID for kap in (.25, .5, 1, 1.5, 2, 3, 5, 1e9) for rho in (0, .25, .5, .75, 1)]
S2, KAP, RHO, e2 = min(g2, key=lambda x: x[3])
print(f"\n[fit] chained rows {len(F)} | SSE none {sse(0):.1f} | DAWN(0-5h, s1) {sse(np.where(F.h <= 5, 1.0, 0.0)):.1f} | M1 exp: s {S1} tau {TAU} SSE {e1:.1f} | M2 situation: s {S2} kappa {KAP} rho(sun) {RHO} SSE {e2:.1f}")
w1h = S1 * np.exp(-np.arange(24) / TAU); print("  M1 weight by hour:", " ".join(f"{x:.2f}" for x in w1h))
hb = F.assign(w2=S2 * np.exp(-F.z / KAP) * np.where(F.sun > 0, RHO, 1.0)).groupby("h").w2.mean(); print("  M2 mean weight by hour:", " ".join(f"{x:.2f}" for x in hb.reindex(range(24)).fillna(0)))
# ---- stage B: judge on EXT10, EXT12, DIAG10 (TV3 frames) ----
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3pch3"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
RES = {}
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    VFr = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    M = M.merge(VFr[["row_id", "V3LF"]], on="row_id", how="left")
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10); t = M.in_temp.values
    FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    AF, CH = chains(recs, pos, h, FO, lambda r, p: p not in HELD[FO[r]])
    run = make_run(AF, CH, pos, h, TXo, sorted(recs, key=lambda r: (r[:3], int(r[4:7])))); z, sun = signals(M, key, h)
    WF = {"DAWN": lambda ix: np.where(h[ix] <= 5, 1.0, 0.0), "M1": lambda ix: S1 * np.exp(-h[ix] / TAU), "M2": lambda ix: S2 * np.exp(-z[ix] / KAP) * np.where(sun[ix] > 0, RHO, 1.0)}
    P = {"B": []}; P.update({k: [] for k in WF})
    for b in bs:
        for fam in "AB":
            q = B2.w40(M, b, cx, fam); q = q + .30 * G8 * ok * (vv - q); P["B"].append(run(q, lambda ix: 0.0)[0])
            for k, wf in WF.items(): P[k].append(run(q, wf)[0])
    dmin = pd.Series(t).groupby(key).transform("min").values
    RES[L] = dict(P=P, y=y, key=key, dmin=dmin, ch=np.isin(key, list(CH)), bk=(M.farm + "_" + (M.day // 5).astype(str)).values); print(L, "judge frame done | chains", len(CH), flush=True)
def cellch(L, c, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"][c], R_["P"]["B"])]
rng = np.random.default_rng(20270154)
for c in ("DAWN", "M1", "M2"):
    for L in RES: print(f"  {c} {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, c))} | chained rows {100 * np.mean(cellch(L, c, RES[L]['ch'])):+.2f}%")
    per = {L: pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for p, b in zip(RES[L]["P"][c], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
    U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean())
    cells = sum((cellch(L, c) for L in RES), []); rows = []
    for L in RES:
        R_ = RES[L]; dd = np.mean([(p - R_["y"]) ** 2 - (b - R_["y"]) ** 2 for p, b in zip(R_["P"][c], R_["P"]["B"])], 0)
        S = pd.DataFrame(dict(rec=R_["key"], d=dd, ch=R_["ch"], dmin=R_["dmin"])); S = S[S.ch & (S.dmin <= 8)]; rows += list(S.groupby("rec").d.sum().values)
    tag = "" if c == "DAWN" else (" -> " + ("PASS" if all(x < 0 for x in cells) and pw < .0125 else "FAIL"))
    print(f"PCH3 {c}: 12 cells all better {all(x < 0 for x in cells)} | dedup block P(worse) {pw:.4f} | chained cold records helped {sum(x < -1e-6 for x in rows)} harmed {sum(x > 1e-6 for x in rows)}{tag}")
