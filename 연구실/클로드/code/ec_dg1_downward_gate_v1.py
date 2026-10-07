# -*- coding: utf-8 -*-
"""DG1 — HG3 의 반대판: 엄격 도메인 보호 '과대예측 일반 날 하향' (실행 전 고정) — 2026-10-07 연구실 클로드
근거: 평가 점수는 2차 일반 날 하루 수준이 지배(6.394; 리더보드 .1384 ≈ 2차 일반 날 검증값). 일반 날 오차는 소수 날 집중, 대표 유형은
      '밀폐된 일반 날을 높게 예측'(ER1 6.358: 과대 28일 DIAG10 SSE 22%). HG3(6.329/6.422)는 올리는 쪽만 다뤄 일반 날 변화 0.
      SG2 는 이웃 차가 .30 넘으면 막혀 크게 과대된 날을 못 고침.
규칙(고정, HG3 와 대칭; 2차 행만, SG2 적용 뒤):
  pm_h ≥ .9 (모델 0..h 누적평균이 높음) AND 가장 가까운 앵커 두 기록 a1, a2 모두 < .8 AND 시각별 인과 S_low ≥ +1.0 (도메인 점수가 'EC 낮음' 쪽)
  → pred = p + 0.5 · (mean(a1, a2) − pm_h).   (앵커·S_low·pm 정의는 HG3 correction 과 동일, 문턱만 반대)
기준선: 9회차 근사 구성 CUR = clip(0.8 R3_DP1 + 0.2 PFN) → SG2 (RV2 와 같은 WT1 OOF, 시드 47/1414/6464). 비교: DG1 출력 vs SG2 출력.
[고정 기준 — 조건부 점검, 통과 ≠ 채택]
 (a) 시드 3 × {DIAG10 2차, P2LOO, EL1} 9칸 모두 2차 시간 RMSE 개선
 (b) 같은 9칸에서 고EC 날(정답 하루평균 ≥ 1) RMSE 가 나빠지지 않음(Δ ≤ 1e-9)
 (c) 일반 날 9칸 모두 개선
 함께: 바뀐 날 목록·날 단위 개선/악화 수·P(worse) 와 구조적 하한, 평가 60 진단(제출14 예측 위 발동 날, 정답 없음).
비고: 연구용 correction 은 잠금 날을 앵커에서 제외(제출 sg2post 와 다름). 같은 46일 자료라 통과해도 사후 탐색 성격(HG3 와 같은 한계).
"""
import os, sys, json, importlib.util
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research")
sys.path.insert(0, RES)
import env  # noqa
import numpy as np, pandas as pd
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("hg3", os.path.join(RES, "ec3_HG3_strict_guard_gate_v1.py")); hg3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg3)
R = os.path.dirname(os.path.abspath(__file__)); DC = os.path.join(R, "..", "local", "drive_copy")
raw, full, lab, lock, signatures, fds = hg3.p3.prepare()
Rr, WV, hrs, SIG, DM = hg3.prepare_structure()
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
SGN = hg3.SGN

def correction_dg(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal, DM, ST):
    """hg3.correction 과 같은 SG2 + 대칭 하향 게이트. 반환 (SG2 출력, DG1 출력)."""
    out = frame[pcol].values.copy(); dg = frame[pcol].values.copy()
    roles = R.set_index(["farm", "day"]).role
    full_cache = {}
    def twin_date(f, d, h):
        E = [e for e in R[R.farm == f].day if (f, e) in ref]
        cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None
    def full_date(f, d):
        if (f, d) in ref:
            return cal[(f, d)]
        if (f, d) in full_cache:
            return full_cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            days = sorted(R[R.farm == f].day); i = days.index(d)
            t = (full_date(f, days[i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        full_cache[(f, d)] = t
        return t
    for (f, d), idx in frame.groupby(["farm", "day"]).groups.items():
        if d < 179:
            continue
        G = R[R.farm == f]
        G = G[[(f, e) in ref and (f, e) in ec.index and (f, e) not in lock for e in G.day]]
        calc = np.array([cal[(f, e)] for e in G.day])
        days = sorted(R[R.farm == f].day); i = days.index(d)
        rows = frame.loc[idx].sort_values("hour")
        cum = rows[pcol].expanding().mean().values
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None:
                cq = full_date(f, days[i - 1]) + 0.1
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any():
                continue
            Gm = G[m]; S = SIG[h]
            RS = S.loc[[(f, e) for e in R[R.farm == f].day if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            o = np.argsort(dist)
            a1 = ec[(f, Gm.iloc[o[0]].day)]; pm = cum[k]
            a2 = ec[(f, Gm.iloc[o[1]].day)] if len(o) > 1 else np.nan
            j = frame.index.get_loc(ii)
            if abs(a1 - pm) <= .30:
                out[j] = rr[pcol] + 0.5 * (a1 - pm)
            dg[j] = out[j]
            mu_, sd_ = ST[(f, h)]
            slow = float(np.nansum(SGN * (DM.loc[(f, d, h)].values - mu_) / sd_))
            if pm >= .9 and a1 < .8 and np.isfinite(a2) and a2 < .8 and slow >= 1.0:
                dg[j] = rr[pcol] + 0.5 * ((a1 + a2) / 2 - pm)
    return out, dg

O = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv"))
SEEDS = (47, 1414, 6464)
D10 = {i: vd for name, i, vd in fds if name == "DIAG10"}
CKD = os.path.join(R, "..", "local", "dg1_ckpt"); os.makedirs(CKD, exist_ok=True)
recs = []
for (vn, k), G in O.groupby(["validator", "validation_fold"]):
    G = G.copy()
    if not (G.day >= 179).any():
        continue
    ckf = os.path.join(CKD, "%s_%d.csv" % (vn, k))
    if os.path.exists(ckf):
        recs.append(pd.read_csv(ckf)); continue
    vd = D10[k] if vn == "DIAG10" else set(zip(G.farm, G.day))
    ref = {x for x in labset if x not in vd}
    cal = hg3.ref_calendar(Rr, WV, ref); ST = hg3.s_low_table(DM, ref)
    part = []
    for s in SEEDS:
        r3 = .6 * G["et_%d" % s] + .3 * G["lgb_%d" % s] + .1 * G["mlp_%d" % s]
        G["x"] = np.clip(.8 * r3 + .2 * G.pfn, G.lo, G.hi).values
        sgx, dgx = correction_dg(G, "x", vd, lockd, ec, Rr, WV, hrs, SIG, ref, cal, DM, ST)
        part.append(pd.DataFrame(dict(validator=vn, fold=k, row_id=G.row_id.values, farm=G.farm.values, day=G.day.values, hour=G.hour.values,
                                      y=G.sub_ec.values, seed=s, sg=np.clip(sgx, G.lo, G.hi), dg=np.clip(dgx, G.lo, G.hi))))
    pd.concat(part).to_csv(ckf, index=False); recs.append(pd.concat(part))
    print("  %s/%d 완료" % (vn, k), flush=True)
P = pd.concat(recs, ignore_index=True); P = P[P.day >= 179].copy()
P["dm"] = P.groupby(["validator", "fold", "seed", "farm", "day"]).y.transform("mean")
RV2 = pd.read_csv(os.path.join(R, "..", "results", "ec_rv2_preds_v1.csv"))
chk = P.merge(RV2[RV2.tag == "CUR"][["validator", "fold", "row_id", "seed", "sg"]], on=["validator", "fold", "row_id", "seed"], suffixes=("", "_rv2"))
dmax = float(np.abs(chk.sg - chk.sg_rv2).max()); print("정합성: RV2 SG2 출력과 최대차 %.2e (행 %d)" % (dmax, len(chk))); assert dmax < 1e-9
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
ok_all = ok_high = ok_norm = True
for vn, nm in (("DIAG10", "DIAG10 2차"), ("P2LOO", "P2LOO"), ("EL1", "EL1")):
    Q = P[P.validator == vn]; cells = []
    for s in SEEDS:
        q = Q[Q.seed == s]; a, b = rm(q.sg - q.y), rm(q.dg - q.y)
        n0, n1 = rm((q.sg - q.y)[q.dm < 1]), rm((q.dg - q.y)[q.dm < 1])
        hm = q.dm >= 1; h0, h1 = rm((q.sg - q.y)[hm]), rm((q.dg - q.y)[hm])
        nhi = q[hm & (np.abs(q.dg - q.sg) > 1e-12)].groupby(["farm", "day"]).ngroups
        cells.append("s%d 전체 %.4f→%.4f (%+.1f%%) | 일반 %.4f→%.4f | 고EC Δ%+.1e (바뀐 고EC 날 %d)" % (s, a, b, 100 * (b / a - 1), n0, n1, h1 - h0, nhi))
        ok_all &= b < a; ok_norm &= n1 < n0; ok_high &= (h1 - h0) <= 1e-9
    print("  %-10s\n    %s" % (nm, "\n    ".join(cells)))
for vn in ("P2LOO", "DIAG10"):
    Z = P[P.validator == vn].groupby("row_id").agg(sg=("sg", "mean"), dg=("dg", "mean"), y=("y", "first"), farm=("farm", "first"), day=("day", "first"))
    Z["cl"] = Z.farm + "_" + (Z.day // 5).astype(str)
    dd = ((Z.dg - Z.y) ** 2 - (Z.sg - Z.y) ** 2).groupby(Z.cl).agg(["sum", "count"]); sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(20261007).integers(0, len(sm), (20000, len(sm))); pw = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    ch = int((np.abs(sm) > 1e-12).sum())
    dday = Z.groupby(["farm", "day"]).apply(lambda g: ((g.dg - g.y) ** 2).sum() - ((g.sg - g.y) ** 2).sum()); dday = dday[np.abs(dday) > 1e-12]
    print("  %-7s P(worse) %.4f (바뀐 묶음 %d/%d, 구조적 하한 %.4f) | 바뀐 날 %d: 개선 %d·악화 %d" % (vn, pw, ch, len(sm), (1 - ch / len(sm)) ** len(sm), len(dday), int((dday < 0).sum()), int((dday > 0).sum())))
ch = P[np.abs(P.dg - P.sg) > 1e-12].groupby(["farm", "day"]).agg(y=("dm", "first")).round(3)
print("  바뀐 날(정답 하루평균):", ", ".join("%s_%d(%.2f)" % (f, d, v) for (f, d), v in ch.y.items()))
print("\n[DG1 조건부 점검] (a) 9칸 개선 %s, (b) 고EC 안 나빠짐 %s, (c) 일반 9칸 개선 %s → %s (통과 ≠ 채택)" % (ok_all, ok_high, ok_norm, "통과" if ok_all and ok_high and ok_norm else "미통과"))
