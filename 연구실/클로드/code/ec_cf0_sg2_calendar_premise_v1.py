# -*- coding: utf-8 -*-
"""CF0 — SG2 달력 위치 보정 전제 검산 (진단, 정답은 '실제 날짜' 오라클·채점에만) — 2026-10-07 연구실 클로드
SG2 는 1차 쌍둥이 없는 2차 기록(C3 등)의 달력을 '앞 기록 + .1' 로 매김(질의·참조 모두).
Q1 C3 정답 기록: SG2 달력 vs 실제 달력(정답 연속 진짜 앞날 t 의 같은 ID 달력 + 1; t 가 다른 ID 면 t 의 날짜 묶음 중 같은 ID 기록의 달력) 차이.
Q2 오라클 상한: C3 질의 행의 cq 를 실제 달력으로 바꾸고(참조 쪽 C3 정답 기록 달력도 실제 달력으로) SG2 보정을 다시 계산 → DIAG10/DIAG10y/EL1 2차 행 RMSE.
   (참조 달력의 실제값은 다른 기록 정답으로 만든 것 = 학습 자료라 원칙상 허용 가능 / 질의 쪽 실제값은 오라클)
"""
import os, sys, json, importlib.util
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research")
sys.path.insert(0, RES)
import env  # noqa
import numpy as np, pandas as pd
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(RES, "ec3_SG2_reference_knn_level_v1.py")); sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(RES, "local")
Rr, WV, hrs, SIG = sg2.prepare_structure()
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
ec = Y[Y.farm.isin(["F13", "F47"])].groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
LOCK = {(s["farm"], int(s["day"])) for s in json.load(open(os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json"), encoding="utf-8"))["selected"]}
cls = pd.read_csv(os.path.join(R, "..", "results", "ec_dt0_date_class_v1.csv")).set_index(["farm", "day"]).cls
V2 = pd.read_csv(os.path.join(R, "..", "results", "ec_dt1_c3_premise_v2.csv")).set_index(["farm", "day"])
calA = sg2.ref_calendar(Rr, WV, labset)          # 전체 정답 참조 달력(SG2 방식)
def twins(k):
    return [x for x in WV.index if x != k and np.sqrt(np.nanmean((WV.loc[x].values - WV.loc[k].values) ** 2)) <= .05]
def true_cal(b, cal):
    t = (V2.loc[b, "t_farm"], int(V2.loc[b, "t_day"]))
    cands = [t] if t[0] == b[0] else []
    cands += [x for x in twins(t) if x[0] == b[0]]
    cs = [cal[x] for x in cands if x in cal]
    return (np.mean(cs) + 1) if cs else np.nan
rows = []
for b in V2.index:
    if cls.get(b) != "C3":
        continue
    rows.append(dict(farm=b[0], day=b[1], sg2cal=calA.get(b, np.nan), truecal=true_cal(b, calA), t=(V2.loc[b, "t_farm"], int(V2.loc[b, "t_day"])), tcost=V2.loc[b, "t_cost"]))
Q1 = pd.DataFrame(rows); Q1["err"] = Q1.sg2cal - Q1.truecal
print("Q1 C3 정답 %d: |SG2 달력 − 실제| 중앙 %.1f, ≤3 비율 %.2f, 실제 달력 없음 %d" % (len(Q1), Q1.err.abs().median(), (Q1.err.abs() <= 3).mean(), Q1.truecal.isna().sum()))
print(Q1.round(2).to_string(index=False))
TRUE = {(r.farm, r.day): r.truecal for r in Q1.itertuples() if np.isfinite(r.truecal)}

def correction_override(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal, qover):
    """sg2.correction 과 같되 qover[(f,d)] 가 있으면 cq 를 그 값으로."""
    out = frame[pcol].values.copy()
    for (f, d), idx in frame.groupby(["farm", "day"]).groups.items():
        if d < 179:
            continue
        G = R[R.farm == f]
        G = G[[(f, e) in ref and (f, e) in ec.index and (f, e) not in lock for e in G.day]]
        calc = np.array([cal[(f, e)] for e in G.day])
        rows_ = frame.loc[idx].sort_values("hour"); cum = rows_[pcol].expanding().mean().values
        for k, (ii, rr) in enumerate(rows_.iterrows()):
            h = int(rr.hour)
            if (f, d) in qover:
                cq = qover[(f, d)]
            else:
                continue   # 다른 날은 원 SG2 그대로(아래에서 원 sg 열 사용)
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any():
                continue
            Gm = G[m]; S = SIG[h]
            RS = S.loc[[(f, e) for e in R[R.farm == f].day if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            b = Gm.iloc[int(np.argmin(dist))]
            a1 = ec[(f, b.day)]; pm = cum[k]
            if abs(a1 - pm) <= .30:
                out[frame.index.get_loc(ii)] = rr[pcol] + 0.5 * (a1 - pm)
    return out
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); SGS = [23, 808, 9090]
rm = lambda x: np.sqrt(np.mean(np.square(x)))
for vn in ("DIAG10", "DIAG10y", "EL1"):
    T = S[S.validator == vn].copy()
    res = []
    for k in sorted(T.validation_fold.unique()):
        F = T[T.validation_fold == k]
        vd = set(zip(F.farm, F.day))
        if vn != "EL1":
            DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); f10 = DP[DP.validator == "DIAG10"].groupby(["farm", "day"]).validation_fold.first()
            vd |= {key for key, kk in f10.items() if kk == k} if vn == "DIAG10" else {key for key in labset if ((key[1] + 2) // 6) % 10 == k and key not in LOCK}
        ref = {x for x in labset if x not in vd}
        cal = sg2.ref_calendar(Rr, WV, ref)
        cal2 = dict(cal); cal2.update({x: v for x, v in TRUE.items() if x in cal2})     # 참조 쪽 C3 실제 달력
        qo = {x: TRUE[x] for x in vd if x in TRUE}
        if not qo:
            continue
        for s in SGS:
            sub = F[F.set_index(["farm", "day"]).index.isin(list(qo))].copy()
            new = correction_override(sub, "base_%d" % s, vd, LOCK, ec, Rr, WV, hrs, SIG, ref, cal2, qo)
            sub["new_%d" % s] = new
            res.append(sub[["row_id", "new_%d" % s]].set_index("row_id"))
    if not res:
        continue
    NEW = pd.concat(res, axis=1).T.groupby(level=0).first().T if len(res) > 1 else res[0]
    NEW = pd.concat([r for r in res], axis=1); NEW = NEW.T.groupby(level=0).max().T
    T = T.set_index("row_id")
    print("\n%s: C3 오라클 달력 적용 행 %d" % (vn, len(NEW)))
    for s in SGS:
        col = T["sg_%d" % s].copy(); col.loc[NEW.index] = NEW["new_%d" % s]
        m = T.index.isin(NEW.index)
        print("  SG2 %4d | 2차 전체 %.4f → %.4f | C3 오라클 날 %.4f → %.4f" % (s, rm(T["sg_%d" % s] - T.sub_ec), rm(col - T.sub_ec),
              rm(T["sg_%d" % s][m] - T.sub_ec[m]), rm(col[m] - T.sub_ec[m])))
