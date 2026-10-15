# -*- coding: utf-8 -*-
"""HGX1 (2026-10-15 집 클로드; 사용자 '이전 모델을 뜯어봐서 살릴 부분만, 5점 규정 확인' — HG3).  서술 분해(판정 없음), 실행 전 커밋.  EC.
HG3(6.329/6.333) 부품: 발동(pm_h >= .9 & S_low <= -1: 자기 기록, F) + 목표(SG2 이웃 2개 하루 평균 EC >= 1.0: 같은 농장, 학습 달력(기록 번호 기반 추정 날짜) +-3일, 제어 서명 거리) + 반영 .5.
질문: 목표 이웃이 (a) 대상의 과거인가 미래인가(달력 기준), (b) 진짜 같은 동 전날/다음날(cert 정답 짝)인가, (c) 27번 FL 확신 어제(K8)로 이미 덮이는가.
반사실(같은 발동·반영, DIAG10 저장 R3S OOF 360일, 기준 = R3S p): ORIG(+-3), PAST(달력상 앞 날만), FUTURE(뒤 날만), FLY(FL 확신 어제 p >= .8 의 하루 평균 EC 하나).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u hgx1_hg3_dissection_v1.py
"""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("hk0", os.path.join(HERE, "hk0_high_gate_anchor_v1.py")); hk0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hk0)
sg2 = hk0.sg2
sys.path.insert(0, os.path.join(HERE, "linker_v7"))
from feat7 import TL7, rid  # noqa: E402
CERT = {rid(int(b)): rid(int(a)) for a, b in zip(TL7.a, TL7.b)}; NEXT = {v: k for k, v in CERT.items()}


def anchors_prov(frame, pcol, ec, R, WV, hrs, SIG, ref, cal, lock, mode):
    out = pd.DataFrame(index=frame.index, columns=["a1", "a2", "s1", "s2", "o1", "o2"], dtype=object)
    roles = R.set_index(["farm", "day"]).role; cache = {}

    def twin_date(f, d, h):
        E = [e for e in R[R.farm == f].day if (f, e) in ref]; cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None

    def full_date(f, d):
        if (f, d) in ref: return cal[(f, d)]
        if (f, d) in cache: return cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            days = sorted(R[R.farm == f].day); i = days.index(d)
            t = (full_date(f, days[i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t; return t

    for (f, d), idx in frame.groupby(["farm", "day"]).groups.items():
        G = R[R.farm == f]; G = G[[(f, e) in ref and (f, e) in ec.index and (f, e) not in lock for e in G.day]]
        calc = np.array([cal[(f, e)] for e in G.day]); days = sorted(R[R.farm == f].day); i = days.index(d)
        rows = frame.loc[idx].sort_values("hour")
        for ii, rr in rows.iterrows():
            h = int(rr.hour); cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None: cq = (full_date(f, days[i - 1]) + 0.1) if i else 0.0
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if mode == "past": m &= calc < cq
            if mode == "future": m &= calc > cq
            if not m.any(): continue
            Gm = G[m]; S = SIG[h]; RS = S.loc[[(f, e) for e in R[R.farm == f].day if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq); o = np.argsort(dist)
            e1 = int(Gm.iloc[o[0]].day); out.loc[ii, ["a1", "s1", "o1"]] = [ec[(f, e1)], f"{f}_{e1:03d}", calc[m][o[0]] - cq]
            if len(o) > 1:
                e2 = int(Gm.iloc[o[1]].day); out.loc[ii, ["a2", "s2", "o2"]] = [ec[(f, e2)], f"{f}_{e2:03d}", calc[m][o[1]] - cq]
    return out


raw, full, lab, lock, signatures, fds = sg2.p3.prepare(); R, WV, hrs, SIG = sg2.prepare_structure()
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int); X = X.sort_values(["farm", "day", "hour"])
DV = ["in_hum", "act_shade", "act_thermal", "act_fog", "in_temp", "in_co2"]; SGN = np.array([1, 1, 1, 1, -1, -1.]); gX = X.groupby(["farm", "day"])
for v in DV: X["cm_" + v] = gX[v].transform(lambda z: z.expanding().mean())
for f in ("F13", "F47"):
    for h in range(24):
        mm = (X.farm == f) & (X.hour == h); Cc = X.loc[mm, ["cm_" + v for v in DV]]
        X.loc[mm, "S_low"] = (((Cc - Cc.mean()) / Cc.std().replace(0, 1)).fillna(0).values * SGN).sum(axis=1)
O = O.merge(X[["row_id", "S_low"]], on="row_id"); O["rec"] = O.farm + "_" + O.day.map("{:03d}".format)
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
res = {}
for mode in ("all", "past", "future"):
    parts = []
    for k, G in O.groupby("validation_fold"):
        vd = set(zip(G.farm, G.day)); ref = {x for x in labset if x not in vd}; cal = sg2.ref_calendar(R, WV, ref)
        parts.append(anchors_prov(G, "p", ec, R, WV, hrs, SIG, ref, cal, lockd, mode))
    res[mode] = pd.concat(parts).reindex(O.index); print("anchors", mode, "done", flush=True)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def hg(A):
    a1, a2 = A.a1.astype(float), A.a2.astype(float)
    gate = a1.notna() & a2.notna() & (a1 >= 1) & (a2 >= 1) & (O.pm >= .9) & (O.S_low <= -1.0)
    return np.where(gate, O.p + .5 * ((a1 + a2) / 2 - O.pm), O.p), gate


FOLDDAYS = {k: set(zip(G.farm, G.day)) for k, G in O.groupby("validation_fold")}
def fly(row_rec, fold):
    if row_rec not in FLK or FLK[row_rec][1] < .8: return np.nan
    p = FLK[row_rec][0]; key = (p[:3], int(p[4:]))
    return ec.get(key, np.nan) if key not in FOLDDAYS[fold] else np.nan
fly_a = pd.Series([fly(a, b) for a, b in zip(O.rec, O.validation_fold)], index=O.index)
gF = fly_a.notna() & (fly_a >= 1) & (O.pm >= .9) & (O.S_low <= -1.0); pF = np.where(gF, O.p + .5 * (fly_a - O.pm), O.p)
V = {"ORIG": hg(res["all"]), "PAST": hg(res["past"]), "FUTURE": hg(res["future"]), "FLY": (pF, gF)}
segs = {"all": O.dm.notna(), "high": O.dm >= 1, "normal": O.dm < 1, "pass-2": O.day >= 179}
print("\nvs R3S p (no SG2):")
for nm, (pp, gt) in V.items():
    days = O[gt].drop_duplicates("rec")
    print(f"  {nm:6s} gated days {len(days):2d} (true high {int((days.dm >= 1).sum())}) | " + " | ".join(
        f"{s} {100 * (r(pp[m] - O.sub_ec[m]) / r(O.p[m] - O.sub_ec[m]) - 1):+.2f}%" for s, m in segs.items()))
A = res["all"]; gt = V["ORIG"][1]; D = O[gt].copy()
D["s1"], D["o1"], D["s2"], D["o2"] = A.s1[gt], A.o1[gt].astype(float), A.s2[gt], A.o2[gt].astype(float)
D = D.drop_duplicates("rec")
D["a1_true_prev"] = [CERT.get(r_) == s for r_, s in zip(D.rec, D.s1)]; D["a1_true_next"] = [NEXT.get(r_) == s for r_, s in zip(D.rec, D.s1)]
D["has_FL_K8"] = [r_ in FLK and FLK[r_][1] >= .8 for r_ in D.rec]
pd.set_option("display.width", 220)
print("\nORIG gated days - anchor provenance (calendar offset o = neighbour - target):")
print(D[["rec", "dm", "pm", "s1", "o1", "s2", "o2", "a1_true_prev", "a1_true_next", "has_FL_K8"]].round(2).to_string(index=False))
print("share a1 in PAST %.2f | a1 true prev %.2f | a1 true next %.2f | FL K8 already %.2f" % ((D.o1 < 0).mean(), D.a1_true_prev.mean(), D.a1_true_next.mean(), D.has_FL_K8.mean()))
