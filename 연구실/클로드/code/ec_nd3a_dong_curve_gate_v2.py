# -*- coding: utf-8 -*-
"""[v2: v1은 앞 기록이 가려진 날의 달력을 못 구해 대상 일부(오차 큰 날 포함)가 빠짐 → SG2 full_date 와 같은 재귀(쌍둥이 없으면 앞 기록 날짜 + (second면 0, 아니면 .1))로 모든 대상 포함. 판정 기준 동일]
ND3a — 동별 달력 곡선(E1)에 일반 날 하루 수준 정보가 있는가: 관문(정보 감사) — 2026-10-07 연구실 클로드
계획 v1 → 계획 단계 비평 반영(문서/ND3_..._계획_v1.md, 비평 요지는 작업일지):
  E2 삭제. st_dong_assign/역할 'first' 사용 안 함(평가 입력·다음 기록 사용). 동 확률 = AF0b dong_model
  (참조 학습 쌍 기록만으로 학습한 시각별 로지스틱, 23시 값; 참조 기록은 쌍 표지 있으면 0/1).
  이 단계는 '정보 유무' 관문만: 채택·성능 주장 없음. 통과해도 2단계는 시각별 인과·2차 전체 행으로 다시 사전 고정.
설정(검증기 DIAG10 / DIAG10y / EL1, SG2와 같은 가림: ref = 정답 기록 − 가림 집합 vd, 잠금 제외):
  달력 cal = sg2.ref_calendar(R, WV, ref) (학습 기록만), 질의 날짜 cq = 23시 외기 쌍둥이 평균, 없으면 앞 기록 날짜 + .1
  E1 = pq·곡선B + (1−pq)·곡선A,  곡선B = Σ pa·y / Σ pa, 곡선A = Σ (1−pa)·y / Σ(1−pa)  (참조 기록 중 |cal−cq| ≤ 5, cal ≠ cq)
  E4(대조) = 같은 창의 단순 평균 (동 구분 없음)
대상: 2차 정답 기록 중 SG2 시드평균 하루평균 < .9 (평가 부분집합으로만 사용)
관문 기준(고정): 하루 잔차 r = y − sg 를 [E1−sg, E4−sg] 에 회귀(절편 포함) 했을 때
  (i) E1 부분 기울기 > 0 이 세 검증기 모두,  (ii) 날 단위 LOO RMSE: 2변수 < E4 단독, 세 검증기 모두.
  둘 다 충족 → '정보 있음' (2단계 설계로), 아니면 정지.
"""
import os, sys, json, importlib.util
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research")
sys.path.insert(0, RES)
import env  # noqa
import numpy as np, pandas as pd
sys.argv = ["x"]
def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(RES, fn)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sg2 = load("sg2", "ec3_SG2_reference_knn_level_v1.py")
af0b = load("af0b", "ec3_AF0b_anchor_dong_trust_v1.py")

R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(RES, "local")
Rr, WV, hrs, SIG = sg2.prepare_structure()
X = af0b.load_X()
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
Y = Y[Y.farm.isin(["F13", "F47"])]
ec = Y.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
LOCK = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]}
SG = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); SG["sg"] = SG[["sg_23", "sg_808", "sg_9090"]].mean(axis=1)
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); DP = DP[DP.validator == "DIAG10"]
f10 = DP.groupby(["farm", "day"]).validation_fold.first()
chk = SG[SG.validator == "DIAG10"].groupby(["farm", "day"]).validation_fold.first()
assert all(f10[k] == v for k, v in chk.items()), "DIAG10 폴드 정의 불일치"
FOLDS = []
for k in range(10):
    FOLDS.append(("DIAG10", {key for key, kk in f10.items() if kk == k}))
    FOLDS.append(("DIAG10y", {(f, d) for f, d in labset if ((d + 2) // 6) % 10 == k and (f, d) not in LOCK}))
el = SG[SG.validator == "EL1"].groupby(["farm", "day"]).validation_fold.first()
for k in sorted(el.unique()):
    FOLDS.append(("EL1", {key for key, kk in el.items() if kk == k}))

roles = Rr.set_index(["farm", "day"]).role
def full_date(f, d, E, cal, ref, memo=None):
    """SG2 correction.full_date 와 같은 규칙: 참조면 달력, 아니면 23시 외기 쌍둥이, 아니면 앞 기록 + (second 0 / 그 밖 .1). 앞 기록만 사용."""
    if (f, d) in ref:
        return cal[(f, d)]
    A = WV.loc[[(f, e) for e in E]].values; b = WV.loc[(f, d)].values
    ok = np.sqrt(np.nanmean((A - b) ** 2, axis=1)) <= .05
    if ok.any():
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o]))
    days = sorted(Rr[Rr.farm == f].day); i = days.index(d)
    return (full_date(f, days[i - 1], E, cal, ref) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0

rows = []
for V, vd in FOLDS:
    tg = [(f, d) for f, d in vd if d >= 179 and ((SG.validator == V) & (SG.farm == f) & (SG.day == d)).any()]
    if not tg:
        continue
    ref = {k for k in labset if k not in vd and k not in LOCK}
    cal = sg2.ref_calendar(Rr, WV, ref)
    pq, pa, npair = af0b.dong_model(X, WV, hrs, ref)
    for f, d in tg:
        E = [e for e in Rr[Rr.farm == f].day if (f, e) in ref]
        A = WV.loc[[(f, e) for e in E]].values; b = WV.loc[(f, d)].values
        ok = np.sqrt(np.nanmean((A - b) ** 2, axis=1)) <= .05
        if ok.any():
            cq = float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o]))
        else:
            days = sorted(Rr[Rr.farm == f].day); i = days.index(d)
            cq = full_date(f, days[i - 1], E, cal, ref) + .1
        win = [e for e in E if (f, e) in cal and abs(cal[(f, e)] - cq) <= 5 and cal[(f, e)] != cq]
        if not win or not np.isfinite(cq):
            continue
        yv = np.array([ec[(f, e)] for e in win]); wb = np.array([pa(f, e) for e in win]); q = pq(f, d, 23)
        cB = (wb * yv).sum() / wb.sum() if wb.sum() > 0 else yv.mean()
        cA = ((1 - wb) * yv).sum() / (1 - wb).sum() if (1 - wb).sum() > 0 else yv.mean()
        rows.append(dict(V=V, farm=f, day=d, q=q, E1=q * cB + (1 - q) * cA, E4=yv.mean(), nwin=len(win), cB=cB, cA=cA))
    print("  %s 폴드 처리 (대상 %d, 쌍 표지 %d)" % (V, len(tg), npair), flush=True)

T = pd.DataFrame(rows)
D = SG.groupby(["validator", "farm", "day"]).agg(y=("sub_ec", "mean"), sg=("sg", "mean")).reset_index().rename(columns={"validator": "V"})
T = T.merge(D, on=["V", "farm", "day"])
T = T[T.sg < .9].copy(); T["r"] = T.y - T.sg
def loo(Xm, y):
    e = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b = np.linalg.lstsq(Xm[m], y[m], rcond=None)[0]; e.append(y[i] - Xm[i] @ b)
    return np.sqrt(np.mean(np.square(e)))
ok_all = []
print("\n관문: 예측 일반 날, r = y − sg")
for V, q in T.groupby("V"):
    one = np.ones(len(q)); x1 = (q.E1 - q.sg).values; x4 = (q.E4 - q.sg).values; y = q.r.values
    b = np.linalg.lstsq(np.c_[one, x1, x4], y, rcond=None)[0]
    l0 = np.sqrt(np.mean((y - y.mean()) ** 2)); l4 = loo(np.c_[one, x4], y); l14 = loo(np.c_[one, x1, x4], y)
    okv = (b[1] > 0) and (l14 < l4)
    ok_all.append(okv)
    print("  %-8s 일 %d | E1 직접 RMSE %.3f, E4 직접 %.3f, sg %.3f | 부분기울기 E1 %+.2f E4 %+.2f | LOO: 상수 %.4f, E4 %.4f, E1+E4 %.4f → %s" % (
        V, len(q), np.sqrt(np.mean((q.E1 - q.y) ** 2)), np.sqrt(np.mean((q.E4 - q.y) ** 2)), np.sqrt(np.mean(q.r ** 2)),
        b[1], b[2], l0, l4, l14, "충족" if okv else "미충족"))
    print("           pq 분포: 중앙 %.2f, |pq−.5|>.3 비율 %.2f, 창 기록 수 중앙 %d" % (q.q.median(), (np.abs(q.q - .5) > .3).mean(), q.nwin.median()))
print("\n[ND3a 관문] %s" % ("정보 있음 → 2단계 설계" if all(ok_all) else "정지 (정보 없음)"))
T.to_csv(os.path.join(R, "..", "results", "ec_nd3a_gate_v2.csv"), index=False, encoding="utf-8-sig")
