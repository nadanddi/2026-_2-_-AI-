# -*- coding: utf-8 -*-
"""RV2 — HG3(SG2 + 엄격 도메인 보호 고EC 보정)를 9회차 실제 구성 위에서 조건부 점검 (실행 전 고정) — 2026-10-07 연구실 클로드
근거: HG3(6.329, 집 클로드) R3S 기준선 위에서 모든 칸 개선·일반 날 변화 0, 그러나 P2LOO P .0386 > .025(구조적 하한 6.341) → 사용자 결정 대기.
      9회차 구성(0.8 R3_DP1 + 0.2 PFN → shrink·clip → SG2) 위에서는 판정한 적 없음.
저장 OOF(재학습 없음): 집 클로드 ec3_WT1_all.csv (Drive 사본) 구성원 et/lgb/mlp 시드 47/1414/6464(DP1 포함) + pfn, lo/hi.
기준선 X = clip(0.8 (.6 et + .3 lgb + .1 mlp) + 0.2 pfn, lo, hi) [= 9회차 EC 본체의 '근사 구성': R3 시드·PFN 문맥(WT1 5~8, 제출 1~4)·
      잠금 날 앵커 처리가 제출과 다름. 구성원·pfn 열은 이미 shrink 된 값이고 shrink 가 선형이라 혼합 후 한 번 shrink 와 같음].
      주판정 = CUR(0.2). B4(0.6/0.4)는 보고용.
보정: HG3 코드의 correction() 그대로(prepare_structure·s_low_table 포함) → (SG2 출력, HG3 출력) → 각각 같은 clip.
      가림 집합: DIAG10 = p3 폴드(1·2차), P2LOO = 그 2차 날, EL1 = 5기록 묶음 (WT1 과 동일).
비교: HG3 출력 vs SG2 출력 (= 9회차 구성 위에 HG3 를 더했을 때의 변화).
[고정 기준 — '조건부 점검'(같은 OOF, 결정적 후처리만 추가; 새 증거 아님)] 통과 iff
 (a) 시드 3 × {DIAG10 2차, P2LOO, EL1} 9칸 모두 2차 시간 RMSE 개선,
 (b) 같은 9칸에서 일반 날(정답 하루평균 < 1) RMSE 가 나빠지지 않음(Δ ≤ 1e-9); 바뀐 일반 날 수·목록은 보고 [계획 비평: '정확히 0'은 구조적 보장 아님],
 (c) 고EC 날 9칸 모두 개선.
 P(worse)(P2LOO 시드평균, 온실×5기록 묶음 20000)와 그 구조적 하한(바뀐 묶음 비율 q 일 때 (1−q)^n)은 보고만 함 — 6.341 에 따라 통과 불가가 구조적으로 정해진 기준이므로 판정에 쓰지 않음(사전 명시).
 함께: TM 2차 행, 바뀐 날 목록, 날 단위 개선/악화 수, P2LOO·DIAG10 2차·TM 2차 P 와 각 구조적 하한, B4 결과.
 통과 ≠ 채택: CLAUDE.md 채택 기준(DIAG10 p_worse < .025)은 구조적으로 충족 불가 → 결과는 '사용자 결정 자료'.
 정합성 검사(필수): 이 스크립트의 SG2 출력이 RV1(ec_rv1_preds_v1.csv)의 sg 와 최대차 < 1e-9.
비고: 연구용 correction 은 잠금 날을 앵커에서 제외(제출 sg2post 와 다름).
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
O = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv"))
TM = pd.read_csv(os.path.join(DC, "tm1_set_v1.csv")); TMs = set(zip(TM.farm, TM.day))
SEEDS = (47, 1414, 6464)
D10 = {i: vd for name, i, vd in fds if name == "DIAG10"}
print("비고: 연구용 correction 은 잠금 날을 앵커에서 제외(제출 sg2post 는 포함)", flush=True)

def build(G, w, s):
    r3 = .6 * G["et_%d" % s] + .3 * G["lgb_%d" % s] + .1 * G["mlp_%d" % s]
    return np.clip((1 - w) * r3 + w * G.pfn, G.lo, G.hi).values

CKD = os.path.join(R, "..", "local", "rv2_ckpt"); os.makedirs(CKD, exist_ok=True)
recs = []
for (vn, k), G in O.groupby(["validator", "validation_fold"]):
    G = G.copy()
    if not (G.day >= 179).any():
        continue
    ckf = os.path.join(CKD, "%s_%d.csv" % (vn, k))
    if os.path.exists(ckf):
        recs.append(pd.read_csv(ckf)); continue
    part = []
    vd = D10[k] if vn == "DIAG10" else set(zip(G.farm, G.day))
    ref = {x for x in labset if x not in vd}
    cal = hg3.ref_calendar(Rr, WV, ref); ST = hg3.s_low_table(DM, ref)
    for w, tag in ((.2, "CUR"), (.4, "B4")):
        for s in SEEDS:
            G["x"] = build(G, w, s)
            sgx, hgx = hg3.correction(G, "x", vd, lockd, ec, Rr, WV, hrs, SIG, ref, cal, DM, ST)
            part.append(pd.DataFrame(dict(validator=vn, fold=k, row_id=G.row_id.values, farm=G.farm.values, day=G.day.values,
                                          hour=G.hour.values, y=G.sub_ec.values, tag=tag, seed=s,
                                          sg=np.clip(sgx, G.lo, G.hi), hg=np.clip(hgx, G.lo, G.hi))))
    pd.concat(part).to_csv(ckf, index=False); recs.append(pd.concat(part))
    print("  %s/%d 완료" % (vn, k), flush=True)
P = pd.concat(recs, ignore_index=True)
RV1 = pd.read_csv(os.path.join(R, "..", "results", "ec_rv1_preds_v1.csv"))
chk = P.merge(RV1[RV1.tag.isin(["CUR", "B4"])][["validator", "fold", "row_id", "tag", "seed", "sg"]], on=["validator", "fold", "row_id", "tag", "seed"], suffixes=("", "_rv1"))
dmax = float(np.abs(chk.sg - chk.sg_rv1).max())
print("정합성: RV1 SG2 출력과 최대차 %.2e (행 %d)" % (dmax, len(chk)))
assert dmax < 1e-9, "SG2 출력 불일치"
P = P[P.day >= 179].copy()
P["dm"] = P.groupby(["validator", "fold", "tag", "seed", "farm", "day"]).y.transform("mean")
P.to_csv(os.path.join(R, "..", "results", "ec_rv2_preds_v1.csv"), index=False, encoding="utf-8-sig")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
res = {}
def boot(Z):
    Z = Z.copy(); Z["cl"] = Z.farm + "_" + (Z.day // 5).astype(str)
    dd = ((Z.hg - Z.y) ** 2 - (Z.sg - Z.y) ** 2).groupby(Z.cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(20261007).integers(0, len(sm), (20000, len(sm)))
    pw = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    ch = int((np.abs(sm) > 1e-12).sum()); floor = (1 - ch / len(sm)) ** len(sm)
    return pw, ch, len(sm), floor
for tag in ("CUR", "B4"):
    print("\n=== 기준선 %s (+SG2) 위에 HG3 %s ===" % (tag, "[주판정]" if tag == "CUR" else "[보고용]"))
    ok_all, ok_norm, ok_high = True, True, True
    for vn, nm in (("DIAG10", "DIAG10 2차"), ("P2LOO", "P2LOO"), ("EL1", "EL1")):
        Q = P[(P.validator == vn) & (P.tag == tag)]
        cells = []
        for s in SEEDS:
            q = Q[Q.seed == s]; a, b = rm(q.sg - q.y), rm(q.hg - q.y)
            n0, n1 = rm((q.sg - q.y)[q.dm < 1]), rm((q.hg - q.y)[q.dm < 1])
            h0, h1 = rm((q.sg - q.y)[q.dm >= 1]), rm((q.hg - q.y)[q.dm >= 1])
            nch = q[(q.dm < 1) & (np.abs(q.hg - q.sg) > 1e-12)].groupby(["farm", "day"]).ngroups
            cells.append("s%d 전체 %.4f→%.4f (%+.1f%%) | 일반 Δ%+.1e (바뀐 일반 날 %d) | 고EC %.3f→%.3f" % (s, a, b, 100 * (b / a - 1), n1 - n0, nch, h0, h1))
            ok_all &= b < a; ok_norm &= (n1 - n0) <= 1e-9; ok_high &= h1 < h0
        print("  %-10s\n    %s" % (nm, "\n    ".join(cells)))
    for nm, m in (("P2LOO", P.validator == "P2LOO"), ("DIAG10 2차", P.validator == "DIAG10"),
                  ("TM 2차", (P.validator == "DIAG10") & np.array([(f, d) in TMs for f, d in zip(P.farm, P.day)]))):
        Z = P[m & (P.tag == tag)].groupby("row_id").agg(sg=("sg", "mean"), hg=("hg", "mean"), y=("y", "first"), farm=("farm", "first"), day=("day", "first"))
        if len(Z) == 0:
            continue
        pw, ch, nb, floor = boot(Z)
        dday = Z.groupby(["farm", "day"]).apply(lambda g: ((g.hg - g.y) ** 2).sum() - ((g.sg - g.y) ** 2).sum())
        dday = dday[np.abs(dday) > 1e-12]
        print("  %-10s RMSE %.4f→%.4f | P(worse) %.4f (바뀐 묶음 %d/%d, 구조적 하한 %.4f) | 바뀐 날 %d: 개선 %d·악화 %d" % (
            nm, rm(Z.sg - Z.y), rm(Z.hg - Z.y), pw, ch, nb, floor, len(dday), int((dday < 0).sum()), int((dday > 0).sum())))
    ch = P[(P.tag == tag) & (np.abs(P.hg - P.sg) > 1e-12)].groupby(["farm", "day"]).size()
    print("  바뀐 날 목록:", ", ".join("%s_%d" % k for k in ch.index))
    res[tag] = ok_all and ok_norm and ok_high
    print("  [%s 조건부 점검] (a) 9칸 개선 %s, (b) 일반 날 안 나빠짐 %s, (c) 고EC 9칸 개선 %s → %s (통과 ≠ 채택: 사용자 결정 자료)" % (
        tag, ok_all, ok_norm, ok_high, "통과" if res[tag] else "미통과"))
