# -*- coding: utf-8 -*-
"""RV1 — WT1(TabPFN 비중 0.4)을 현재 제출 구성(SG2 후처리 포함) 위에서 재확인 (실행 전 고정) — 2026-10-07 연구실 클로드
근거: WT1(6.355, 집 클로드) B4 = 0.6 R3_DP1 + 0.4 PFN 이 CUR(0.8/0.2) 대비 TM −4.3~−4.6%·P2LOO −5.3~−5.6%·EL1 −2.2~−2.3%, P .0068 ADOPT.
      단 SG2(제출 구성의 마지막 후처리, 2차 행에만 작용)는 넣지 않은 판정이었음.
저장 OOF(재학습 없음): 집 클로드 ec3_WT1_all.csv (Drive 사본, 연구실/클로드/local/drive_copy/) — 구성원 et/lgb/mlp 시드 47/1414/6464(DP1 포함), pfn(문맥 5~8 평균), lo/hi.
구성: X_w = clip((1−w)·(.6 et + .3 lgb + .1 mlp) + w·pfn, lo, hi), w ∈ {.2 = CUR, .4 = B4}
      → SG2 후처리(ec3_SG2 의 correction 그대로: 참조 = 정답 기록 − 가림 집합, 잠금 제외 앵커, 보호 .30, 0.5) → 같은 clip.
      가림 집합(vd): DIAG10 = DP1 파일의 DIAG10 폴드(1·2차 날 모두), P2LOO = 그 2차 날 하나, EL1 = 그 5기록 묶음. (WT1 과 동일)
집합: TM(평가 닮은 111일, DIAG10 행; SG2 는 그중 2차 행에만 작용), P2LOO 2차 행, EL1 2차 행, 참고 DIAG10 2차 행.
[고정 기준 — 계획 단계 비평 반영; '조건부 점검'(같은 OOF 에 결정적 후처리 SG2 만 얹음, 새 증거 아님)] B4+SG2 가 CUR+SG2 대비 통과 iff
 (a) TM 에서 R3 시드 3 모두 개선, (b) 시드 × {P2LOO, EL1, DIAG10 2차} 모든 칸 개선(1% 허용 없음),
 (c) TM 시드평균 온실×5기록 묶음 부트스트랩 20000 P(worse) < .025, (d) DIAG10 전체 행 같은 방식 P(worse) < .025,
 (e) 다른 PFN 추출본(Codex DC4 통합 OOF season_pfn, DIAG10, 시드 7/101/2024 를 R3 시드 47/1414/6464 와 짝지음)으로 바꿔도
     TM 과 DIAG10 2차에서 세 짝 모두 개선.  함께: TM 2차 행만, 일반 날·고EC 날 분해.
비고: 연구용 SG2 correction 은 잠금 날을 앵커에서 빼지만 제출 패키지 sg2post 는 400일 전부 참조 → 이 점검은 제출보다 이웃이 적음.
"""
import os, sys, json, importlib.util
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research")
sys.path.insert(0, RES)
import env  # noqa
import numpy as np, pandas as pd
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(RES, "ec3_SG2_reference_knn_level_v1.py")); sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
R = os.path.dirname(os.path.abspath(__file__)); DC = os.path.join(R, "..", "local", "drive_copy")
raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
Rr, WV, hrs, SIG = sg2.prepare_structure()
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
O = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv"))
TM = pd.read_csv(os.path.join(DC, "tm1_set_v1.csv")); TMs = set(zip(TM.farm, TM.day))
SEEDS = (47, 1414, 6464)
D10 = {}
for name, i, vd in fds:
    if name == "DIAG10":
        D10[i] = vd

ALT = pd.read_csv(os.path.join(DC, "codex_v2_integration_oof.csv"))
ALT = ALT[ALT.validator == "DIAG10"].pivot_table(index="row_id", columns="seed", values="season_pfn")
PAIR = {47: 7, 1414: 101, 6464: 2024}
print("비고: 연구용 SG2 는 잠금 날을 앵커에서 제외(제출 sg2post 는 포함) → 이 점검의 SG2 는 제출보다 이웃이 적음", flush=True)

def build(G, w, s, alt=False):
    r3 = .6 * G["et_%d" % s] + .3 * G["lgb_%d" % s] + .1 * G["mlp_%d" % s]
    pf = ALT.loc[G.row_id, PAIR[s]].values if alt else G.pfn.values
    return np.clip((1 - w) * r3.values + w * pf, G.lo, G.hi).values

out = {}
for (vn, k), G in O.groupby(["validator", "validation_fold"]):
    G = G.copy()
    if vn == "DIAG10":
        vd = D10[k]
    else:
        vd = set(zip(G.farm, G.day))
    ref = {x for x in labset if x not in vd}
    cal = sg2.ref_calendar(Rr, WV, ref)
    combos = [(.2, "CUR", False), (.4, "B4", False)] + ([(.2, "CURa", True), (.4, "B4a", True)] if vn == "DIAG10" else [])
    for w, tag, alt in combos:
        for s in SEEDS:
            G["x"] = build(G, w, s, alt)
            G["sgx"] = np.clip(sg2.correction(G, "x", vd, lockd, ec, Rr, WV, hrs, SIG, ref, cal), G.lo, G.hi)
            for rid, v0, v1 in zip(G.row_id, G.x, G.sgx):
                out[(vn, k, rid, tag, s)] = (v0, v1)
    print("  %s/%d 완료" % (vn, k), flush=True)

rows = []
for (vn, k, rid, tag, s), (v0, v1) in out.items():
    rows.append(dict(validator=vn, fold=k, row_id=rid, tag=tag, seed=s, nosg=v0, sg=v1))
P = pd.DataFrame(rows).merge(O[["validator", "validation_fold", "row_id", "farm", "day", "hour", "sub_ec"]].rename(columns={"validation_fold": "fold"}),
                             on=["validator", "fold", "row_id"])
P.to_csv(os.path.join(R, "..", "results", "ec_rv1_preds_v1.csv"), index=False, encoding="utf-8-sig")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
TMm = np.array([(f, d) in TMs for f, d in zip(P.farm, P.day)])
sets = {"TM": (P.validator == "DIAG10") & TMm, "TM 2차만": (P.validator == "DIAG10") & TMm & (P.day >= 179),
        "P2LOO": (P.validator == "P2LOO") & (P.day >= 179), "EL1": (P.validator == "EL1") & (P.day >= 179),
        "DIAG10 2차": (P.validator == "DIAG10") & (P.day >= 179), "DIAG10 전체": (P.validator == "DIAG10")}
def cmp(Q, ta, tb):
    cells, oks = [], []
    for s in SEEDS:
        a = Q[(Q.tag == ta) & (Q.seed == s)].sort_values("row_id"); b = Q[(Q.tag == tb) & (Q.seed == s)].sort_values("row_id")
        ra, rb = rm(a.sg - a.sub_ec), rm(b.sg - b.sub_ec)
        cells.append("s%d %.4f→%.4f (%+.1f%%)" % (s, ra, rb, 100 * (rb / ra - 1))); oks.append(rb < ra)
    return cells, oks
ok = {}
for nm, m in sets.items():
    cells, oks = cmp(P[m], "CUR", "B4"); ok[nm] = all(oks)
    print("%-11s %s" % (nm, "  ".join(cells)))
for nm in ("TM", "DIAG10 2차"):
    cells, oks = cmp(P[sets[nm]], "CURa", "B4a"); ok[nm + " (다른 PFN)"] = all(oks)
    print("%-11s (다른 PFN) %s" % (nm, "  ".join(cells)))
def boot(m):
    T = P[m]
    bm = T[T.tag == "CUR"].groupby("row_id").agg(p=("sg", "mean"), y=("sub_ec", "first"), farm=("farm", "first"), day=("day", "first"))
    bm["c"] = T[T.tag == "B4"].groupby("row_id").sg.mean(); bm["cl"] = bm.farm + "_" + (bm.day // 5).astype(str)
    dd = ((bm.c - bm.y) ** 2 - (bm.p - bm.y) ** 2).groupby(bm.cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(20261007).integers(0, len(sm), (20000, len(sm)))
    return float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()), bm
p_tm, bm = boot(sets["TM"]); p_d10, _ = boot(sets["DIAG10 전체"])
bm["dm"] = bm.groupby(["farm", "day"]).y.transform("mean")
for nm, m in (("TM 전체", bm.dm > -1), ("TM 일반", bm.dm < 1), ("TM 고EC", bm.dm >= 1)):
    print("  %s 시드평균 CUR+SG2 %.4f → B4+SG2 %.4f" % (nm, rm(bm.p[m] - bm.y[m]), rm(bm.c[m] - bm.y[m])))
print("  P(worse): TM %.4f, DIAG10 전체 %.4f" % (p_tm, p_d10))
crit = {"(a) TM 세 시드": ok["TM"], "(b) 보호 칸 모두 개선": ok["P2LOO"] and ok["EL1"] and ok["DIAG10 2차"],
        "(c) TM P<.025": p_tm < .025, "(d) DIAG10 P<.025": p_d10 < .025, "(e) 다른 PFN": ok["TM (다른 PFN)"] and ok["DIAG10 2차 (다른 PFN)"]}
print("\n[RV1 조건부 점검 기준]", crit, "→", "통과(SG2 위에서도 B4 이득 유지)" if all(crit.values()) else "미통과")
