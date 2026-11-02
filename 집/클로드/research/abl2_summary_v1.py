"""ABL2 요약(abl2_PLAN.md 보고 항목). local/abl2/preds_abl2_*.csv → 판 × 묶음 표, 유형별 탐지율, A 대비 바뀐 장수."""
import sys, os, csv, glob
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local", "abl2")
NAME = {"A": "A 그대로", "D": "D 지문 DB 끔", "E": "E 형식 게이트 끔", "F": "F 둘 다 끔", "G": "G 게이트 늘 엶"}
print("| 묶음 | 실제/가짜 | 판 | 균형 정확도 | 실제 오탐률 | 가짜 놓침률 | A 대비 실제→가짜 | A 대비 가짜→실제 |")
print("|---|---|---|---|---|---|---|---|")
kinds = []
for p in sorted(glob.glob(os.path.join(W, "preds_abl2_*.csv"))):
    R = list(csv.DictReader(open(p, encoding="utf-8"))); sn = R[0]["set"]
    y = [int(r["label"]) for r in R]; nr, nf = y.count(0), y.count(1)
    for v in "ADEFG":
        pr = [float(r["p" + v]) >= .5 for r in R]; pa = [float(r["pA"]) >= .5 for r in R]
        fpr = sum(p_ for p_, t in zip(pr, y) if t == 0) / nr if nr else None
        fnr = sum(not p_ for p_, t in zip(pr, y) if t == 1) / nf if nf else None
        ba = 1 - (fpr + fnr) / 2 if nr and nf else None
        rf = sum(1 for a, b, t in zip(pa, pr, y) if t == 0 and not a and b); fr = sum(1 for a, b, t in zip(pa, pr, y) if t == 1 and a and not b)
        f = lambda x: "—" if x is None else f"{x:.4f}"
        print(f"| {sn} | {nr}/{nf} | {NAME[v]} | {f(ba)} | {f(fpr)} | {f(fnr)} | {'—' if v == 'A' else rf} | {'—' if v == 'A' else fr} |")
    by = defaultdict(list)
    for r in R:
        if r["label"] == "1": by[r["kind"]].append(r)
    for k, rr in sorted(by.items()):
        kinds.append((sn, k, len(rr), [sum(float(r["p" + v]) >= .5 for r in rr) / len(rr) for v in "ADEF"]))
print("\n| 묶음 | 가짜 유형 | 장수 | A | D | E | F |\n|---|---|---|---|---|---|---|")
for sn, k, n, a in kinds:
    print(f"| {sn} | {k} | {n} | " + " | ".join(f"{x:.3f}" for x in a) + " |")
