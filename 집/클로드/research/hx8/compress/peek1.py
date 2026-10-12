import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm_common import *
sgD, foldD = oof("DIAG10"); sgP, _ = oof("P2LOO")
print("DIAG10 없는 학습 기록:", sorted([k for k in r.rid[TRAIN] if k not in sgD])[:50])
for f, lo, hi in [("F13", 205, 244), ("F47", 200, 236)]:
    for n in range(lo, hi + 1):
        k = f"{f}_{n:03d}"
        if k not in rid2i: print(k, "결번"); continue
        i = rid2i[k]
        if TRAIN[i]:
            p = sgD.get(k, sgP.get(k)); print(k, "T", f"y {ydm[i]:.2f}", f"oof {p.mean():.2f}" if p is not None else "", "cal", cal.get(k), "prev", r.rid[prev[i]] if i in prev else None, "next", r.rid[nxt[i]] if i in nxt else None)
        else:
            print(k, "e", f"sub22 {sub22[k].mean():.3f} h0 {sub22[k][0]:.2f} h23 {sub22[k][23]:.2f}")
hi, ep, E = episodes_true(1.0)
print("고EC 학습", len(hi))
for i in sorted(hi, key=lambda i: (ep[i], E[i])):
    k = r.rid[i]; p = sgD.get(k, sgP.get(k))
    print(k, "ep", r.rid[ep[i]], "E", E[i], f"y {ydm[i]:.2f}", f"oof {p.mean():.2f}" if p is not None else "NA", "cal", cal.get(k), "inD", k in sgD)
