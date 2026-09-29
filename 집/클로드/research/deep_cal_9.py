# -*- coding: utf-8 -*-
"""Q1: build the global calendar (analysis only) = first-pass order + best insertions; measure consistency."""
import env  # noqa
import numpy as np, pandas as pd
C = np.load(env.LOCAL + "/deep_cal_8_C.npy")
k = pd.read_csv(env.LOCAL + "/deep_cal_7_days.csv")
base = [i for i in range(114) if i != 37]
def total(seq):
    return sum(C[a, b] for a, b in zip(seq[:-1], seq[1:]))
def best_insert(seq, block):
    best = None
    for p in range(len(seq) + 1):
        s = seq[:p] + block + seq[p:]
        t = total(s)
        if best is None or t < best[0]:
            best = (t, p)
    return best
seq = base[:]
print("base total cost %.1f" % total(seq))
for blk in ([115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125], [37], [114], [126]):
    t, p = best_insert(seq, blk)
    print("insert %s after date %s  (total %.1f)" % (blk[:2], seq[p - 1] if p else None, t))
    seq = seq[:p] + blk + seq[p:]
# local 2-opt style improvement: adjacent swaps only, report how many improve
imp = []
for i in range(len(seq) - 1):
    s2 = seq[:]; s2[i], s2[i + 1] = s2[i + 1], s2[i]
    if total(s2) < total(seq) - 0.5:
        imp.append((seq[i], seq[i + 1], round(total(seq) - total(s2), 1)))
print("adjacent swaps that would lower cost:", imp)
cal = {d: i for i, d in enumerate(seq)}
k["cal"] = k.date.map(cal)
k.to_csv(env.LOCAL + "/deep_cal_9_days.csv", index=False)
link = np.array([C[a, b] for a, b in zip(seq[:-1], seq[1:])])
print("final calendar: %d dates; link cost median %.2f, share <2: %.2f, share >10: %.2f" % (len(seq), np.median(link), (link < 2).mean(), (link > 10).mean()))
# uniqueness: for each link, is it the mutual best?
mb = [(np.argmin(C[a]) == b) and (np.argmin(C[:, b]) == a) for a, b in zip(seq[:-1], seq[1:])]
top3 = [((C[a] < C[a, b]).sum() < 3) for a, b in zip(seq[:-1], seq[1:])]
print("links that are mutual-best by continuity alone: %.2f ; successor within top3: %.2f" % (np.mean(mb), np.mean(top3)))
# per record monotonicity: is cal increasing with day in first pass
for f in ["F13", "F47"]:
    s = k[k.farm == f].sort_values("day")
    fp = s[s.day <= 178]
    dd = np.diff(fp.cal.values)
    print(f, "first pass: cal diffs share 0:%.2f 1:%.2f  >1:%.2f  <0:%.2f" % ((dd == 0).mean(), (dd == 1).mean(), (dd > 1).mean(), (dd < 0).mean()))
    sp = s[s.day > 178]
    print("  second pass day->cal:", list(zip(sp.day, sp.cal)))
