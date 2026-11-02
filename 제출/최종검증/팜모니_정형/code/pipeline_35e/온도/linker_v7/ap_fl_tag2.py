# submission_33 (2026-10-16 집 클로드; catalog 7.030 TAG2): field-type two-view agreement anchors for TEMPERATURE only.
# Same eval picks as ap_fl.py eval (same weights w_fl.npy, same candidates, hour-0 decision, per-record independent, same visibility).
# From the same candidate table two extra views: OUT = outdoor-continuity score only (oj + excl + xf), IN = indoor sensor/actuator continuity only (v0..v7 + excl + xf).
# Added anchors (TAG2, option chosen by data on validation blocks: agreement >= 1, |dT_air| <= 1.5, p0 >= .5):
#   records without the round-27 anchor whose FL previous record is a LABELLED training record, .5 <= p0 < .8, |dT_air| <= 1.5,
#   and the OUT or IN view picks the same previous record  -> appended to the anchor rows (same columns).  EC keeps the original anchor file.
# Run:  python ap_fl_tag2.py <out dir with fl_eval_anchor.csv> <w_fl.npy>   (LINKER_WORK as for ap_fl.py)  -> <out dir>/fl_eval_anchor_tag2.csv
import sys, os
OUTD = sys.argv[1]
from feat7 import *
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]
MO = np.array([1] + [0] * 8 + [1, 1], float); MI = np.array([0] + [1] * 8 + [1, 1], float)
w = np.load(sys.argv[2])
def soft(s): p = np.exp(s - s.max()); return p / p.sum()
A = pd.read_csv(os.path.join(OUTD, "fl_eval_anchor.csv"), encoding="utf-8-sig"); have = set(A.row); add = []
for f in ["F13", "F47"]:
    for r_ in sorted(np.where((FARM == f) & ~TRAIN)[0], key=lambda i: NUM[i]):
        if rid(r_) in have: continue
        df, _ = features7(r_, 0, TRAIN.copy(), removed=set())
        if len(df) == 0: continue
        X_ = df[FL].values.astype(float); a_ = df.a.values
        p = soft(X_ @ w); k = int(np.argmax(p)); a0 = int(a_[k]); p0 = float(p[k])
        ko = int(np.argmax(X_ @ (w * MO))); ki = int(np.argmax(X_ @ (w * MI)))
        agree = int(int(a_[ko]) == a0) + int(int(a_[ki]) == a0); dT = IN_[r_, 0, 0] - IN_[a0, 23, 0]
        if TRAIN[a0] and .5 <= p0 < .8 and np.isfinite(dT) and abs(dT) <= 1.5 and agree >= 1 and np.isfinite(ST[a0, 23]):
            add.append(dict(row=rid(r_), rec=rid(r_), prev_h0=rid(a0), prev0=rid(a0), p_h0=round(p0, 3), p0=round(p0, 4), prev_is_labelled=True, xf=FARM[a0] != f,
                            steps_to_anchor=1, anchor_ec23=round(EC[a0, 23], 3), anchor_st23=round(ST[a0, 23], 2), dT_air=round(dT, 2)))
O = pd.concat([A, pd.DataFrame(add)], ignore_index=True) if add else A
O.to_csv(os.path.join(OUTD, "fl_eval_anchor_tag2.csv"), index=False, encoding="utf-8-sig")
print("TAG2 eval: original anchors", len(A), "| added", len(add), [(d["row"], d["prev0"], d["p0"]) for d in add])
