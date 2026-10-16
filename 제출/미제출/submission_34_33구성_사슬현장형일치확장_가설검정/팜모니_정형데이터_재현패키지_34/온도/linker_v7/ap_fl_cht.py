# submission_34 (2026-10-16 집 클로드; catalog CHT1 7.046): chain extension by field two-view agreement (temperature only).
# Same eval picks as ap_fl.py eval (w_fl.npy, hour-0 decision, per-record independent, same visibility).  For every evaluation record whose FL
# previous record is an UNLABELLED same-farm EARLIER-numbered evaluation record with p0 < .5 and |dT_air| <= 1.5, the record joins the
# prediction chain if the OUT (outdoor continuity) or IN (indoor sensor/actuator continuity) view picks the same previous record (option chosen by data on
# validation blocks: p_min 0, agreement >= 1).  Writes <out dir>/fl_eval_chain_ext.csv (rec, prev, p0, agree).
# Run:  python ap_fl_cht.py <out dir> <w_fl.npy>
import sys, os
OUTD = sys.argv[1]
from feat7 import *
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]
MO = np.array([1] + [0] * 8 + [1, 1], float); MI = np.array([0] + [1] * 8 + [1, 1], float)
w = np.load(sys.argv[2]); out = []
def soft(s): p = np.exp(s - s.max()); return p / p.sum()
for f in ["F13", "F47"]:
    for r_ in sorted(np.where((FARM == f) & ~TRAIN)[0], key=lambda i: NUM[i]):
        df, _ = features7(r_, 0, TRAIN.copy(), removed=set())
        if len(df) == 0: continue
        X_ = df[FL].values.astype(float); a_ = df.a.values
        p = soft(X_ @ w); k = int(np.argmax(p)); a0 = int(a_[k]); p0 = float(p[k])
        ag = int(int(a_[int(np.argmax(X_ @ (w * MO)))]) == a0) + int(int(a_[int(np.argmax(X_ @ (w * MI)))]) == a0); dT = IN_[r_, 0, 0] - IN_[a0, 23, 0]
        if (not TRAIN[a0]) and FARM[a0] == f and NUM[a0] < NUM[r_] and p0 < .5 and np.isfinite(dT) and abs(dT) <= 1.5 and ag >= 1:
            out.append(dict(rec=rid(r_), prev=rid(a0), p0=round(p0, 4), agree=ag))
pd.DataFrame(out, columns=["rec", "prev", "p0", "agree"]).to_csv(os.path.join(OUTD, "fl_eval_chain_ext.csv"), index=False)
print("CHT eval chain extension:", len(out), [(d["rec"], d["prev"], d["p0"]) for d in out])
