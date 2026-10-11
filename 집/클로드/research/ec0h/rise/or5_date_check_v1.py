# OR5: OR4 연결이 날짜상 맞는지 — 각 기록과 외기 24h가 90% 이상 같은 기록(같은 날짜) 목록, 앞날 후보의 날짜+1 이 대상 날짜인지
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
Xf = X.reshape(N, -1)
def same_date(i): return [rid(j) + ("" if TRAIN[j] else "(E)") for j in range(N) if j != i and (np.abs(Xf[j] - Xf[i]) < 1e-6).mean() >= .9]
def m23_0(a, b):  # outdoor continuity a 23h -> b 0h (abs diff of 4 outdoor vars / sd)
    return float(np.nansum(np.abs(X[a, 23] - X[b, 0]) / sdO))
for t, p in (("F13_217", "F13_127"), ("F47_229", "F13_217"), ("F47_231", "F47_229"), ("F13_241", "F13_125"), ("F13_243", "F47_151")):
    i, j = idx[t], idx[p]
    print(f"{t}: same-date records {same_date(i)}")
    print(f"   claimed prev {p}: same-date {same_date(j)} | outdoor 23h->0h jump (z sum) {m23_0(j, i):.2f} | EC {EC[j,23]:.2f}->{EC[i,0] if TRAIN[i] else float('nan'):.2f} | sub_temp {ST[j,23]:.1f}->{ST[i,0]:.1f} | in_temp {IN_[j,23,0]:.1f}->{IN_[i,0,0]:.1f}")
# typical outdoor 23->0 jump for certain true links vs random
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
jj = [m23_0(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b)]; rr = [m23_0(a, b) for a, b in np.random.default_rng(1).integers(0, N, (500, 2))]
print("outdoor jump z-sum: certain links median %.2f (90%% %.2f) | random pairs median %.2f" % (np.median(jj), np.quantile(jj, .9), np.median(rr)))
