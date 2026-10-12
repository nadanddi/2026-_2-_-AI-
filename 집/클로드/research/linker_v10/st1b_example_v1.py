# ST1b (2026-10-12 집 클로드; 사용자: 'F13_130_23 외기와 비슷하게 이어지는 0시 외기가 은근 있다는 거냐') 서술 예시:
# F13_130 의 같은 동 다음날(진단용 확실 연결)을 b 로 두고, 모든 기록의 23시 외기에서 b 의 0시 외기로의 자정 연속 비용(oj, v7 공식)을 계산해
# 진짜 전날(F13_130)보다 비용이 작거나 비슷한 기록이 몇 개인지, 그 기록들의 날짜.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAY = dict(zip(CAL.rec, CAL.day))
TLa = pd.read_pickle(S3 + r"\truth.pkl"); idx = {rid(i): i for i in range(N)}
succ = [rid(int(b)) for a, b, c in zip(TLa.a1, TLa.b, TLa.cert) if rid(int(a)) == "F13_130"]
print("next day of F13_130 (diagnostic link):", succ)
b = idx[succ[0]]; a_true = idx["F13_130"]
pa = X[:, 23] + .5 * (X[:, 23] - X[:, 22]); oj = np.sum(np.array([1, 1, .3, .5]) * np.abs(pa - X[b, 0]) / sdO, 1)
names = ["out_temp", "out_hum", "out_rad", "out_wspd"]
print(f"target {succ[0]} (day {DAY.get(succ[0])}) 0h outdoor: " + ", ".join(f"{n} {v:.1f}" for n, v in zip(names, X[b, 0])))
print(f"true prev F13_130 (day {DAY.get('F13_130')}) 23h outdoor: " + ", ".join(f"{n} {v:.1f}" for n, v in zip(names, X[a_true, 23])) + f" | cost {oj[a_true]:.2f}")
order = np.argsort(oj); rank_true = int(np.where(order == a_true)[0][0]) + 1
print(f"rank of true prev among all {N} records by outdoor midnight cost: {rank_true}")
print(f"records with cost <= true prev's: {int((oj <= oj[a_true]).sum()) - 1}, within +0.5 of it: {int((oj <= oj[a_true] + .5).sum()) - 1}")
print("cheapest 12 records:")
for i in order[:12]:
    print(f"   {rid(i)} day {DAY.get(rid(i))} split {'train' if TRAIN[i] else 'eval'} cost {oj[i]:.2f} | 23h " + ", ".join(f"{v:.1f}" for v in X[i, 23]))
