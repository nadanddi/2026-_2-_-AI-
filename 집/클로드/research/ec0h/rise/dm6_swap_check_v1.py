# DM6: F47_148 과 F13_147 (형제와 하루 종일 먼 두 기록) — 실내가 서로 바뀐(다른 날짜 실내가 붙은) 것인지: 교차 거리 + 시각별 원자료
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
idx = {f"{FARM[i]}_{NUM[i]:03d}": i for i in range(N)}
sd = np.nanstd(IN_.reshape(-1, 3), 0)
def dist(a, b, hs=slice(0, 24)): return np.nansum(np.abs(IN_[idx[a], hs] - IN_[idx[b], hs]) / sd, 1)
names = ["F47_148", "F13_147", "F13_145", "F13_146", "F47_149", "F47_235", "F47_147", "F47_150", "F13_148"]
print("mean hourly indoor distance matrix:")
print("          " + " ".join(f"{n:>8s}" for n in names))
for a in names[:2]:
    print(f"{a:9s} " + " ".join(f"{np.nanmean(dist(a, b)):8.1f}" for b in names))
# search: for each of the two records, the training/eval record anywhere whose indoor 0..23 is closest (any date) — is their indoor a copy/shift of another record?
for a in ("F47_148", "F13_147"):
    i = idx[a]; d = np.array([np.nanmean(np.nansum(np.abs(IN_[i] - IN_[j]) / sd, 1)) if j != i else np.inf for j in range(N)])
    o = np.argsort(d)[:5]; print(a, "closest indoor records anywhere:", ", ".join(f"{FARM[j]}_{NUM[j]:03d}{'' if TRAIN[j] else '*'} {d[j]:.2f}" for j in o))
for a in ("F47_148", "F13_147", "F47_149", "F13_146"):
    i = idx[a]; print(f"{a}: in_temp " + " ".join(f"{IN_[i,h,0]:.1f}" for h in range(24)))
    print(f"{'':8s} in_co2  " + " ".join(f"{IN_[i,h,2]:.0f}" for h in range(24)))
    print(f"{'':8s} out_temp " + " ".join(f"{X[i,h,0]:.1f}" for h in range(24)))
    if TRAIN[i]: print(f"{'':8s} sub_temp " + " ".join(f"{ST[i,h]:.1f}" for h in range(24)) + " | EC " + " ".join(f"{EC[i,h]:.2f}" for h in (0, 12, 23)))
