# DM3: 날짜가 어긋난 1차 기록(F47_178, F47_074)을 같은 날짜 형제·번호 이웃과 원자료로 나란히 비교 (서술)
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
def show(names, hrs=(0, 4, 8, 12, 16, 20, 23)):
    for nm, arr, k in (("out_temp", X, 0), ("out_rad", X, 2), ("in_temp", IN_, 0), ("in_hum", IN_, 1), ("heating", AC, 3), ("thermal", AC, 2)):
        print(f"   {nm:8s} " + " | ".join(f"{n}: " + " ".join(f"{arr[idx[n], h, k]:5.1f}" for h in hrs) for n in names))
    for nm, arr in (("sub_temp", ST), ("sub_ec", EC)):
        print(f"   {nm:8s} " + " | ".join(f"{n}: " + " ".join(f"{arr[idx[n], h]:5.2f}" if TRAIN[idx[n]] else "  nan" for h in hrs) for n in names))
for tgt, sib, nb in (("F47_178", ["F47_008", "F13_009"], ["F47_177", "F47_179"]), ("F47_074", ["F47_073", "F13_049"], ["F47_075", "F47_072"])):
    print(f"\n=== {tgt}: same-date siblings {sib} | number neighbours {nb}  (hours 0,4,8,12,16,20,23)")
    show([tgt] + sib); print("   -- vs number neighbours"); show([tgt] + nb)
    i = idx[tgt]
    for n in sib + nb:
        j = idx[n]; print(f"   outdoor equal share with {n}: {(np.abs(X[i]-X[j])<1e-6).mean():.2f} | in_temp mean abs diff {np.nanmean(np.abs(IN_[i,:,0]-IN_[j,:,0])):.1f} | sub_temp diff {np.nanmean(np.abs(ST[i]-ST[j])):.1f}")
