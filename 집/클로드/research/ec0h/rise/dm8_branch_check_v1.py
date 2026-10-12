# DM8: 1차 80번째 날짜(R80) 다음날이 두 갈래(1차 R81 묶음 vs 순수 2차 F13_216 묶음)인지, −14 연결은 무엇인지 확인
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
from collections import defaultdict
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}" + ("" if TRAIN[i] else "*")
T = pd.read_csv("dm7_calendar_v1.csv")
Xf = X.reshape(N, -1); par = list(range(N))
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i in range(N):
    eq = (np.abs(Xf[i + 1:] - Xf[i]) < 1e-6).mean(1)
    for j in np.where(eq >= .9)[0] + i + 1: par[fd(j)] = fd(i)
comp = np.array([fd(i) for i in range(N)]); G = defaultdict(list)
for i in range(N): G[comp[i]].append(i)
idx = {f"{FARM[i]}_{NUM[i]:03d}": i for i in range(N)}
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
def grp(name): return sorted(rid(i) for i in G[comp[idx[name]]])
def mc(a, b): return float(np.nansum(np.abs(X[idx[a], 23] - X[idx[b], 0]) / sdO))
print("group of F13_215 (R80?):", grp("F13_215"), "| of F13_216:", grp("F13_216"), "| of F13_122:", grp("F13_122"))
for a, b in zip(TLc.a1, TLc.b):
    a, b = int(a), int(b)
    if comp[a] == comp[idx["F13_215"]] or comp[b] == comp[idx["F13_216"]]:
        print(f"  certain link {rid(a)} -> {rid(b)} | EC {EC[a,23]:.2f}->{EC[b,0]:.2f} | outdoor 23->0 cont {np.nansum(np.abs(X[a,23]-X[b,0])/sdO):.2f}")
print("outdoor 23h of R80 group rep vs 0h of next pass-1 group and of F13_216:")
for nm in ("F13_122", "F13_123", "F13_216", "F47_211"):
    print(f"  F13_121 -> {nm}: cont {mc('F13_121', nm):.2f} | out_temp 23h {X[idx['F13_121'],23,0]:.1f} -> 0h {X[idx[nm],0,0]:.1f}")
# -14 link
D = T.set_index("rec")
for a, b in zip(TLc.a1, TLc.b):
    a, b = int(a), int(b); ra, rb = rid(a).rstrip("*"), rid(b).rstrip("*")
    if D.at[rb, "day"] - D.at[ra, "day"] < 0: print("backward certain link:", rid(a), "day", D.at[ra, "day"], "->", rid(b), "day", D.at[rb, "day"], f"| EC {EC[a,23]:.2f}->{EC[b,0]:.2f}, ST {ST[a,23]:.1f}->{ST[b,0]:.1f}")
# outdoor day profile comparison: pure window dates vs pass-1 dates of similar season (is the pure window a separate period?)
for nm in ("F13_216", "F13_224", "F13_231", "F13_237"):
    i = idx[nm]; d = np.array([np.nanmean(np.abs(X[i] - X[j]) / sdO) for j in range(N)]); d[comp == comp[i]] = 9
    o = np.argsort(d)[:3]; print(f"  {nm} (pure) nearest-weather other dates:", ", ".join(f"{rid(j)} {d[j]:.2f}" for j in o), "| out_temp mean %.1f" % np.nanmean(X[i, :, 0]))
