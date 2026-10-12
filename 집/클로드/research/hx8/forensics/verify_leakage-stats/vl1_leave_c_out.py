# -*- coding: utf-8 -*-
# verify_leakage-stats vl1: fx11 동 장부 검증의 두 의심점 재계산
#  (a) truth.pkl 은 b 마다 '모든 학습 기록' 중 최선 선행 a1 을 고른 표 -> 가짜 사례 c 자신의 정답이 후보로 들어가 c 의 진짜 다음날을
#      c 에게 묶어 둠 -> 쌍둥이 후속(배정)에서 c 의 진짜 다음날이 절대 빠지지 않음(평가에서는 c 정답이 없으니 이 보호가 없다).
#      여기서는 c 를 후보에서 뺀(leave-c-out) 연결표로 succ 를 다시 만든다.
#  (b) 보장 조건이 len(twins) 를 쓰는데 R = B - assigned 이므로 배정 안 된 쌍둥이의 후속도 R 에 남을 수 있음 -> 올바른 셈은 |R| > D-1-|assigned|.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)]); ix = {r: i for i, r in enumerate(rid)}
tr = np.where(TRAIN)[0]
sdA = np.nanstd(AC[tr].reshape(-1, 5), 0) + 1e-6
def acd(x, y): return np.nanmean(np.abs(AC[x] - AC[y]) / sdA)
key = {}
for i in tr: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
bund = {i: v for v in key.values() for i in v}
D = max(len(v) for v in key.values())
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
CE = np.abs(pe(EC, tr)[:, None] - pb(EC, tr)[None, :]) / 0.01 + np.abs(pe(ST, tr)[:, None] - pb(ST, tr)[None, :]) / 0.3
np.fill_diagonal(CE, np.inf)
top = np.argsort(CE, axis=0)[:3]  # per b column: top-3 predecessor positions
t = pd.read_pickle(S3 + r"\truth.pkl")
# sanity: reproduce truth c1
chk = []
for jb, b in enumerate(tr[:50]):
    q = t[t.b == b]
    if len(q): chk.append(abs(q.c1.values[0] - CE[top[0, jb], jb]))
print("reproduce truth c1 max abs diff (50 b):", np.nanmax(chk))
pos = {i: k for k, i in enumerate(tr)}
def succ_map(excl):
    S = {}
    for jb, b in enumerate(tr):
        if b == excl: continue
        o = [tr[k] for k in top[:, jb] if tr[k] != excl][:2]
        c1, c2 = CE[pos[o[0]], jb], CE[pos[o[1]], jb]
        if c1 <= 2.0 and c2 / max(c1, .05) >= 10: S.setdefault(int(o[0]), []).append(int(b))
    return S
S_full = succ_map(-1)
tt = t[t.a1.map(lambda a: TRAIN[a]) & t.b.map(lambda b: TRAIN[b])]
truth_next = {int(r.a1): int(r.b) for _, r in tt[tt.c1 <= 1.0].iterrows()}
Q90 = 0.623
def ledger(c, twins, succ):
    S = [s for k in twins for s in succ.get(k, [])]
    if not S: return None
    B = [k for k in bund.get(S[0], [S[0]]) if k != c]
    assigned = set()
    for k in twins:
        s = [x for x in succ.get(k, []) if x in B]
        if s: assigned.add(s[0]); continue
        rest = [x for x in B if x not in assigned]
        if rest:
            j = min(rest, key=lambda x: acd(k, x))
            if acd(k, j) <= Q90: assigned.add(j)
    R = [x for x in B if x not in assigned]
    return dict(R=R, g_orig=len(R) > D - len(twins) - 1, g_fix=len(R) > D - 1 - len(assigned),
                pick=min(R, key=lambda x: acd(c, x)) if R else None, nT=len(twins), nA=len(assigned))
P = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
P = P[P.tag == "CUR"].copy(); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int)
sgm = P.groupby(["validator", "seed", "rec"]).sg.mean()
rows = []
for c in tr:
    tw = [k for k in bund.get(c, []) if k != c]
    for mode, succ in (("orig", S_full), ("lco", succ_map(c))):
        L = ledger(c, tw, succ)
        if L is None or L["pick"] is None: continue
        tn = truth_next.get(c)
        rows.append(dict(rec=rid[c], mode=mode, nT=L["nT"], nA=L["nA"], nR=len(L["R"]), g_orig=L["g_orig"], g_fix=L["g_fix"],
                         has_truth=tn is not None, true_in_R=(tn in L["R"]) if tn is not None else np.nan,
                         pick_true=(tn == L["pick"]) if tn is not None else np.nan,
                         allnormal=all(EC[x, 0] < .9 for x in L["R"]), d0=EC[L["pick"], 0], y=EC[c].mean(), pick=rid[L["pick"]]))
A = pd.DataFrame(rows)
A.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vl1_cases_v1.csv"), index=False)
H = A[A.has_truth]
for mode in ("orig", "lco"):
    h = H[H["mode"] == mode]
    print(f"[{mode}] recs w/ ledger&truth {len(h)} | g_orig: n {int(h.g_orig.sum())} trueInR {h[h.g_orig].true_in_R.mean():.3f}, not {h[~h.g_orig].true_in_R.mean():.3f}"
          f" | g_fix: n {int(h.g_fix.sum())} trueInR {h[h.g_fix].true_in_R.mean():.3f}"
          f" | g_orig & nA<nT: n {int((h.g_orig & (h.nA < h.nT)).sum())} trueInR {h[h.g_orig & (h.nA < h.nT)].true_in_R.mean():.3f}"
          f" | g_orig&nA<nT&nR==2: n {int((h.g_orig & (h.nA < h.nT) & (h.nR==2)).sum())} trueInR {h[h.g_orig & (h.nA < h.nT)&(h.nR==2)].true_in_R.mean():.3f}")
# E1a firing per validator/seed under each mode & formula
for mode in ("orig", "lco"):
    for gcol in ("g_orig", "g_fix"):
        a = A[A["mode"] == mode].set_index("rec")
        out = []
        for (v, s), _ in sgm.groupby(level=[0, 1]):
            sv = sgm.loc[(v, s)]
            f = [r for r in a.index if r in sv.index and sv[r] >= 1.0 and a.loc[r, gcol] and a.loc[r, "allnormal"]]
            if f:
                m = np.sqrt(np.mean([(sv[r] - a.loc[r, "y"]) ** 2 for r in f])); e = np.sqrt(np.mean([(a.loc[r, "d0"] - a.loc[r, "y"]) ** 2 for r in f]))
            else: m = e = np.nan
            out.append(f"{v}/{s}:{f} {m:.3f}->{e:.3f}")
        print(f"E1a [{mode},{gcol}]", " | ".join(out))
c = ix["F13_237"]; tw = [k for k in tr if (np.abs(X[k] - X[c]) < 1e-6).all()]
L = ledger(c, tw, S_full)
print("F13_237:", {k: ([rid[x] for x in v] if k == "R" else (rid[v] if k == "pick" else v)) for k, v in L.items()})
