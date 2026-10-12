# -*- coding: utf-8 -*-
"""verify_leakage-stats (2026-10-13): increment 결과 반박용 점검. 기존 파일 수정 없음, 읽기만.
(1) 에피소드 라벨(comp, pre 사전 기반)과 INC2/INC4 경로(모든 a1->b 연결) 불일치 -> 교차적합 누수 여부
(2) INC4 k2: H(반반) vs 상수 앵커 H_c (앵커 정보 없는 단순 하향) vs 순열(앵커 섞기) 기준
(3) INC2 bwd k1/k2: 앵커 순열 기준, M 아래로 치우침만 고친 기준(상수 앵커)
(4) 같은 표적(target) 중복 사용 정도
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import EC, FARM, NUM, N, TRAIN, S3
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
inc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
T1 = pd.read_csv(os.path.join(inc, "inc1_targets_v1.csv"), index_col=0); y, comp = T1.y, T1.comp
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
def consec(a, b): return a in CAL.index and b in CAL.index and CAL.seg[a] == CAL.seg[b] and CAL.day[b] == CAL.day[a] + 1
L = [(rid(int(a)), rid(int(b))) for a, b in zip(TL.a1, TL.b)]
print("truth links total", len(L), "| distinct b", len(set(b for _, b in L)), "| distinct a", len(set(a for a, _ in L)))
Lc = [(a, b) for a, b in L if a in y.index and b in y.index and consec(a, b)]
nb = pd.Series([b for _, b in Lc]).value_counts(); na = pd.Series([a for a, _ in Lc]).value_counts()
print("consec train links", len(Lc), "| b with >1 predecessor", int((nb > 1).sum()), "| a with >1 successor", int((na > 1).sum()))
cross = [(a, b) for a, b in Lc if comp[a] != comp[b]]
print("links whose ends are in different comp labels:", len(cross), cross[:10])
print("cross-ID links (F13<->F47):", sum(a[:3] != b[:3] for a, b in Lc))

# ---- (1) INC2 cases: does target share comp with anchor?
C = pd.read_csv(os.path.join(inc, "inc2_cases_v1.csv"))
C["ep_t"] = C.t.map(comp)
print("\nINC2 cases where target comp != anchor comp:", int((C.ep_t != C.ep).sum()), "of", len(C))
# same target appearing under different ep labels
g = C.groupby("t").ep.nunique(); print("targets appearing with >1 ep label:", int((g > 1).sum()))
# target reuse
S1 = C[C.m >= 0.9]
for (d, k), gg in S1.groupby(["dir", "k"]):
    if k > 4: continue
    print(f"   S1 {d} k{k}: n {len(gg)}, distinct targets {gg.t.nunique()}, distinct anchors {gg.a.nunique()}, eps {gg.ep.nunique()}, max share of one ep {gg.ep.value_counts().max()/len(gg):.2f}")

rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
rng = np.random.default_rng(7)

def perm_test(G, est, label, nperm=4000, within_ep=False):
    """est(G_with_anc) -> predictions; permutation shuffles anchors across cases"""
    base = rm(est(G, G.anc.values), G.yt) - rm(G.m, G.yt)
    ds = []
    for _ in range(nperm):
        a = G.anc.values.copy()
        if within_ep:
            for e in G.ep.unique():
                ii = np.where(G.ep.values == e)[0]; a[ii] = rng.permutation(a[ii])
        else:
            a = rng.permutation(a)
        ds.append(rm(est(G, a), G.yt) - rm(G.m, G.yt))
    ds = np.array(ds)
    print(f"   {label}: observed diff {base:+.3f} | perm mean {ds.mean():+.3f}, perm 2.5% {np.quantile(ds,.025):+.3f} | p(perm <= obs) {np.mean(ds <= base):.3f}")

# ---- (2) INC4 k2 rebuild (identical to INC4 build)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
Mm = O.groupby(["seed", "rec"]).sg.mean().unstack(0).mean(axis=1)
succ = {}
for a, b in Lc: succ.setdefault(a, []).append(b)
def build(k, mid_normal=False, anchor_max=1.0):
    rows = []
    for a in y.index:
        if y[a] >= anchor_max: continue
        ps = [[a]]
        for _ in range(k): ps = [p + [q] for p in ps for q in succ.get(p[-1], [])]
        for p in ps:
            t = p[-1]
            if t not in Mm.index or Mm[t] < 0.9: continue
            mids = p[1:-1]
            if mid_normal and any((m_ not in Mm.index) or Mm[m_] >= 0.9 for m_ in mids): continue
            rows.append(dict(k=k, a=a, t=t, anc=EC[idx[a], 23], yanc=y[a], yt=y[t], m=Mm[t], ep=comp[a], ep_t=comp[t]))
    return pd.DataFrame(rows).drop_duplicates(["a", "t"]).reset_index(drop=True)

print("\n(2) INC4 normal anchor -> model-high target")
for k in (1, 2, 3):
    G = build(k)
    print(f" k{k}: n {len(G)}, distinct targets {G.t.nunique()}, eps {G.ep.nunique()}, ep(anchor)!=ep(target) {int((G.ep != G.ep_t).sum())}, ep counts {G.ep.value_counts().to_dict()}")
    H = lambda G_, a: (G_.m.values + a) / 2
    # constant-anchor control: replace anchor by out-of-episode mean anchor of the same case set (no anchor info)
    cst = np.array([G[G.ep != e].anc.mean() for e in G.ep])
    print(f"   RMSE M {rm(G.m, G.yt):.3f} | H(real anchor) {rm(H(G, G.anc.values), G.yt):.3f} | Hc(constant anchor, out-of-ep mean {cst.mean():.3f}) {rm(H(G, cst), G.yt):.3f} | M - out-of-ep mean bias {rm(G.m + np.array([ (G[G.ep!=e].yt - G[G.ep!=e].m).mean() for e in G.ep]), G.yt):.3f}")
    perm_test(G, H, f"k{k} H perm(all)")
    perm_test(G, lambda G_, a: a, f"k{k} P perm(all)")
    if k == 2:
        # leave-one-episode-out for H
        out = []
        for e in G.ep.unique():
            g2 = G[G.ep != e]; out.append(f"-{e}({int((G.ep==e).sum())}) {rm((g2.m+g2.anc)/2, g2.yt)-rm(g2.m, g2.yt):+.3f}")
        print("   LOEO H diff:", " | ".join(out))
        # per-case squared-error contributions
        G["dH"] = ((G.m + G.anc) / 2 - G.yt) ** 2 - (G.m - G.yt) ** 2
        G["dHc"] = ((G.m + cst) / 2 - G.yt) ** 2 - (G.m - G.yt) ** 2
        print(G[["a", "t", "ep", "anc", "yt", "m", "dH", "dHc"]].round(3).to_string(index=False))
        print(f"   share of total SSE reduction from top-2 cases: {np.sort(-G.dH.values)[-2:].sum() / (-G.dH.sum()):.2f}")
        # winner's curse check: mean residual y - M in this set vs in all train model-high days
        allhi = [r for r in Mm.index if r in y.index and Mm[r] >= .9]
        print(f"   mean(y-M) this set {np.mean(G.yt - G.m):+.3f} vs all train model-high days {np.mean(y[allhi] - Mm[allhi]):+.3f} (n {len(allhi)})")
        # gap distribution vs F13_237 gap 0.613
        print(f"   gap M-anchor: {np.sort((G.m - G.anc).values).round(3)}")

# ---- (3) INC2 bwd k1..4 S1: permutation of anchors and constant-anchor control (P)
print("\n(3) INC2 S1 bwd/fwd: anchor permutation baseline for P (anchor as is)")
for d, k in [("bwd", 1), ("bwd", 2), ("bwd", 3), ("bwd", 4), ("fwd", 2)]:
    G = S1[(S1.dir == d) & (S1.k == k)].reset_index(drop=True)
    perm_test(G, lambda G_, a: a, f"{d} k{k} P perm(all)")
    perm_test(G, lambda G_, a: a, f"{d} k{k} P perm(within ep)", within_ep=True)
