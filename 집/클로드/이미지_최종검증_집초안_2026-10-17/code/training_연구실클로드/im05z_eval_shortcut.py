"""IM05z: G3 지름길 반증(IM_G3_지름길반증_v1.md). A / G3a, 시드 0/1/2. 시험 세트(v3·v3R·realN·realG)는 학습에 안 씀."""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05x_eval_G3 as X  # noqa
P, V3, D, bb, lookfeats = X.P, X.V3, X.D, X.bb, X.lookfeats
V3.TYPE.update({"real_n": "real", "real_g": "real"})
SEEDS = (0, 1, 2); SETS = X.SETS + ["im_set_gpt3r", "im_set_realn", "im_set_realg"]

def main():
    rows, Xd, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = Xd["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"; N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v3 = sets == "im_set_gpt3"; v3r = sets == "im_set_gpt3r"
    rn = sets == "im_set_realn"; rg = sets == "im_set_realg"; test = v3 | v3r | rn | rg; ev = ~gtr & ~test
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    print("v3", v3.sum(), "v3R", v3r.sum(), "realN", rn.sum(), "realG", rg.sum(), flush=True)
    auc = lambda q, sel: roc_auc_score(np.r_[np.zeros(realm.sum()), np.ones(sel.sum())], np.r_[q[realm], q[sel]])
    res = {}
    for arm in ("A", "G3a"):
        m = ev.copy() if arm == "A" else (ev | gtr); w = np.zeros(N); w[m] = V3.weights(kind[m]); rr = []
        for s in SEEDS:
            c_ = np.zeros(N); l_ = np.zeros(N)
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]; c_[te] = D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1]; l_[te] = D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]
            p = P.rule((c_ + l_) / 2, Lk)
            r = dict(d3=(p[v3] >= .5).mean(), d3r=(p[v3r] >= .5).mean(), c3=auc(c_, v3), c3r=auc(c_, v3r), l3=auc(l_, v3), l3r=auc(l_, v3r),
                     fn=(p[rn] >= .5).mean(), fg=(p[rg] >= .5).mean(), foof=(p[realm] >= .5).mean()); rr.append(r)
            print(f"  [{arm}] s{s} 탐지 v3 {r['d3']:.3f} v3R {r['d3r']:.3f} | CLIP AUC v3 {r['c3']:.3f} v3R {r['c3r']:.3f} | 저수준 AUC v3 {r['l3']:.3f} v3R {r['l3r']:.3f}"
                  f" | 진짜 오탐 realN {r['fn']:.3f} realG {r['fg']:.3f} (OOF 진짜 {r['foof']:.3f})", flush=True)
        res[arm] = rr
    A, G = res["A"], res["G3a"]
    s1 = any(g["d3r"] < g["d3"] - .10 or g["c3r"] < g["c3"] - .03 for g in G)
    s2 = any((g["fg"] - g["fn"]) > (a["fg"] - a["fn"]) + .03 for g, a in zip(G, A))
    print(f"\n== 판정(IM_G3_지름길반증_v1.md) ==\n  S1(경로 바꾸면 효과 사라짐) {'걸림' if s1 else '안 걸림'} · S2(GPT 경로 진짜를 가짜로) {'걸림' if s2 else '안 걸림'}"
          f" → {'지름길 있음' if (s1 or s2) else '지름길 없음'}")

if __name__ == "__main__":
    main()
