"""IM05q: 후보 G v2 — 생성기 하나(H=XL / SD2)를 빼고, GPT 추가가 그 미지 생성기 탐지로 옮겨 가는지 (IM_GPT가짜_v2.md).
안: A0(GPT 제외) / G1(GPT 추가) / G2(GPT 몫 25%). 주 지표 = H 가짜 L2 전 OOF 평균 확률. 진단 = GPT 자체 탐지(L2 전후), CLIP/저수준 단독, 세로만.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05p_eval_gpt as P  # noqa  (TYPE 갱신·rule·SETS 공유)
V3, D, bb, lookfeats = P.V3, P.D, P.bb, P.lookfeats
SEEDS = (0, 1, 2); XL = {"ai_gen_xl", "full_style_xl", "partial_gen_xl", "partial_style_xl"}; NB = 2000

def main():
    rows, X, isred = D.load(P.SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); gpt = sets == "im_set_gpt1"; realm = kind == "real"
    port = np.array([("768x1024" in r["info"]) if r["_set"] == "im_set_gpt1" else True for r in R])
    fz = {}
    for sn in P.SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    HS = {"XL": np.isin(kind, list(XL)), "SD2": (sets == "im_set_sd2") & (y == 1)}
    print("행", len(R), "GPT", gpt.sum(), "세로 GPT", (gpt & port).sum(), {h: int(v.sum()) for h, v in HS.items()}, flush=True)
    final = {}
    for hn, H in HS.items():
        out = {}
        for arm in ("A0", "G1", "G2"):
            m = ~H & (~gpt if arm == "A0" else True); w = np.zeros(len(R)); w[m] = V3.weights(kind[m])
            if arm == "G2":
                for x in P.GT:
                    g, o_ = gpt & (t == x), ~gpt & (t == x) & m; tot = w[g].sum() + w[o_].sum()
                    if w[o_].sum() == 0: continue   # 그 유형에 GPT 외 자료가 없으면 가중 그대로
                    w[g] = tot * P.SHARE / g.sum(); w[o_] *= tot * (1 - P.SHARE) / w[o_].sum()
            pre, oc, ol = [], [], []
            for s in SEEDS:
                c_ = np.zeros(len(y)); l_ = np.zeros(len(y))
                for tr, te in V3.folds(grp, s):
                    tr = tr[m[tr]]; c_[te] = D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1]; l_[te] = D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]
                o = (c_ + l_) / 2; pre.append(o); oc.append(c_); ol.append(l_); post = P.rule(o, Lk)
                print(f"  [{hn}/{arm}] s{s} H평균p(L2전) {o[H].mean():.4f} H탐지 L2전 {(o[H] >= .5).mean():.3f} 후 {(post[H] >= .5).mean():.3f} | 실제오탐(L2후) {(post[realm] >= .5).mean():.4f}"
                      f" | GPT탐지 L2전 {(o[gpt] >= .5).mean():.2f} 후 {(post[gpt] >= .5).mean():.2f} CLIP단독 {(c_[gpt] >= .5).mean():.2f} 저수준단독 {(l_[gpt] >= .5).mean():.2f} 세로만 {(o[gpt & port] >= .5).mean():.2f}", flush=True)
            out[arm] = (np.array(pre), m)
        A = out["A0"][0]; final[hn] = {}
        for arm in ("G1", "G2"):
            G = out[arm][0]; up = all(G[i][H].mean() > A[i][H].mean() for i in range(len(SEEDS)))
            fp = all((P.rule(G[i], Lk)[realm] >= .5).mean() <= (P.rule(A[i], Lk)[realm] >= .5).mean() + .01 for i in range(len(SEEDS)))
            d = (G.mean(0) - A.mean(0))[H]; fh = grp[H]; farms = np.array(sorted(set(fh))); rng = np.random.default_rng(7)
            idx = {f: np.where(fh == f)[0] for f in farms}; bs = []
            for _ in range(NB):
                pick = rng.choice(farms, len(farms)); bs.append(np.concatenate([d[idx[f]] for f in pick]).mean())
            pw = float((np.array(bs) <= 0).mean())
            final[hn][arm] = (up, pw < .025, fp, d.mean())
            print(f"  ==> {hn}/{arm}: H평균p 차 {d.mean():+.4f} 시드모두↑ {up} p_worse {pw:.4f} 오탐 {fp}", flush=True)
    print("\n== 판정(IM_GPT가짜_v2.md: H 2개 모두에서 3조건) ==")
    ok = {}
    for arm in ("G1", "G2"):
        ok[arm] = all(all(final[h][arm][:3]) for h in HS)
        print(f"  {arm}: " + " | ".join(f"{h}: ↑{final[h][arm][0]} p{final[h][arm][1]} 오탐{final[h][arm][2]}" for h in HS) + f" → {'후보' if ok[arm] else '불채택'}")
    if all(ok.values()): print("  둘 다 후보 → 상승 큰 쪽:", max(ok, key=lambda a: np.mean([final[h][a][3] for h in HS])))

if __name__ == "__main__":
    main()
