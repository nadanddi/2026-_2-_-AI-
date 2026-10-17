"""IM05p: 후보 G(GPT 가짜 추가) 비교 — IM_GPT가짜_v1.md. im05o(B2, 자료 B, +L2) 기반.
A = GPT 행 학습 제외(예측만), G1 = GPT 그냥 추가, G2 = GPT 가중을 유형 안 25%로 상향.
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05c_eval_v3 as V3  # noqa
V3.TYPE.update({"composite_v5b": "composite", "ai_gen_xl": "AI_gen", "full_style_xl": "full_style", "partial_style_xl": "partial_style", "partial_gen_xl": "partial_gen",
                "gpt_gen": "AI_gen", "gpt_fstyle": "full_style", "gpt_pgen": "partial_gen", "gpt_pstyle": "partial_style"})
import im05d_eval_v4 as D  # noqa
from im05k_backbones import bb  # noqa
from im05i_eval_L1 import lookfeats  # noqa
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); SETS = ["im_set_v4", "im_set_sd2", "im_set_v5", "im_set_gpt1"]; SHARE = .25
GT = ["AI_gen", "full_style", "partial_gen", "partial_style"]
def rule(p, L): hit = (L[:, 0] > 3.5) & (L[:, 0] < 20) & (L[:, 2] > 8); return np.where(hit, np.maximum(p, .9), p)

def main():
    rows, X, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk, EXk = L[keep], EX[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); gpt = np.array([r["_set"] == "im_set_gpt1" for r in R]); realm = kind == "real"
    print("행", len(R), "GPT", gpt.sum(), {x: int((gpt & (t == x)).sum()) for x in GT}, flush=True)
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R]); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    arms = {}
    for arm in ("A", "G1", "G2"):
        m = ~gpt if arm == "A" else np.ones(len(R), bool); w = np.zeros(len(R)); w[m] = V3.weights(kind[m])
        if arm == "G2":
            for x in GT:
                g, o_ = gpt & (t == x), ~gpt & (t == x); tot = w[g].sum() + w[o_].sum()
                w[g] = tot * SHARE / g.sum(); w[o_] *= tot * (1 - SHARE) / w[o_].sum()
        res = []
        for s in SEEDS:
            o = np.zeros(len(y))
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            p = rule((D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2, PL)
            o = rule(o, Lk); ng = ~gpt
            r = dict(gdet=(o[gpt] >= .5).mean(), gtype={x: (o[gpt & (t == x)] >= .5).mean() for x in GT}, fp=(o[realm] >= .5).mean(),
                     fp_ex=(o[realm & EXk] >= .5).mean(), bal=V3.bal_acc(o[ng], y[ng], t[ng]), ll=log_loss(py, np.clip(p, 1e-4, 1 - 1e-4)), pub=int(((p >= .5) == py).sum()),
                     gl1=float(np.mean(o[gpt])))
            res.append(r)
            print(f"  [{arm}] s{s} GPT탐지 {r['gdet']:.3f} (" + " ".join(f"{x[:7]}={v:.2f}" for x, v in r["gtype"].items()) + f") 평균p {r['gl1']:.2f} | 실제오탐 {r['fp']:.3f} 원본제외 {r['fp_ex']:.3f}"
                  f" | GPT외 균형 {r['bal']:.4f} | 공개 {r['pub']}/21 logloss {r['ll']:.3f}", flush=True)
        arms[arm] = res
    print("\n== 판정(IM_GPT가짜_v1.md, 시드 3개 모두) ==")
    A = arms["A"]; ok = {}
    for arm in ("G1", "G2"):
        G = arms[arm]
        c = [all(g["gdet"] > a["gdet"] for g, a in zip(G, A)), all(g["fp"] <= a["fp"] + .01 for g, a in zip(G, A)),
             all(g["bal"] >= a["bal"] - .003 for g, a in zip(G, A)), all(g["ll"] <= a["ll"] for g, a in zip(G, A))]
        ok[arm] = all(c)
        print(f"  {arm}: GPT탐지↑ {c[0]} · 오탐 {c[1]} · GPT외균형 {c[2]} · 공개logloss {c[3]} → {'후보' if all(c) else '불채택'}")
    if all(ok.values()):
        print("  둘 다 후보 → GPT 탐지율 평균 높은 쪽:", max(ok, key=lambda a: np.mean([r["gdet"] for r in arms[a]])))

if __name__ == "__main__":
    main()
