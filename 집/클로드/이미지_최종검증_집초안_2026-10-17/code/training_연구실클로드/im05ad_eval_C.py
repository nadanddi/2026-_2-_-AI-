"""IM05ad: 조합 후보 C1·C2·C3 — IM_조합후보_C_v1.md. 시드 6/7/8, 기준 A+R0(.5), α=.0083."""
import sys, re, zlib, numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05z_eval_shortcut as Z  # noqa
from im05ab_eval_L2rule import rule  # noqa
from im05v_eval_patch import pfeat, pmask  # noqa
P, V3, D, bb, lookfeats = Z.P, Z.V3, Z.D, Z.bb, Z.lookfeats
LOCAL = D.LOCAL; SETS = Z.SETS; SEEDS = (6, 7, 8); NB = 2000; ALPHA = .025 / 3; FLIP = .004; S2, HW = 224 * 224, 768 * 1024

def main():
    rows, Xd, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; EXk = EX[keep]; Xl = Xd["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v3 = sets == "im_set_gpt3"; rn = sets == "im_set_realn"
    test = np.isin(sets, ["im_set_gpt3", "im_set_gpt3r", "im_set_realn", "im_set_realg"]); ev = ~gtr & ~test; realE = ev & (kind == "real")
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    pf_ = {}
    for sn in SETS: pf_.update({(sn, k): v for k, v in pfeat(sn).items()})
    PF = np.stack([pf_[(r["_set"], r["file"])] for r in R]).astype(np.float32); NP = PF.shape[1]
    pm, garea = {}, {}
    for sn in SETS:
        pm.update({(sn, k): v for k, v in pmask(sn).items()}); p_ = LOCAL / "feats" / f"{sn}_pmask.npz"
        if p_.exists() and "area" in np.load(p_).files: zz = np.load(p_); garea.update({(sn, f): float(a) for f, a in zip(zz["files"], zz["area"])})
    lab, pfl, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pfl])
    pp16 = pfeat("public21"); PP = np.stack([pp16[f] for f in pfl]).astype(np.float32)
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pfl])
    gpart = gtr & np.isin(kind, ["gpt_pgen", "gpt_pstyle"])
    def make_ptr(seed):
        out = []
        for i, r in enumerate(R):
            if not (ev[i] or gtr[i]): continue
            key = (r["_set"], r["file"]); rng = np.random.default_rng(zlib.crc32(f"{seed}|{key}".encode()))
            if kind[i] == "real": out.append((i, rng.choice(NP, 12, replace=False), 0))
            elif kind[i] in ("gpt_gen", "gpt_fstyle"): out.append((i, rng.choice(NP, 12, replace=False), 1))
            elif key in pm and pm[key][1]:
                ov = pm[key][0]; area = garea.get(key) or float(re.search(r"area=([\d.]+)", r["info"]).group(1))
                pos = np.where((ov >= .25) | (ov * S2 >= .5 * area * HW))[0]
                if len(pos): out.append((i, pos, 1))
                if gpart[i]: continue
                zero = np.where(ov == 0)[0]
                if len(zero): out.append((i, rng.choice(zero, min(6, len(zero)), replace=False), 0))
        return out
    def fit_patch(ptr, tr):
        trs = set(tr.tolist()); Xs, ys = [], []
        for i, sel, lb in ptr:
            if i in trs: Xs.append(PF[i, sel]); ys.append(np.full(len(sel), lb))
        Xs = np.concatenate(Xs); ys = np.concatenate(ys); w = np.where(ys == 1, .5 / (ys == 1).sum(), .5 / (ys == 0).sum()) * len(ys)
        return make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=3000)).fit(Xs, ys, logisticregression__sample_weight=w)
    top1 = lambda mdl, F: mdl.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(len(F), -1).max(1)
    wA = np.zeros(N); wA[ev] = V3.weights(kind[ev]); mG = ev | gtr; wG = np.zeros(N); wG[mG] = V3.weights(kind[mG])
    for x in P.GT:
        g, o_ = gtr & (t == x), ev & (t == x); tot = wG[g].sum() + wG[o_].sum()
        if wG[o_].sum() and g.sum(): wG[g] = tot * .10 / g.sum(); wG[o_] *= tot * .90 / wG[o_].sum()
    EXP = {"E_pg": "partial_gen", "E_ps": "partial_style"}
    def wexp(ty):
        sel = (ev | gtr) & ((kind == "real") | (t == ty)); w = np.zeros(N); f = sel & (y == 1); r_ = sel & (y == 0); w[f] = .5 / f.sum(); w[r_] = .5 / r_.sum(); return sel, w * sel.sum()
    WE = {k: wexp(ty) for k, ty in EXP.items()}
    def balh(h, rs): return .5 * (1 - h[rs].mean()) + .5 * np.mean([h[ev & (t == x)].mean() for x in V3.TYPES6 if (ev & (t == x)).any()])
    out = {a: [] for a in ("A0", "C1", "C2", "C3")}; diag = []
    for s in SEEDS:
        ptr = make_ptr(s); oA = np.zeros(N); oG = np.zeros(N); E = {k: np.zeros(N) for k in ("E_pg", "E_ps", "E_L")}; fold = np.zeros(N, int)
        for f, (tr, te) in enumerate(V3.folds(grp, s)):
            ta = tr[ev[tr]]; tg = tr[mG[tr]]; fold[te] = f
            oA[te] = (D.fit("clip", Xb, y, wA, ta, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, wA, ta, s).predict_proba(Xl[te])[:, 1]) / 2
            oG[te] = (D.fit("clip", Xb, y, wG, tg, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, wG, tg, s).predict_proba(Xl[te])[:, 1]) / 2
            for k, (sel, w) in WE.items(): E[k][te] = D.fit("clip", Xb, y, w, tr[sel[tr]], s).predict_proba(Xb[te])[:, 1]
            E["E_L"][te] = top1(fit_patch(ptr, tr), PF[te])
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); inb = np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False))
        bA = np.where(inb & ev)[0]; bG = np.where(inb & mG)[0]
        qA = (D.fit("clip", Xb, y, wA, bA, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, wA, bA, s).predict_proba(PX["low"])[:, 1]) / 2
        qG = (D.fit("clip", Xb, y, wG, bG, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, wG, bG, s).predict_proba(PX["low"])[:, 1]) / 2
        qE = {k: D.fit("clip", Xb, y, w, np.where(inb & sel)[0], s).predict_proba(Pb)[:, 1] for k, (sel, w) in WE.items()}; qE["E_L"] = top1(fit_patch(ptr, np.where(inb & (ev | gtr))[0]), PP)
        pA0 = rule(oA, Lk, "R0")[0]; hA0 = pA0 >= .5; out["A0"].append(dict(h=hA0, pub=((rule(qA, PL, "R0")[0] >= .5) == py).sum()))
        def tau_fit(p):
            ths = [float(np.quantile(p[realE & (fold != f)], 1 - (pA0[realE & (fold != f)] >= .5).mean())) for f in range(5)]; return np.array(ths)[fold], float(np.mean(ths)), ths
        for arm, o, q in (("C1", oG, qG), ("C2", (oA + oG) / 2, (qA + qG) / 2)):
            p = rule(o, Lk, "R1")[0]; th, tp, ths = tau_fit(p); out[arm].append(dict(h=p >= th, pub=((rule(q, PL, "R1")[0] >= tp) == py).sum(), tau=ths))
        base = rule(oA, Lk, "R1")[0] >= .5; hC3 = base.copy(); pubC3 = rule(qA, PL, "R1")[0] >= .5; info = {}
        for k in ("E_pg", "E_ps", "E_L"):
            th = np.zeros(N); ths = []
            for f in range(5):
                c = np.sort(E[k][realE & (fold != f) & ~base])[::-1]; kk = int(round(FLIP * (realE & (fold != f)).sum())); ths.append(float(c[kk - 1]) if kk >= 1 else 1.0)
            th = np.array(ths)[fold]; fl = E[k] >= th; new_f = fl & ~hC3
            info[k] = (round(float(np.mean(ths)), 3), int((new_f & ev & (y == 1)).sum()), int((new_f & v3).sum()), int((new_f & realE).sum()))
            hC3 |= fl; pubC3 |= qE[k] >= np.mean(ths)
        out["C3"].append(dict(h=hC3, pub=(pubC3 == py).sum(), info=info))
        for arm in out:
            d = out[arm][-1]; h = d["h"]; d.update(bal=balh(h, realE), balK=balh(h, realE & ~EXk), det=h[v3].mean(), fp=h[realE].mean(), fn=h[rn].mean())
            print(f"  [{arm}] s{s} 균형 표준 {d['bal']:.4f} 원본있음 {d['balK']:.4f} | v3 {d['det']:.3f} | 진짜오탐 {d['fp']:.4f} realN {d['fn']:.3f} | 공개 {int(d['pub'])}/21 | 유형 "
                  + " ".join(f"{x[:6]}={h[ev & (t == x)].mean():.2f}" for x in V3.TYPES6) + (f" | τ {[round(x, 3) for x in d['tau']]}" if "tau" in d else "")
                  + (f" | 전문가(θ, 새로잡은 우리가짜, 새로잡은 v3, 새 진짜오탐) {d['info']}" if "info" in d else ""), flush=True)
    vi = np.where(v3)[0]; ri = np.where(realE)[0]; farms = np.array(sorted(set(grp[vi]) | set(grp[ri]))); fi = {f: (vi[grp[vi] == f], ri[grp[ri] == f]) for f in farms}
    H0 = np.mean([d["h"] for d in out["A0"]], 0)
    print("\n== 판정(IM_조합후보_C_v1.md) =="); ok = {}
    for arm in ("C1", "C2", "C3"):
        Hk = np.mean([d["h"] for d in out[arm]], 0); rng = np.random.default_rng(7); bs = []
        for _ in range(NB):
            pick = rng.choice(farms, len(farms)); a_ = np.concatenate([fi[f][0] for f in pick]); b_ = np.concatenate([fi[f][1] for f in pick])
            if len(a_) == 0 or len(b_) == 0: continue
            bs.append(((Hk[a_] >= .5).mean() - (Hk[b_] >= .5).mean()) - ((H0[a_] >= .5).mean() - (H0[b_] >= .5).mean()))
        pw = float((np.array(bs) <= 0).mean()); A_, C_ = out["A0"], out[arm]
        c = [all(b["bal"] >= a["bal"] - .001 for a, b in zip(A_, C_)), all(b["balK"] >= a["balK"] - .002 for a, b in zip(A_, C_)),
             all(b["det"] >= a["det"] + .05 for a, b in zip(A_, C_)), all(b["pub"] >= a["pub"] for a, b in zip(A_, C_)), pw < ALPHA]; ok[arm] = all(c)
        print(f"  {arm}: 균형비열등 {c[0]} · 원본있음비열등 {c[1]} · v3+.05 {c[2]} · 공개 {c[3]} · p_worse {pw:.4f} ({c[4]}) → {'후보' if ok[arm] else '불채택'}")
    cands = [a for a in ("C1", "C2", "C3") if ok[a]]
    if len(cands) > 1: print("  후보 여럿 → v3+균형 합 최대:", max(cands, key=lambda a: np.mean([d["det"] + d["bal"] for d in out[a]])))

if __name__ == "__main__":
    main()
