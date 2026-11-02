"""IM05w: 후보 L v2 — IM_국소패치_L_v2.md (v1 im05v 기반, 비평 반영).
변경: 양성 = 겹침≥.25 또는 마스크 픽셀 50% 이상 담은 조각 / GPT 부분편집은 음성 조각 안 씀 / θ 교차 적합(폴드 f는 다른 폴드 OOF 진짜로) /
판정1 최소 +.03 / 조각 표본 시드별 / farm_code 검산.
"""
import sys, re, zlib, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05t_eval_D1c as T  # noqa
from im05v_eval_patch import pfeat, pmask  # noqa
P, V3, D, bb, lookfeats = T.P, T.V3, T.D, T.bb, T.lookfeats
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); NB = 2000; FLIP = .010; SETS = T.SETS; S2 = 224 * 224; HW = 768 * 1024
OTHER4 = ["AI_gen", "composite", "filter", "full_style"]

def main():
    rows, X, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"; N = len(R)
    for r in R:
        if r["_set"] in ("im_set_v6", "im_set_v5") and r["source"]: assert r["farm_code"] == Path(r["source"].replace("\\", "/")).stem.split("_")[0], r["file"]
    gptall = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v6 = sets == "im_set_v6"; gpart = gptall & np.isin(kind, ["gpt_pgen", "gpt_pstyle"])
    half = np.array([zlib.crc32(g.encode()) % 2 for g in grp]); val = gpart & (half == 1)
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    pf = {}
    for sn in SETS: pf.update({(sn, k): v for k, v in pfeat(sn).items()})
    PF = np.stack([pf[(r["_set"], r["file"])] for r in R]).astype(np.float32); NP = PF.shape[1]
    pm, garea = {}, {}
    for sn in SETS:
        pm.update({(sn, k): v for k, v in pmask(sn).items()})
        p = LOCAL / "feats" / f"{sn}_pmask.npz"
        if p.exists() and "area" in np.load(p).files: z = np.load(p); garea.update({(sn, f): float(a) for f, a in zip(z["files"], z["area"])})
    lab, pfl, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pfl]); pp16 = pfeat("public21"); PP = np.stack([pp16[f] for f in pfl]).astype(np.float32)
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pfl])
    def make_ptr(seed):
        out, npos_gpt = [], []
        for i, r in enumerate(R):
            key = (r["_set"], r["file"]); rng = np.random.default_rng(zlib.crc32(f"{seed}|{key}".encode()))
            if realm[i]: out.append((i, rng.choice(NP, 12, replace=False), 0))
            elif kind[i] in ("gpt_gen", "gpt_fstyle"): out.append((i, rng.choice(NP, 12, replace=False), 1))
            elif key in pm and pm[key][1]:
                ov = pm[key][0]; area = garea.get(key) or float(re.search(r"area=([\d.]+)", r["info"]).group(1))
                pos = np.where((ov >= .25) | (ov * S2 >= .5 * area * HW))[0]
                if len(pos): out.append((i, pos, 1))
                if gpart[i]: npos_gpt.append(len(pos)); continue        # GPT는 음성 조각 없음
                zero = np.where(ov == 0)[0]
                if len(zero): out.append((i, rng.choice(zero, min(6, len(zero)), replace=False), 0))
        return out, npos_gpt
    ptr0, ng = make_ptr(0)
    print("행", N, "GPT 검증 부분편집", val.sum(), "(부분생성", (val & (t == "partial_gen")).sum(), "부분스타일", (val & (t == "partial_style")).sum(), ")",
          "| GPT 신뢰 마스크 사진의 양성 조각 0개 비율", round(float(np.mean(np.array(ng) == 0)), 3), "평균", round(float(np.mean(ng)), 2), flush=True)
    def fit_patch(ptr, tr_rows):
        trs = set(tr_rows.tolist()); Xs, ys = [], []
        for i, sel, lb in ptr:
            if i in trs and not val[i]: Xs.append(PF[i, sel]); ys.append(np.full(len(sel), lb))
        Xs = np.concatenate(Xs); ys = np.concatenate(ys); w = np.where(ys == 1, .5 / (ys == 1).sum(), .5 / (ys == 0).sum()) * len(ys)
        return make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=3000)).fit(Xs, ys, logisticregression__sample_weight=w)
    def top1(mdl, F): return mdl.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(len(F), -1).max(1)
    def theta_from(a, q, sel):
        c = np.sort(q[sel & realm & (a < .5)])[::-1]; k = int(round(FLIP * (sel & realm).sum())); return float(c[k - 1]) if k >= 1 else 1.0
    mA = ~gptall & ~v6; mP = ~val; wA = np.zeros(N); wA[mA] = V3.weights(kind[mA]); ev4 = ~gptall & ~v6
    def est(p, idx=None):
        wt = np.ones(N) if idx is None else np.bincount(idx, minlength=N).astype(float); hit = (p >= .5).astype(float)
        rate = lambda sel: (hit[sel] * wt[sel]).sum() / max(wt[sel].sum(), 1e-9)
        fp = rate(realm); det = [rate(ev4 & (t == x)) for x in OTHER4] + [rate(val & (t == x)) for x in ("partial_gen", "partial_style")]
        return .5 * (1 - fp) + .5 * np.mean(det), fp
    pA, pL, res = [], [], []
    for s in SEEDS:
        ptr = ptr0 if s == 0 else make_ptr(s)[0]; o = np.zeros(N); q = np.zeros(N); fold = np.zeros(N, int)
        for f, (tr, te) in enumerate(V3.folds(grp, s)):
            ta = tr[mA[tr]]; o[te] = (D.fit("clip", Xb, y, wA, ta, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, wA, ta, s).predict_proba(Xl[te])[:, 1]) / 2
            q[te] = top1(fit_patch(ptr, tr[mP[tr]]), PF[te]); fold[te] = f
        a = P.rule(o, Lk); ths = [theta_from(a, q, fold != f) for f in range(5)]; th = np.array(ths)[fold]
        l = np.where(q >= th, np.maximum(a, .9), a); pA.append(a); pL.append(l)
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); inb = np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False))
        ba = np.where(inb & mA)[0]; pa = P.rule((D.fit("clip", Xb, y, wA, ba, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, wA, ba, s).predict_proba(PX["low"])[:, 1]) / 2, PL)
        pq = top1(fit_patch(ptr, np.where(inb & mP)[0]), PP); pl = np.where(pq >= np.mean(ths), np.maximum(pa, .9), pa)
        eA, fA = est(a); eL, fL = est(l); r = dict(eA=eA, eL=eL, fA=fA, fL=fL, vA=(a[val] >= .5).mean(), vL=(l[val] >= .5).mean()); res.append(r)
        aucv = roc_auc_score(np.r_[np.zeros(realm.sum()), np.ones(val.sum())], np.r_[q[realm], q[val]])
        print(f"  s{s} θ폴드 {[round(x, 3) for x in ths]} | 조각top1 AUC GPT검증 {aucv:.4f} 6판 {roc_auc_score(np.r_[np.zeros(realm.sum()), np.ones(v6.sum())], np.r_[q[realm], q[v6]]):.4f}"
              f" | 추정정확도 A {eA:.4f} L {eL:.4f} | GPT검증 부분편집 A {r['vA']:.3f} L {r['vL']:.3f} (부분생성 {(a[val & (t == 'partial_gen')] >= .5).mean():.2f}→{(l[val & (t == 'partial_gen')] >= .5).mean():.2f}"
              f" 부분스타일 {(a[val & (t == 'partial_style')] >= .5).mean():.2f}→{(l[val & (t == 'partial_style')] >= .5).mean():.2f}) | 진짜오탐 A {fA:.4f} L {fL:.4f}"
              f" | 6판 A {(a[v6] >= .5).mean():.2f} L {(l[v6] >= .5).mean():.2f} | 유형(L) " + " ".join(f"{x[:6]}={(l[ev4 & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6), flush=True)
        print(f"     공개 A {int(((pa >= .5) == py).sum())}/21 L {int(((pl >= .5) == py).sum())}/21 | 공개 진짜 top1 분위(OOF 진짜 대비) "
              + " ".join(f"{f[:4]}={np.mean(q[realm] < v):.3f}" for f, tt, v in zip(pfl, pt, pq) if tt == "real") + " | 공개 부분편집 top1 "
              + " ".join(f"{f[:4]}({tt[8:13]})={v:.2f}" for f, tt, v in zip(pfl, pt, pq) if tt.startswith("partial")), flush=True)
    Am, Lm = np.mean(pA, 0), np.mean(pL, 0); use = np.where(realm | val | (ev4 & np.isin(t, OTHER4)))[0]; farms = np.array(sorted(set(grp[use]))); fi = {f: use[grp[use] == f] for f in farms}
    rng = np.random.default_rng(7); bs = []
    for _ in range(NB):
        idx = np.concatenate([fi[f] for f in rng.choice(farms, len(farms))])
        if not (realm[idx].any() and (val & (t == "partial_gen"))[idx].any() and (val & (t == "partial_style"))[idx].any()): continue
        bs.append(est(Lm, idx)[0] - est(Am, idx)[0])
    pw = float((np.array(bs) <= 0).mean())
    c = [all(r["vL"] >= r["vA"] + .03 for r in res), all(r["eL"] > r["eA"] for r in res), all(r["fL"] <= r["fA"] + .012 for r in res), pw < .025]
    print(f"\n== 판정(IM_국소패치_L_v2.md) ==\n  L: 부분편집≥+.03 {c[0]} · 추정정확도↑ {c[1]} · 오탐 {c[2]} · p_worse {pw:.4f} ({c[3]}) → {'후보' if all(c) else '불채택'}")
    np.savez(LOCAL / "preds" / "L2_patch_oof.npz", pA=np.array(pA), pL=np.array(pL))

if __name__ == "__main__":
    main()
