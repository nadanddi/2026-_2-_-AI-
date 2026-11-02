"""IM05v: 후보 L(마스크 지도 국소 조각 판별기) — IM_국소패치_L_v1.md. α=.025, 시드 3개 모두.
A = 제출 01 구성(OOF, 6판·GPT 학습 안 함). L = A + (조각 top1 ≥ θ → p ≥ .9). θ = OOF 진짜 1.0% 뒤집힘 기준(시드 평균).
GPT 부분 편집(v1+v2) 200장은 농장 crc32 % 2로 학습/검증 분할, 검증분은 어떤 학습에도 안 씀.
"""
import sys, zlib, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05t_eval_D1c as T  # noqa (TYPE·SETS·rule)
P, V3, D, bb, lookfeats = T.P, T.V3, T.D, T.bb, T.lookfeats
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); NB = 2000; FLIP = .010; SETS = T.SETS
OTHER4 = ["AI_gen", "composite", "filter", "full_style"]

def pfeat(sn):
    z = np.load(LOCAL / "feats" / f"{sn}_P16.npz"); return dict(zip(z["files"], z["feats"]))

def pmask(sn):
    p = LOCAL / "feats" / f"{sn}_pmask.npz"
    if not p.exists(): return {}
    z = np.load(p); ok = z["ok"] if "ok" in z.files else np.ones(len(z["files"]), bool)
    return {f: (o, bool(k)) for f, o, k in zip(z["files"], z["ov"], ok)}

def main():
    rows, X, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"; N = len(R)
    gptall = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v6 = sets == "im_set_v6"; gpart = gptall & np.isin(kind, ["gpt_pgen", "gpt_pstyle"])
    half = np.array([zlib.crc32(g.encode()) % 2 for g in grp]); val = gpart & (half == 1); gtrain = gptall & ~val
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    pf = {}
    for sn in SETS: pf.update({(sn, k): v for k, v in pfeat(sn).items()})
    PF = np.stack([pf[(r["_set"], r["file"])] for r in R]).astype(np.float32); NP = PF.shape[1]
    pm = {}
    for sn in SETS: pm.update({(sn, k): v for k, v in pmask(sn).items()})
    lab, pfl, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pfl]); pp16 = pfeat("public21"); PP = np.stack([pp16[f] for f in pfl]).astype(np.float32)
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pfl])
    # 조각 라벨 후보(행마다 고정 난수)
    ptr_idx, ptr_lab = [], []
    for i, r in enumerate(R):
        rng = np.random.default_rng(zlib.crc32(f"{r['_set']}|{r['file']}".encode()))
        key = (r["_set"], r["file"])
        if realm[i]: sel, lb = rng.choice(NP, 12, replace=False), 0
        elif kind[i] in ("gpt_gen", "gpt_fstyle"): sel, lb = rng.choice(NP, 12, replace=False), 1
        elif key in pm and pm[key][1]:
            ov = pm[key][0]; pos = np.where(ov >= .25)[0]; zero = np.where(ov == 0)[0]
            neg = rng.choice(zero, min(6, len(zero)), replace=False) if len(zero) else np.array([], int)
            ptr_idx.append((i, pos, 1)); ptr_idx.append((i, neg, 0)); continue
        else: continue
        ptr_idx.append((i, sel, lb))
    print("행", N, "GPT 검증 부분편집", val.sum(), "(부분생성", (val & (t == "partial_gen")).sum(), "부분스타일", (val & (t == "partial_style")).sum(), ")",
          "조각학습 후보 사진", len({a for a, _, _ in ptr_idx}), flush=True)
    def fit_patch(tr_rows, s):
        trs = set(tr_rows.tolist()); Xs, ys = [], []
        for i, sel, lb in ptr_idx:
            if i in trs and len(sel) and not val[i]: Xs.append(PF[i, sel]); ys.append(np.full(len(sel), lb))
        Xs = np.concatenate(Xs); ys = np.concatenate(ys); w = np.where(ys == 1, .5 / (ys == 1).sum(), .5 / (ys == 0).sum()) * len(ys)
        return make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=3000)).fit(Xs, ys, logisticregression__sample_weight=w)
    def top1(mdl, F): return mdl.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(len(F), -1).max(1)
    mA = ~gptall & ~v6; mP = ~val; wA = np.zeros(N); wA[mA] = V3.weights(kind[mA])
    pA, tp, pubA, pubT = [], [], [], []
    for s in SEEDS:
        o = np.zeros(N); q = np.zeros(N)
        for tr, te in V3.folds(grp, s):
            ta = tr[mA[tr]]; o[te] = (D.fit("clip", Xb, y, wA, ta, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, wA, ta, s).predict_proba(Xl[te])[:, 1]) / 2
            q[te] = top1(fit_patch(tr[mP[tr]], s), PF[te])
        pA.append(P.rule(o, Lk)); tp.append(q)
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); inb = np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False))
        ba = np.where(inb & mA)[0]; pubA.append(P.rule((D.fit("clip", Xb, y, wA, ba, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, wA, ba, s).predict_proba(PX["low"])[:, 1]) / 2, PL))
        pubT.append(top1(fit_patch(np.where(inb & mP)[0], s), PP))
        print(f"  s{s} 조각 top1 AUC(GPT검증 부분편집 대 진짜) {roc_auc_score(np.r_[np.zeros(realm.sum()), np.ones(val.sum())], np.r_[q[realm], q[val]]):.4f}"
              f" 6판 대 진짜 {roc_auc_score(np.r_[np.zeros(realm.sum()), np.ones(v6.sum())], np.r_[q[realm], q[v6]]):.4f}", flush=True)
    pA, tp = np.array(pA), np.array(tp); nr = realm.sum(); k = int(round(FLIP * nr)); th = []
    for a, q in zip(pA, tp):
        c = np.sort(q[realm & (a < .5)])[::-1]; th.append(float(c[k - 1]) if k >= 1 else 1.0)
    theta = float(np.mean(th)); print(f"θ 시드별 {[round(x, 4) for x in th]} → 평균 {theta:.4f}", flush=True)
    pL = np.where(tp >= theta, np.maximum(pA, .9), pA)
    ev4 = ~gptall & ~v6
    def est(p, idx=None):   # idx: 부트스트랩 행(중복 허용) → 횟수 가중
        wt = np.ones(N) if idx is None else np.bincount(idx, minlength=N).astype(float); hit = (p >= .5).astype(float)
        rate = lambda sel: (hit[sel] * wt[sel]).sum() / max(wt[sel].sum(), 1e-9)
        fp = rate(realm); det = [rate(ev4 & (t == x)) for x in OTHER4] + [rate(val & (t == x)) for x in ("partial_gen", "partial_style")]
        return .5 * (1 - fp) + .5 * np.mean(det), fp
    res = []
    for i, s in enumerate(SEEDS):
        a, l = pA[i], pL[i]; eA, fA = est(a); eL, fL = est(l)
        r = dict(eA=eA, eL=eL, fA=fA, fL=fL, vA=(a[val] >= .5).mean(), vL=(l[val] >= .5).mean()); res.append(r)
        print(f"  s{s} 추정정확도 A {eA:.4f} L {eL:.4f} | GPT검증 부분편집 탐지 A {r['vA']:.3f} L {r['vL']:.3f} (부분생성 {(a[val & (t == 'partial_gen')] >= .5).mean():.2f}→{(l[val & (t == 'partial_gen')] >= .5).mean():.2f}"
              f" 부분스타일 {(a[val & (t == 'partial_style')] >= .5).mean():.2f}→{(l[val & (t == 'partial_style')] >= .5).mean():.2f}) | 진짜오탐 A {fA:.4f} L {fL:.4f}"
              f" | 6판 A {(a[v6] >= .5).mean():.2f} L {(l[v6] >= .5).mean():.2f} | 유형(L,우리) " + " ".join(f"{x[:6]}={(l[ev4 & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6), flush=True)
        pl_ = np.where(pubT[i] >= theta, np.maximum(pubA[i], .9), pubA[i])
        print(f"     공개 A {int(((pubA[i] >= .5) == py).sum())}/21 L {int(((pl_ >= .5) == py).sum())}/21 | 공개 top1 " + " ".join(f"{f[:4]}({tt[:5]})={v:.2f}" for f, tt, v in zip(pfl, pt, pubT[i])), flush=True)
    Am, Lm = pA.mean(0), pL.mean(0); use = np.where(realm | val | (ev4 & np.isin(t, OTHER4)))[0]; farms = np.array(sorted(set(grp[use]))); fi = {f: use[grp[use] == f] for f in farms}
    rng = np.random.default_rng(7); bs = []
    for _ in range(NB):
        idx = np.concatenate([fi[f] for f in rng.choice(farms, len(farms))])
        if not (realm[idx].any() and (val & (t == "partial_gen"))[idx].any() and (val & (t == "partial_style"))[idx].any()): continue
        bs.append(est(Lm, idx)[0] - est(Am, idx)[0])
    pw = float((np.array(bs) <= 0).mean())
    c = [all(r["vL"] > r["vA"] for r in res), all(r["eL"] > r["eA"] for r in res), all(r["fL"] <= r["fA"] + .012 for r in res), pw < .025]
    print(f"\n== 판정(IM_국소패치_L_v1.md) ==\n  L: 부분편집↑ {c[0]} · 추정정확도↑ {c[1]} · 오탐 {c[2]} · p_worse {pw:.4f} ({c[3]}) → {'후보' if all(c) else '불채택'}  θ={theta:.4f}")
    np.savez(LOCAL / "preds" / "L_patch_oof.npz", pA=pA, tp=tp, theta=theta)

if __name__ == "__main__":
    main()
