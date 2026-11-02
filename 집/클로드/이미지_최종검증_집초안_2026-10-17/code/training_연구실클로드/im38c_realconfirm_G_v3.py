"""IM38c: 후보 G v3 — IM_진짜확정조건_G_v3.md(v2 + R2 묶음·봉인 원천 제외, v4만). 서버(nadan, GPU 0). G-LR(특징 5) vs G-1(|glob_e|) 진짜 확정 조건, 본페로니."""
import sys, csv, json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im36_finefp_eval import load_db, rows_of, L, B  # noqa
from im37b_f2_eval_v2 import fe_paths, run_pkg, NB  # noqa

FILT = ("filter_jpeg", "filter_noise", "filter_blur"); REAL = {"real", "real_resampled"}
PH = {"LR": lambda X: np.c_[X[:, 0], X[:, 2], X[:, 3], X[:, 4], np.abs(X[:, 4])], "G1": lambda X: np.abs(X[:, 4:5])}
stem = lambda s: Path(str(s).replace("\\", "/")).stem

def main():
    names, DG, DE = load_db(); T = json.load(open(L / "preds/finefp_thresholds_v2.json")); print("정밀 DB", len(names), flush=True)
    r2 = rows_of("im_set_r2"); z = np.load(L / "preds/sealed_c3v2.npz", allow_pickle=True); st, pp5, sp = z["set"], z["p5"], z["path"]
    bg = {r["group"] for r in r2}; bs_ = {stem(r["source"]) for r in r2 if r.get("source")}   # R2: 묶음+원천, 봉인: 원천만(v3)
    for sn in ("real", "brown", "bright", "qwen", "v7"):
        for r in csv.DictReader(open(L / "sealed_c3" / sn / "manifest.csv", encoding="utf-8")): bs_.add(stem(r["source"]))
    M = lambda X: (X[:, 0] <= T["m"]) & (X[:, 1] >= T["gap"])
    FK = lambda X: M(X) & ((X[:, 2] > T["f_g"]) | (X[:, 3] > T["f_e"]))
    RC = lambda X: M(X) & ~FK(X) & (X[:, 2] <= T["r_g"]) & (X[:, 3] <= T["r_e"]) & (np.abs(X[:, 4]) <= T["r_glob"])
    allr = [(sn, r) for sn in ("im_set_v4",) for r in rows_of(sn) if r["kind"] in FILT or r["kind"] in REAL]
    tr = [(sn, r) for sn, r in allr if r["group"] not in bg and stem(r.get("source", "")) not in bs_]
    print(f"학습 후보 {len(allr)} → 묶음·원천 제외 후 {len(tr)}", flush=True)
    fx = L / "preds/g_train_X_v3.npy"
    if fx.exists(): Xt = np.load(fx)
    else: Xt = fe_paths([str(L / sn / "img" / r["file"]) for sn, r in tr], DG, DE); np.save(fx, Xt)
    yt = np.array([int(r["kind"] in FILT) for _, r in tr]); kt = np.array([r["kind"] for _, r in tr]); use = M(Xt) & ~FK(Xt)
    print(f"원본판정·비가짜확정 {use.sum()} (진짜 {(use & (yt == 0)).sum()}, 필터 {(use & (yt == 1)).sum()})", flush=True); assert (use & (yt == 0)).sum() >= 30 and (use & (yt == 1)).sum() >= 30
    rr = RC(Xt) & (yt == 0); rf = RC(Xt) & (yt == 1); assert rr.sum() >= 30
    def fit(ph, sel):
        sc = StandardScaler().fit(ph(Xt[sel])); lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(ph(Xt[sel])), yt[sel])
        S = lambda X: lr.decision_function(sc.transform(ph(X))); return S, float(np.quantile(S(Xt[rr]), .975))
    blk = lambda S, t, m: float((S(Xt[m]) >= t).mean()) if m.any() else float("nan")   # 빈 집합 보호(실행 오류 수정)
    G = {k: fit(PH[k], use) for k in PH}
    for k, (S, t) in G.items():
        print(f"[{k}] t={t:.3f} | 학습: 진짜확정 가능 진짜 {rr.sum()} 유지 {(S(Xt[rr]) < t).mean():.3f}, 확정조건 통과 필터 {rf.sum()} 차단 {blk(S, t, rf):.3f} (" + ", ".join(f"{f} {blk(S, t, rf & (kt == f)):.2f} (n={(rf & (kt == f)).sum()})" for f in FILT) + ")", flush=True)
    for f in FILT:   # 필터 하나 빼고 학습 → 뺀 종류 차단율
        S, t = fit(PH["LR"], use & (kt != f)); m = rf & (kt == f); print(f"   [LR, {f} 제외 학습] 뺀 종류 차단 {blk(S, t, m):.3f} (n={m.sum()})", flush=True)
    def rule(p, X, k):
        fk = FK(X); rl = RC(X) & (p < .8)
        if k is not None: S, t = G[k]; rl &= S(X) < t
        q = p.copy(); q[fk] = np.maximum(q[fk], .95); q[rl] = np.minimum(q[rl], .3); return q
    yr = np.array([int(r["label"]) for r in r2]); gr = np.array([r["group"] for r in r2]); kd = np.array([r["kind"] for r in r2]); p5 = np.load(L / "preds/_r2_나단5.npy")
    zz = np.load(L / "preds/f2_eval_v2.npz"); Xr = zz["Xr"]; q6 = rule(p5, Xr, None); assert np.abs(q6 - zz["q6"]).max() < 1e-9
    acc = lambda p, y, ix: .5 * (p[ix][y[ix] == 0] < .5).mean() + .5 * (p[ix][y[ix] == 1] >= .5).mean(); allix = np.arange(len(yr))
    gs_ = np.array(sorted(set(gr))); gi = {g: np.where(gr == g)[0] for g in gs_}; rng = np.random.default_rng(38); boots = [np.concatenate([gi[g] for g in rng.choice(gs_, len(gs_))]) for _ in range(NB)]
    PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; lab = list(csv.DictReader(open(PUB / "sample_labels.csv"))); pf = [str(PUB / "images" / r["file"]) for r in lab]
    py = np.array([int(r["label"]) for r in lab]); pp = np.load(L / "preds/_pub_n5.npy"); Xp = fe_paths(pf, DG, DE)
    SX = {sn: fe_paths(list(sp[st == sn]), DG, DE) for sn in ("real", "brown", "bright", "qwen", "v7")}
    hp = sorted(str(p) for p in (L / "preds/_jig_hold").glob("j*.jpg")); ph = np.load(L / "preds/_jig_hold_n5.npy"); Xh = fe_paths(hp, DG, DE)
    h1 = [r for r in rows_of("im_set_h1") if r["group"] not in bg and stem(r.get("source", "")) not in bs_]; yh = np.array([int(r["label"]) for r in h1]); hpth = [str(L / "im_set_h1/img" / r["file"]) for r in h1]
    p5h = run_pkg(B / "submit_img05/팜모니_나단5", hpth, "h1x3_n5"); Xh1 = fe_paths(hpth, DG, DE); ih = np.arange(len(yh))
    RES = {}
    for k in ("LR", "G1"):
        qg = rule(p5, Xr, k); d = [acc(qg, yr, ix) - acc(q6, yr, ix) for ix in boots if (yr[ix] == 0).any() and (yr[ix] == 1).any()]; pw = float((np.array(d) <= 0).mean())
        a6, ag = acc(q6, yr, allix), acc(qg, yr, allix); print(f"\n[{k}] R2 정확도 나단6 {a6:.4f} → {ag:.4f}, p_worse {pw:.4f}", flush=True)
        for kk in sorted(set(kd)):
            m = kd == kk; print(f"     {kk:14s} n={m.sum():4d} 나단5 {((p5[m] >= .5) == (yr[m] == 1)).mean():.3f} 나단6 {((q6[m] >= .5) == (yr[m] == 1)).mean():.3f} {k} {((qg[m] >= .5) == (yr[m] == 1)).mean():.3f}", flush=True)
        fpd = 0
        for sn in ("real", "brown", "bright", "qwen", "v7"):
            m = st == sn; a, b = rule(pp5[m], SX[sn], None), rule(pp5[m], SX[sn], k)
            if sn in ("real", "brown", "bright"): fpd += int((b >= .5).sum() - (a >= .5).sum())
            print(f"     [봉인 {sn}] 양성 비율 나단6 {(a >= .5).mean():.4f} → {(b >= .5).mean():.4f}", flush=True)
        h6, hg = (rule(ph, Xh, None) >= .5).mean(), (rule(ph, Xh, k) >= .5).mean()
        c6, cg = int(((rule(pp, Xp, None) >= .5) == py).sum()), int(((rule(pp, Xp, k) >= .5) == py).sum()); S, t = G[k]
        mg = [float(t - s) for s, y, x in zip(S(Xp), py, Xp) if y == 0 and M(x[None])[0]]
        a6h, agh = acc(rule(p5h, Xh1, None), yh, ih), acc(rule(p5h, Xh1, k), yh, ih)
        print(f"     [흔들기 진짜 {len(hp)}] 오탐 {h6:.4f} → {hg:.4f} | [공개] {c6} → {cg}, 원본판정 공개 진짜 여유 {[round(v, 2) for v in mg]} | [h1 {len(h1)}] {a6h:.4f} → {agh:.4f}", flush=True)
        c = [ag > a6 and pw < .0125, fpd <= 3, hg - h6 <= .03, cg == 21 and all(v >= .5 for v in mg), agh >= a6h]; RES[k] = (all(c), ag)
        print(f"  == {k} 판정: R2↑&p {c[0]} · 봉인진짜 +{fpd} {c[1]} · 흔들기 {c[2]} · 공개 {c[3]} · h1 {c[4]} → {'통과' if all(c) else '불통과'}", flush=True)
    ok = [k for k in RES if RES[k][0]]
    pick = None if not ok else ("LR" if "LR" in ok and ("G1" not in ok or RES["LR"][1] > RES["G1"][1]) else "G1")
    print(f"\n== 최종(IM_진짜확정조건_G_v3.md, 탐색적): {'불채택' if pick is None else 'G-' + pick + ' 탐색적 후보'} ==", flush=True)
    print("G3_DONE", flush=True)

if __name__ == "__main__":
    main()
