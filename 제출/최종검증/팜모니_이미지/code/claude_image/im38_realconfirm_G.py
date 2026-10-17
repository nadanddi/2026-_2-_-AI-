"""IM38: 후보 G — IM_진짜확정조건_G_v1.md. 서버(nadan, GPU 0). 진짜 확정에 필터 점수 조건 추가(학습 자료 v2·v4로만 적합)."""
import sys, csv, json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im36_finefp_eval import load_db, rows_of, L, B  # noqa
from im37b_f2_eval_v2 import fe_paths, run_pkg, NB  # noqa

FILT = {"filter_jpeg", "filter_noise", "filter_blur"}; REAL = {"real", "real_resampled"}
phi = lambda X: np.c_[X[:, 0], X[:, 2], X[:, 3], X[:, 4], np.abs(X[:, 4])]

def main():
    names, DG, DE = load_db(); T = json.load(open(L / "preds/finefp_thresholds_v2.json")); print("정밀 DB", len(names), flush=True)
    r2 = rows_of("im_set_r2"); z = np.load(L / "preds/sealed_c3v2.npz", allow_pickle=True); st, pp5, sp = z["set"], z["p5"], z["path"]
    bad = {r["group"] for r in r2} | {Path(p).stem.rsplit("_", 1)[0] for p in sp}
    M = lambda X: (X[:, 0] <= T["m"]) & (X[:, 1] >= T["gap"])
    FK = lambda X: M(X) & ((X[:, 2] > T["f_g"]) | (X[:, 3] > T["f_e"]))
    RC = lambda X: M(X) & ~FK(X) & (X[:, 2] <= T["r_g"]) & (X[:, 3] <= T["r_e"]) & (np.abs(X[:, 4]) <= T["r_glob"])   # p 조건 제외
    tr = [(sn, r) for sn in ("im_set_v2", "im_set_v4") for r in rows_of(sn) if r["group"] not in bad and (r["kind"] in FILT or r["kind"] in REAL)]
    Xt = fe_paths([str(L / sn / "img" / r["file"]) for sn, r in tr], DG, DE); yt = np.array([int(r["kind"] in FILT) for _, r in tr])
    use = M(Xt) & ~FK(Xt); print(f"학습 자료 {len(tr)} (묶음 제외 후), 원본판정·비가짜확정 {use.sum()} (진짜 {(use & (yt == 0)).sum()}, 필터 {(use & (yt == 1)).sum()})", flush=True)
    sc = StandardScaler().fit(phi(Xt[use])); lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(phi(Xt[use])), yt[use])
    S = lambda X: lr.decision_function(sc.transform(phi(X)))
    rr = RC(Xt) & (yt == 0); t = float(np.quantile(S(Xt[rr]), .975))
    print(f"t = {t:.3f} | 학습: 진짜확정 가능 진짜 {rr.sum()} 중 유지 {(S(Xt[rr]) < t).mean():.3f}, 진짜확정 조건 통과 필터 {(RC(Xt) & (yt == 1)).sum()} 중 차단 {(S(Xt[RC(Xt) & (yt == 1)]) >= t).mean():.3f}", flush=True)
    json.dump(dict(t=t, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_.ravel().tolist(), b=float(lr.intercept_[0])), open(L / "preds/g_rule_v1.json", "w"))
    def rule(p, X, gate):
        fk = FK(X); rl = RC(X) & (p < .8) & ((S(X) < t) if gate else True)
        q = p.copy(); q[fk] = np.maximum(q[fk], .95); q[rl] = np.minimum(q[rl], .3); return q
    # R2
    yr = np.array([int(r["label"]) for r in r2]); gr = np.array([r["group"] for r in r2]); kd = np.array([r["kind"] for r in r2]); p5 = np.load(L / "preds/_r2_나단5.npy")
    Xr = np.load(L / "preds/f2_eval_v2.npz")["Xr"]; q6 = rule(p5, Xr, False); qg = rule(p5, Xr, True)
    assert np.abs(q6 - np.load(L / "preds/f2_eval_v2.npz")["q6"]).max() < 1e-9
    acc = lambda p, y, ix: .5 * (p[ix][y[ix] == 0] < .5).mean() + .5 * (p[ix][y[ix] == 1] >= .5).mean(); allix = np.arange(len(yr))
    gs_ = np.array(sorted(set(gr))); gi = {g: np.where(gr == g)[0] for g in gs_}; rng = np.random.default_rng(38); bs = []
    for _ in range(NB):
        ix = np.concatenate([gi[g] for g in rng.choice(gs_, len(gs_))])
        if (yr[ix] == 0).any() and (yr[ix] == 1).any(): bs.append(acc(qg, yr, ix) - acc(q6, yr, ix))
    pw = float((np.array(bs) <= 0).mean())
    print(f"[R2] 정확도 나단6 {acc(q6, yr, allix):.4f} → G {acc(qg, yr, allix):.4f}, p_worse {pw:.4f}", flush=True)
    for k in sorted(set(kd)):
        m = kd == k; print(f"     {k:14s} n={m.sum():4d} 정답률 나단5 {((p5[m] >= .5) == (yr[m] == 1)).mean():.3f} 나단6 {((q6[m] >= .5) == (yr[m] == 1)).mean():.3f} G {((qg[m] >= .5) == (yr[m] == 1)).mean():.3f}", flush=True)
    # 봉인
    fpd = 0
    for sn, lb in (("real", 0), ("brown", 0), ("bright", 0), ("qwen", 1), ("v7", 1)):
        m = st == sn; X = fe_paths(list(sp[m]), DG, DE); a6, ag = rule(pp5[m], X, False), rule(pp5[m], X, True)
        if lb == 0: fpd += int((ag >= .5).sum() - (a6 >= .5).sum())
        print(f"[봉인 {sn}] n={m.sum()} 양성 비율 나단6 {(a6 >= .5).mean():.4f} → G {(ag >= .5).mean():.4f}", flush=True)
    # 흔들기 별도 진짜 500
    hp = sorted(str(p) for p in (L / "preds/_jig_hold").glob("j*.jpg")); ph = np.load(L / "preds/_jig_hold_n5.npy"); Xh = fe_paths(hp, DG, DE)
    h6, hg = (rule(ph, Xh, False) >= .5).mean(), (rule(ph, Xh, True) >= .5).mean(); print(f"[흔들기 진짜 {len(hp)}] 오탐 나단6 {h6:.4f} → G {hg:.4f}", flush=True)
    # 공개 21
    PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; lab = list(csv.DictReader(open(PUB / "sample_labels.csv"))); pf = [str(PUB / "images" / r["file"]) for r in lab]
    py = np.array([int(r["label"]) for r in lab]); pp = np.load(L / "preds/_pub_n5.npy"); Xp = fe_paths(pf, DG, DE); c6, cg = int(((rule(pp, Xp, False) >= .5) == py).sum()), int(((rule(pp, Xp, True) >= .5) == py).sum())
    print(f"[공개 21] 나단6 {c6} → G {cg} | 공개 진짜 s: " + ", ".join(f"{r['file'][:4]} {s:.2f}" for r, s, y in zip(lab, S(Xp), py) if y == 0) + f" (t {t:.2f})", flush=True)
    # h1 (묶음 제외)
    h1 = [r for r in rows_of("im_set_h1") if r["group"] not in bad]; yh = np.array([int(r["label"]) for r in h1]); hpth = [str(L / "im_set_h1/img" / r["file"]) for r in h1]
    p5h = run_pkg(B / "submit_img05/팜모니_나단5", hpth, "h1_n5"); Xh1 = fe_paths(hpth, DG, DE); ih = np.arange(len(yh))
    a6h, agh = acc(rule(p5h, Xh1, False), yh, ih), acc(rule(p5h, Xh1, True), yh, ih); print(f"[h1 묶음제외 {len(h1)}] 정확도 나단6 {a6h:.4f} → G {agh:.4f}", flush=True)
    c = [acc(qg, yr, allix) > acc(q6, yr, allix) and pw < .025, fpd <= 3, hg - h6 <= .03, cg == 21, agh >= a6h]
    print(f"\n== 판정(IM_진짜확정조건_G_v1.md, 탐색적) ==\n  R2↑&p {c[0]} · 봉인진짜 +{fpd}장≤3 {c[1]} · 흔들기 {c[2]} · 공개21 {c[3]} · h1 {c[4]} → {'탐색적 후보' if all(c) else '불채택'}", flush=True)
    print("G1_DONE", flush=True)

if __name__ == "__main__":
    main()
