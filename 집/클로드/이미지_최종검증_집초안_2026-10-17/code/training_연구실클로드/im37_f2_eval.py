"""IM37: 후보 F2 — IM_원본식별편집_F2_v1.md. 서버(nadan, GPU 0). 원본 식별(gap 큼) + 재압축 이상 차이 → 가짜 확정. 문턱은 진짜에서만."""
import sys, csv, json, random, zlib, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im36_finefp_eval import finefp_img, load_db, rows_of, L, B, DEV  # noqa
from im36b_finefp_eval_v2 import jiggle, run_pkg, NB  # noqa

def feats2(gs, es, DG, DE, own=None):   # own: 제외할 DB 인덱스(없으면 -1)
    q = torch.from_numpy(np.stack(gs)).to(DEV); d = torch.cdist(q, DG, p=1) / q.shape[1]
    if own is not None:
        for j, o in enumerate(own):
            if o >= 0: d[j, o] = 1e9
    v, ix = torch.topk(d, 2, largest=False); v, ix = v.cpu().numpy(), ix.cpu().numpy(); out = []
    for j in range(len(gs)):
        i1 = ix[j, 0]; dg = np.abs(gs[j] - DG[i1].cpu().numpy()); de = es[j] - DE[i1]; c = np.abs(de - np.median(de))
        out.append([v[j, 0], v[j, 1] - v[j, 0], np.sort(dg - np.median(dg))[-int(len(dg) * .02):].mean(), np.sort(c)[-int(len(c) * .05):].mean(), np.median(de)])
    return np.array(out)

def fe_imgs(imgs, DG, DE, own=None):
    out = []
    for k in range(0, len(imgs), 64):
        ge = [finefp_img(im) for im in imgs[k:k + 64]]; out.append(feats2([a for a, _ in ge], [b for _, b in ge], DG, DE, None if own is None else own[k:k + 64]))
    return np.concatenate(out)

def fe_paths(paths, DG, DE):
    out = []
    for k in range(0, len(paths), 64):
        out.append(fe_imgs([Image.open(p) for p in paths[k:k + 64]], DG, DE))
    return np.concatenate(out)

def main():
    names, DG, DE = load_db(); idx_of = {n: i for i, n in enumerate(names)}; print("정밀 DB", len(names), flush=True)
    T = json.load(open(L / "preds/finefp_thresholds_v2.json"))
    stem = lambda s: Path(s.replace("\\", "/")).stem
    r2 = rows_of("im_set_r2"); z = np.load(L / "preds/sealed_c3v2.npz", allow_pickle=True); st, pp5, sp = z["set"], z["p5"], z["path"]
    excl = {stem(r["source"]) for r in r2 if r.get("source")}
    raw = [r for r in csv.DictReader(open(L / "im_real_raw/manifest.csv", encoding="utf-8"))]; rng = random.Random(20261014); rng.shuffle(raw)
    J = []
    for r in raw[2000:]:
        p = L / "im_real_raw" / r["file"].replace("\\", "/"); s_ = stem(r["file"])
        if p.exists() and s_ not in excl and s_ in idx_of: J.append((p, s_))
    nc = 1800; cal, hold = J[:nc], J[nc:]
    print(f"진짜 원천 {len(J)} (문턱용 {len(cal)}×2, 별도 {len(hold)}), R2 원천 제외 {len(excl)}", flush=True)
    def fx(lst, seeds):
        A, O = [], []
        for sd in seeds:
            imgs = [jiggle(p, random.Random(zlib.crc32((s_ + sd).encode()))) for p, s_ in lst]; own = np.array([idx_of[s_] for _, s_ in lst])
            A.append(fe_imgs(imgs, DG, DE)); O.append(fe_imgs(imgs, DG, DE, own))
        return np.concatenate(A), np.concatenate(O)
    CA, CO = fx(cal, ["a", "b"]); HA, HO = fx(hold, ["h"])
    T2 = dict(m2=float(np.quantile(CA[:, 0], .999)), g2=float(np.quantile(CO[:, 1], .999)), e2=float(np.quantile(np.abs(CA[:, 4]), .999)))
    print("F2 문턱", {k: round(v, 3) for k, v in T2.items()}, "| 참고: 자기원본 제외 dg1 중앙", round(float(np.median(CO[:, 0])), 2), flush=True)
    json.dump(T2, open(L / "preds/f2_thresholds_v1.json", "w"))
    def base(p, X):   # 나단6(F v2) 재현
        m = (X[:, 0] <= T["m"]) & (X[:, 1] >= T["gap"]); fk = m & ((X[:, 2] > T["f_g"]) | (X[:, 3] > T["f_e"]))
        rl = m & ~fk & (X[:, 2] <= T["r_g"]) & (X[:, 3] <= T["r_e"]) & (np.abs(X[:, 4]) <= T["r_glob"]) & (p < .8)
        q = p.copy(); q[fk] = np.maximum(q[fk], .95); q[rl] = np.minimum(q[rl], .3); return q
    def fire(X):
        a = (X[:, 0] > T2["m2"]) & (X[:, 1] >= T2["g2"]); b = (X[:, 0] <= T["m"]) & (X[:, 1] >= T["gap"]) & (np.abs(X[:, 4]) > T2["e2"]); return a | b, a, b
    def f2(q, X): f, _, _ = fire(X); q = q.copy(); q[f] = np.maximum(q[f], .95); return q
    hA, hO = fire(HA)[0].mean(), fire(HO)[0].mean()
    print(f"[별도 진짜 {len(hold)}] F2 발동률 자기원본 포함 {hA:.4f} / 제외(이웃 프레임) {hO:.4f}", flush=True)
    yr = np.array([int(r["label"]) for r in r2]); p5 = np.load(L / "preds/_r2_나단5.npy"); gr = np.array([r["group"] for r in r2]); kd = np.array([r["kind"] for r in r2])
    Xr = fe_paths([str(L / "im_set_r2/img" / r["file"]) for r in r2], DG, DE); q6 = base(p5, Xr); q7 = f2(q6, Xr); f_, fa, fb = fire(Xr)
    acc = lambda p, ix: .5 * (p[ix][yr[ix] == 0] < .5).mean() + .5 * (p[ix][yr[ix] == 1] >= .5).mean(); allix = np.arange(len(yr))
    gs_ = np.array(sorted(set(gr))); gi = {g: np.where(gr == g)[0] for g in gs_}; rng2 = np.random.default_rng(15); bs = []
    for _ in range(NB):
        ix = np.concatenate([gi[g] for g in rng2.choice(gs_, len(gs_))])
        if (yr[ix] == 0).any() and (yr[ix] == 1).any(): bs.append(acc(q7, ix) - acc(q6, ix))
    pw = float((np.array(bs) <= 0).mean())
    print(f"[R2 보류] n={len(yr)} 발동 진짜 {f_[yr == 0].mean():.4f} 가짜 {f_[yr == 1].mean():.4f} (a {fa.mean():.3f} b {fb.mean():.3f}) | 오탐 {(q6[yr == 0] >= .5).mean():.4f} → {(q7[yr == 0] >= .5).mean():.4f} 탐지 {(q6[yr == 1] >= .5).mean():.4f} → {(q7[yr == 1] >= .5).mean():.4f}", flush=True)
    print(f"     R2 정확도 나단6 {acc(q6, allix):.4f} → F2 {acc(q7, allix):.4f}, p_worse {pw:.4f}", flush=True)
    for k in sorted(set(kd[yr == 1])):
        m = (kd == k) & (yr == 1); print(f"     {k:16s} n={m.sum():4d} 탐지 {(q6[m] >= .5).mean():.3f} → {(q7[m] >= .5).mean():.3f}", flush=True)
    S = {}
    for sn, lab in (("real", 0), ("qwen", 1), ("v7", 1), ("brown", 0), ("bright", 0)):
        m = st == sn; X = fe_paths(list(sp[m]), DG, DE); a6 = base(pp5[m], X); a7 = f2(a6, X); S[sn] = ((a6 >= .5).mean(), (a7 >= .5).mean())
        print(f"[봉인 {sn}] n={m.sum()} 발동 {fire(X)[0].mean():.4f} | 양성 비율 {S[sn][0]:.4f} → {S[sn][1]:.4f}", flush=True)
    PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; lab = list(csv.DictReader(open(PUB / "sample_labels.csv"))); pf = [str(PUB / "images" / r["file"]) for r in lab]
    py = np.array([int(r["label"]) for r in lab]); pp = np.load(L / "preds/_pub_n5.npy"); Xp = fe_paths(pf, DG, DE); qp6 = base(pp, Xp); qp7 = f2(qp6, Xp); fp_ = fire(Xp)[0]
    print("[공개 21] 정답 나단6", int(((qp6 >= .5) == py).sum()), "→ F2", int(((qp7 >= .5) == py).sum()), "| 발동", [r["file"][:4] + ":" + r["type"] for r, f in zip(lab, fp_) if f], flush=True)
    c = [acc(q7, allix) > acc(q6, allix) and pw < .025, hA <= .005 and hO <= .005, all(S[s][1] <= S[s][0] for s in ("real", "brown", "bright")), int(((qp7 >= .5) == py).sum()) == 21]
    print(f"\n== 판정(IM_원본식별편집_F2_v1.md, 탐색적) ==\n  R2↑&p {c[0]} · 진짜발동≤.5% {c[1]} · 봉인진짜 {c[2]} · 공개21 {c[3]} → {'탐색적 후보' if all(c) else '불채택'}", flush=True)
    np.savez(L / "preds/f2_eval_v1.npz", Xr=Xr, q6=q6, q7=q7, CA=CA, CO=CO, HA=HA, HO=HO)

if __name__ == "__main__":
    main()
