"""IM37b: 후보 F2 v2 — IM_원본식별편집_F2_v2.md. 서버(nadan, GPU 0). 문턱 = 진짜 최댓값×1.2, 강한 흔들기 5종, 나단6 재현 assert."""
import sys, io, csv, json, random, zlib, numpy as np, cv2, torch
from pathlib import Path
from PIL import Image, ImageFilter
from scipy.stats import beta
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im36_finefp_eval import finefp_img, load_db, rows_of, L, B, DEV  # noqa
from im36b_finefp_eval_v2 import jiggle, run_pkg, NB, KER  # noqa

def jig2(raw, rng, kind):   # 강한 흔들기: 전처리 → 표준 흔들기와 같은 덮게 축소·가운데 크롭·JPEG
    im = Image.open(raw).convert("RGB"); w, h = im.size; q = rng.randint(70, 95)
    if kind == "crop":
        f = rng.uniform(.90, .95); cw, ch = int(w * f), int(h * f); x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch); im = im.crop((x0, y0, x0 + cw, y0 + ch))
    elif kind == "aspect":
        f = rng.uniform(.95, 1.05); im = im.resize((int(w * f), h), Image.BICUBIC)
    elif kind == "gamma":
        g = rng.uniform(.8, 1.25); a = np.asarray(im, np.float32) / 255; im = Image.fromarray(np.clip(255 * a ** g, 0, 255).astype(np.uint8))
    elif kind == "sharp":
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=rng.randint(60, 150), threshold=2))
    elif kind == "lowq":
        q = rng.randint(40, 70)
    w, h = im.size; tw, th = (768, 1024) if h >= w else (1024, 768); s = max(tw / w, th / h); nw, nh = max(tw, round(w * s)), max(th, round(h * s)); k = rng.choice(KER)
    im = Image.fromarray(cv2.resize(np.asarray(im), (nw, nh), interpolation=cv2.INTER_AREA)) if k == "AREA" else im.resize((nw, nh), k)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2; im = im.crop((x0, y0, x0 + tw, y0 + th)); b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue()))

def feats2(gs, es, DG, DE, own=None):
    q = torch.from_numpy(np.stack(gs)).to(DEV); d = torch.cdist(q, DG, p=1) / q.shape[1]
    if own is not None:   # 자기 원본 + 그 사본(DB 거리 ≤ .5) 제외
        so = torch.cdist(DG[torch.as_tensor(own, device=DEV)], DG, p=1) / q.shape[1]; d[so <= .5] = 1e9
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
    return np.concatenate([fe_imgs([Image.open(p) for p in paths[k:k + 64]], DG, DE) for k in range(0, len(paths), 64)])

cp_hi = lambda k, n: float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0

def main():
    names, DG, DE = load_db(); idx_of = {n: i for i, n in enumerate(names)}; assert len(idx_of) == len(names); print("정밀 DB", len(names), flush=True)
    T = json.load(open(L / "preds/finefp_thresholds_v2.json")); stem = lambda s: Path(s.replace("\\", "/")).stem
    r2 = rows_of("im_set_r2"); z = np.load(L / "preds/sealed_c3v2.npz", allow_pickle=True); st, pp5, sp = z["set"], z["p5"], z["path"]
    excl = {stem(r["source"]) for r in r2 if r.get("source")} | {stem(p) for p in sp}
    raw = [r for r in csv.DictReader(open(L / "im_real_raw/manifest.csv", encoding="utf-8"))]; rng = random.Random(20261014); rng.shuffle(raw)
    J = [(L / "im_real_raw" / r["file"].replace("\\", "/"), stem(r["file"])) for r in raw[2000:]]
    J = [(p, s_) for p, s_ in J if p.exists() and s_ not in excl and s_ in idx_of]; cal, hold = J[:1200], J[1200:]   # R2·봉인 원천 제외 후 1,693장 → 1200/493 (실행 중 확인, 결과 전 수정)
    print(f"진짜 원천 {len(J)} (문턱용 {len(cal)}×2, 별도 {len(hold)}), 제외 목록 {len(excl)}", flush=True)
    own_of = lambda lst: np.array([idx_of[s_] for _, s_ in lst])
    def fx(lst, seeds, kind=None):
        A, O = [], []
        for sd in seeds:
            imgs = [(jiggle(p, random.Random(zlib.crc32((s_ + sd).encode()))) if kind is None else jig2(p, random.Random(zlib.crc32((s_ + sd + kind).encode())), kind)) for p, s_ in lst]
            A.append(fe_imgs(imgs, DG, DE)); O.append(fe_imgs(imgs, DG, DE, own_of(lst)))
        return np.concatenate(A), np.concatenate(O)
    CA, CO = fx(cal, ["a", "b"])
    T2 = dict(m2=float(CA[:, 0].max() * 1.2), g2=float(CO[:, 1].max() * 1.2), e2=float(np.abs(CA[:, 4]).max() * 1.2))
    print("F2 문턱", {k: round(v, 3) for k, v in T2.items()}, "| 참고 분위 dg1 .999", round(float(np.quantile(CA[:, 0], .999)), 3), "gap(제외) .999", round(float(np.quantile(CO[:, 1], .999)), 3), "|glob| .999", round(float(np.quantile(np.abs(CA[:, 4]), .999)), 2), flush=True)
    json.dump(T2, open(L / "preds/f2_thresholds_v2.json", "w"))
    def base(p, X):   # 나단6(F v2) 재현
        m = (X[:, 0] <= T["m"]) & (X[:, 1] >= T["gap"]); fk = m & ((X[:, 2] > T["f_g"]) | (X[:, 3] > T["f_e"]))
        rl = m & ~fk & (X[:, 2] <= T["r_g"]) & (X[:, 3] <= T["r_e"]) & (np.abs(X[:, 4]) <= T["r_glob"]) & (p < .8)
        q = p.copy(); q[fk] = np.maximum(q[fk], .95); q[rl] = np.minimum(q[rl], .3); return q
    def fire(X, use_a=True):
        a = (X[:, 0] > T2["m2"]) & (X[:, 1] >= T2["g2"]); b = (X[:, 0] <= T["m"]) & (X[:, 1] >= T["gap"]) & (np.abs(X[:, 4]) > T2["e2"]); return (a | b) if use_a else b, a, b
    def f2(q, X, use_a=True): f = fire(X, use_a)[0]; q = q.copy(); q[f] = np.maximum(q[f], .95); return q
    # 나단6 재현 확인
    P6 = B / "submit_img06/팜모니_나단6"
    PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; lab = list(csv.DictReader(open(PUB / "sample_labels.csv"))); pf = [str(PUB / "images" / r["file"]) for r in lab]
    py = np.array([int(r["label"]) for r in lab]); pp = np.load(L / "preds/_pub_n5.npy"); Xp = fe_paths(pf, DG, DE); qp6 = base(pp, Xp)
    r2p = [str(L / "im_set_r2/img" / r["file"]) for r in r2]; p5 = np.load(L / "preds/_r2_나단5.npy"); Xr = fe_paths(r2p, DG, DE); q6 = base(p5, Xr)
    n6p = run_pkg(P6, pf, "pub_n6"); n6r = run_pkg(P6, r2p[:256], "r2h_n6")
    dmax = max(np.abs(n6p - qp6).max(), np.abs(n6r - q6[:256]).max()); print(f"나단6 재현 최대 차 {dmax:.2e}", flush=True); assert dmax < 1e-4
    # 별도 진짜 발동률
    HA, HO = fx(hold, ["h"]); R = {}
    for ua in (True, False):
        k = int(fire(HA, ua)[0].sum() + fire(HO, ua)[0].sum()); n = len(HA) + len(HO); R[("std", ua)] = cp_hi(k, n)
        print(f"[별도 진짜 표준, {'F2a+b' if ua else 'F2b만'}] 발동 {k}/{n} (포함 {fire(HA, ua)[0].mean():.4f} 제외 {fire(HO, ua)[0].mean():.4f}) CP상한 {R[('std', ua)]:.4f}", flush=True)
    for kind in ("crop", "lowq", "gamma", "sharp", "aspect"):
        SA, SO = fx(hold, ["s"], kind)
        for ua in (True, False):
            R[(kind, ua)] = float(np.r_[fire(SA, ua)[0], fire(SO, ua)[0]].mean())
        print(f"[강한 흔들기 {kind:6s}] 발동 F2a+b {R[(kind, True)]:.4f} / F2b만 {R[(kind, False)]:.4f} | dg1 중앙 {np.median(SA[:, 0]):.2f} gap 중앙 {np.median(SA[:, 1]):.2f}", flush=True)
    use_a = all(R[(k, True)] <= .02 for k in ("crop", "lowq", "gamma", "sharp", "aspect")); print("F2a 유지" if use_a else "강한 흔들기 기준 미달 → F2b만 판정(사전 결정)", flush=True)
    yr = np.array([int(r["label"]) for r in r2]); gr = np.array([r["group"] for r in r2]); kd = np.array([r["kind"] for r in r2]); q7 = f2(q6, Xr, use_a); f_ = fire(Xr, use_a)[0]
    acc = lambda p, ix: .5 * (p[ix][yr[ix] == 0] < .5).mean() + .5 * (p[ix][yr[ix] == 1] >= .5).mean(); allix = np.arange(len(yr))
    gs_ = np.array(sorted(set(gr))); gi = {g: np.where(gr == g)[0] for g in gs_}; rng2 = np.random.default_rng(15); bs = []
    for _ in range(NB):
        ix = np.concatenate([gi[g] for g in rng2.choice(gs_, len(gs_))])
        if (yr[ix] == 0).any() and (yr[ix] == 1).any(): bs.append(acc(q7, ix) - acc(q6, ix))
    pw = float((np.array(bs) <= 0).mean())
    print(f"[R2 보류] n={len(yr)} 발동 진짜 {f_[yr == 0].mean():.4f} 가짜 {f_[yr == 1].mean():.4f} | 오탐 {(q6[yr == 0] >= .5).mean():.4f} → {(q7[yr == 0] >= .5).mean():.4f} 탐지 {(q6[yr == 1] >= .5).mean():.4f} → {(q7[yr == 1] >= .5).mean():.4f}", flush=True)
    print(f"     R2 정확도 나단6 {acc(q6, allix):.4f} → F2 {acc(q7, allix):.4f}, p_worse {pw:.4f}", flush=True)
    for k in sorted(set(kd[yr == 1])):
        m = (kd == k) & (yr == 1); print(f"     {k:16s} n={m.sum():4d} 탐지 {(q6[m] >= .5).mean():.3f} → {(q7[m] >= .5).mean():.3f}", flush=True)
    S = {}
    for sn, lb in (("real", 0), ("qwen", 1), ("v7", 1), ("brown", 0), ("bright", 0)):
        m = st == sn; X = fe_paths(list(sp[m]), DG, DE); a6 = base(pp5[m], X); a7 = f2(a6, X, use_a); S[sn] = ((a6 >= .5).mean(), (a7 >= .5).mean())
        print(f"[봉인 {sn}] n={m.sum()} 발동 {fire(X, use_a)[0].mean():.4f} | 양성 비율 {S[sn][0]:.4f} → {S[sn][1]:.4f}", flush=True)
    qp7 = f2(qp6, Xp, use_a); fp_ = fire(Xp, use_a)[0]
    print("[공개 21] 정답 나단6", int(((qp6 >= .5) == py).sum()), "→ F2", int(((qp7 >= .5) == py).sum()), "| 발동", [r["file"][:4] + ":" + r["type"] for r, f in zip(lab, fp_) if f], flush=True)
    for r, x in zip(lab, Xp):
        if r["label"] == "0": print(f"   공개 진짜 {r['file'][:4]} dg1 {x[0]:.2f} (m2 {T2['m2']:.2f}) gap {x[1]:.2f} (g2 {T2['g2']:.2f}) |glob| {abs(x[4]):.1f} (e2 {T2['e2']:.1f})", flush=True)
    c = [acc(q7, allix) > acc(q6, allix) and pw < .025, R[("std", use_a)] <= .005, all(S[s][1] <= S[s][0] for s in ("real", "brown", "bright")), int(((qp7 >= .5) == py).sum()) == 21]
    print(f"\n== 판정(IM_원본식별편집_F2_v2.md, 탐색적, {'F2a+b' if use_a else 'F2b만'}) ==\n  R2↑&p {c[0]} · 표준진짜 CP≤.5% {c[1]} · 봉인진짜 {c[2]} · 공개21 {c[3]} → {'탐색적 후보' if all(c) else '불채택'}", flush=True)
    np.savez(L / "preds/f2_eval_v2.npz", Xr=Xr, q6=q6, q7=q7, CA=CA, CO=CO, HA=HA, HO=HO, Xp=Xp)
    print("F2V2_DONE", flush=True)

if __name__ == "__main__":
    main()
