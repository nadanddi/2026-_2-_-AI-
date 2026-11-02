"""IM36b: 후보 F v2 — IM_정밀원본대조_F_v2.md. 서버(nadan, GPU 0). 국소 특징·처리 흔들기 진짜로 문턱·진짜 확정은 나단5 p<.8일 때만."""
import sys, csv, io, json, random, zlib, subprocess, numpy as np, cv2, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im36_finefp_eval import finefp_img, load_db, rows_of, L, B, DEV  # noqa
NB = 2000; KER = [Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, "AREA"]

def feats(gs, es, DG, DE):
    q = torch.from_numpy(np.stack(gs)).to(DEV); d = torch.cdist(q, DG, p=1) / q.shape[1]; v, ix = torch.topk(d, 2, largest=False); v, ix = v.cpu().numpy(), ix.cpu().numpy(); out = []
    for j in range(len(gs)):
        i1 = ix[j, 0]; dg = np.abs(gs[j] - DG[i1].cpu().numpy()); de = es[j] - DE[i1]; c = np.abs(de - np.median(de))
        out.append(([v[j, 0], v[j, 1] - v[j, 0], np.sort(dg - np.median(dg))[-int(len(dg) * .02):].mean(), np.sort(c)[-int(len(c) * .05):].mean(), np.median(de)], i1))
    return out

def feats_paths(paths, DG, DE):
    out = []
    for k in range(0, len(paths), 64):
        ge = [finefp_img(Image.open(p)) for p in paths[k:k + 64]]; out += feats([a for a, _ in ge], [b for _, b in ge], DG, DE)
    return out

def jiggle(raw, rng):   # 처리 흔들기: 덮게 축소(커널 4종) → 가운데 크롭 → JPEG q70~95
    im = Image.open(raw).convert("RGB"); w, h = im.size; tw, th = (768, 1024) if h >= w else (1024, 768); s = max(tw / w, th / h); nw, nh = max(tw, round(w * s)), max(th, round(h * s))
    k = rng.choice(KER)
    im = Image.fromarray(cv2.resize(np.asarray(im), (nw, nh), interpolation=cv2.INTER_AREA)) if k == "AREA" else im.resize((nw, nh), k)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2; im = im.crop((x0, y0, x0 + tw, y0 + th)); b = io.BytesIO(); im.save(b, "JPEG", quality=rng.randint(70, 95)); return Image.open(io.BytesIO(b.getvalue()))

def run_pkg(pk, paths, tag):
    out = L / f"preds/_{tag}.npy"; lst = L / f"preds/_{tag}.json"
    if out.exists(): return np.load(out)
    lst.write_text(json.dumps(paths))
    code = "import sys,json,numpy as np; sys.path.insert(0, %r); import inference as I; m=I.load_model(%r); np.save(%r, np.array(I.predict(m, json.load(open(%r)))))" % (str(pk), str(pk / "model"), str(out), str(lst))
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True); return np.load(out)

def main():
    names, DG, DE = load_db(); idx_of = {n: i for i, n in enumerate(names)}; print("정밀 DB", len(names), flush=True)
    raw = [r for r in csv.DictReader(open(L / "im_real_raw/manifest.csv", encoding="utf-8"))]; rng = random.Random(20261014); rng.shuffle(raw)
    J = []
    for r in raw[:2000]:
        p = L / "im_real_raw" / r["file"].replace("\\", "/")
        if p.exists(): J.append((p, Path(r["file"].replace("\\", "/")).stem))
    imgs = [jiggle(p, random.Random(zlib.crc32(st.encode()))) for p, st in J]
    FJ = []
    for k in range(0, len(imgs), 64):
        ge = [finefp_img(im) for im in imgs[k:k + 64]]; FJ += feats([a for a, _ in ge], [b for _, b in ge], DG, DE)
    XJ = np.array([f for f, _ in FJ]); own = np.array([names[i] == st for (_, i), (_, st) in zip(FJ, J)])
    cal, hold = np.arange(len(J)) < 1500, np.arange(len(J)) >= 1500; Rc = cal & own
    print(f"흔들기 진짜 {len(J)} (문턱용 {cal.sum()}, 별도 {hold.sum()}), 1위=자기원본 비율 {own.mean():.3f}", flush=True)
    T = dict(m=np.quantile(XJ[Rc, 0], .995), gap=np.quantile(XJ[Rc, 1], .05), f_g=np.quantile(XJ[Rc, 2], .999), f_e=np.quantile(XJ[Rc, 3], .999),
             r_g=np.quantile(XJ[Rc, 2], .95), r_e=np.quantile(XJ[Rc, 3], .95), r_glob=np.quantile(np.abs(XJ[Rc, 4]), .95))
    print("문턱", {k: round(float(v), 3) for k, v in T.items()}, flush=True); json.dump({k: float(v) for k, v in T.items()}, open(L / "preds/finefp_thresholds_v2.json", "w"))
    def apply(p, X):
        m = (X[:, 0] <= T["m"]) & (X[:, 1] >= T["gap"]); fk = m & ((X[:, 2] > T["f_g"]) | (X[:, 3] > T["f_e"]))
        rl = m & ~fk & (X[:, 2] <= T["r_g"]) & (X[:, 3] <= T["r_e"]) & (np.abs(X[:, 4]) <= T["r_glob"]) & (p < .8)
        q = p.copy(); q[fk] = np.maximum(q[fk], .95); q[rl] = np.minimum(q[rl], .3); return q, m, rl, fk
    # 흔들기 별도 500장: 나단5 예측 필요 → 임시 저장 후 패키지 실행
    hd = L / "preds/_jig_hold"; hd.mkdir(exist_ok=True); hp = []
    for k in np.where(hold)[0]:
        f = hd / f"j{k:04d}.jpg"; imgs[k].save(f, quality=95); hp.append(str(f))   # 이미 JPEG 흔들기 된 사진을 고품질로 보존
    P5 = B / "submit_img05/팜모니_나단5"
    ph = run_pkg(P5, hp, "jig_hold_n5"); qh, mh, rh, fh = apply(ph, XJ[hold])
    print(f"[흔들기 별도 진짜 {hold.sum()}] 원본판정 {mh.mean():.3f} | 오탐 나단5 {(ph >= .5).mean():.4f} → F {(qh >= .5).mean():.4f} (가짜확정 {fh.sum()}, 진짜확정 {rh.sum()}) | 이웃프레임(1위≠원본) {(~own[hold]).sum()}장 중 가짜확정 {(fh & ~own[hold]).sum()}", flush=True)
    def evalset(nm, paths, yy, p5):
        Fx = feats_paths(paths, DG, DE); X = np.array([f for f, _ in Fx]); q, m, rl, fk = apply(p5, X)
        print(f"[{nm}] n={len(yy)} 원본판정 {m.mean():.3f} 진짜확정 {rl.mean():.3f} 가짜확정 {fk.mean():.3f} | "
              + (f"오탐 {(p5[yy == 0] >= .5).mean():.4f} → {(q[yy == 0] >= .5).mean():.4f} " if (yy == 0).any() else "")
              + (f"탐지 {(p5[yy == 1] >= .5).mean():.4f} → {(q[yy == 1] >= .5).mean():.4f}" if (yy == 1).any() else ""), flush=True)
        return q, X
    r2 = rows_of("im_set_r2"); yr = np.array([int(r["label"]) for r in r2]); p5 = np.load(L / "preds/_r2_나단5.npy"); gr = np.array([r["group"] for r in r2])
    qr, Xr = evalset("R2 보류", [str(L / "im_set_r2/img" / r["file"]) for r in r2], yr, p5)
    acc = lambda p, ix: .5 * (p[ix][yr[ix] == 0] < .5).mean() + .5 * (p[ix][yr[ix] == 1] >= .5).mean(); allix = np.arange(len(yr))
    gs_ = np.array(sorted(set(gr))); gi = {g: np.where(gr == g)[0] for g in gs_}; rng2 = np.random.default_rng(7); bs = []
    for _ in range(NB):
        ix = np.concatenate([gi[g] for g in rng2.choice(gs_, len(gs_))])
        if (yr[ix] == 0).any() and (yr[ix] == 1).any(): bs.append(acc(qr, ix) - acc(p5, ix))
    pw = float((np.array(bs) <= 0).mean()); print(f"     R2 정확도 나단5 {acc(p5, allix):.4f} → F {acc(qr, allix):.4f}, p_worse {pw:.4f}", flush=True)
    z = np.load(L / "preds/sealed_c3v2.npz", allow_pickle=True); st, pp5, sp = z["set"], z["p5"], z["path"]; R = {}
    for sn, lab in (("real", 0), ("qwen", 1), ("v7", 1), ("brown", 0), ("bright", 0)):
        m = st == sn; R[sn] = (evalset("봉인 " + sn, list(sp[m]), np.full(m.sum(), lab), pp5[m])[0], pp5[m])
    PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; lab = list(csv.DictReader(open(PUB / "sample_labels.csv"))); pf = [str(PUB / "images" / r["file"]) for r in lab]
    py = np.array([int(r["label"]) for r in lab]); pp = run_pkg(P5, pf, "pub_n5"); Fp = feats_paths(pf, DG, DE); Xp = np.array([f for f, _ in Fp]); qp, mp, rp, fp_ = apply(pp, Xp)
    print("[공개 21] 정답 나단5", int(((pp >= .5) == py).sum()), "→ F", int(((qp >= .5) == py).sum()))
    for r, x, a, b, m_, r_, f_ in zip(lab, Xp, pp, qp, mp, rp, fp_):
        print(f"   {r['file'][:4]} {r['type'][:12]:12s} dg1 {x[0]:5.2f} gap {x[1]:5.2f} loc_g {x[2]:5.1f} loc_e {x[3]:5.1f} glob_e {x[4]:6.1f} | 원본판정 {int(m_)} 진짜확정 {int(r_)} 가짜확정 {int(f_)} | p {a:.2f}→{b:.2f}")
    fr = lambda s: (R[s][0] >= .5).mean(); f5 = lambda s: (R[s][1] >= .5).mean()
    det = lambda w: np.r_[(R["qwen"][w] >= .5), (R["v7"][w] >= .5)].mean()
    c = [acc(qr, allix) > acc(p5, allix) and pw < .025, (qh >= .5).mean() <= (ph >= .5).mean() + .005, fr("real") <= f5("real") and fr("bright") <= f5("bright") and fr("brown") <= f5("brown"),
         det(0) >= det(1), int(((qp >= .5) == py).sum()) >= int(((pp >= .5) == py).sum())]
    print(f"\n== 판정(IM_정밀원본대조_F_v2.md, 탐색적) ==\n  R2↑&p {c[0]} · 흔들기진짜 {c[1]} · 봉인진짜/밝음/갈변 {c[2]} · 봉인가짜 {c[3]} · 공개 {c[4]} → {'탐색적 후보' if all(c) else '불채택'}")

if __name__ == "__main__":
    main()
