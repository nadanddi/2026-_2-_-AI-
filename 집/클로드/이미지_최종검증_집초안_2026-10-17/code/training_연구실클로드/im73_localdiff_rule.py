"""IM73: 후보 규칙 'LD'(원본 뚜렷 + 일부 크게 다름) 점검 — 나단9 가짜 확정 규칙의 dg1 범위 확장.
규칙: gap ≥ 3.474 & 1.0111 < dg1 ≤ 15 & lg > T → 가짜(p≥.95).  lg = 칸별 |지문 차| − 중앙값의 상위 2% 평균(나단9 inference와 같은 계산).
T(실행 전 고정) = 진짜 전체(변형·원본 제외·처리·보호)에서 영역(gap·dg1 조건) 안 lg 최댓값. 서버(nadan, GPU 0). 탐색."""
import sys, io, csv, re, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im36_finefp_eval import finefp_img, load_db, DEV  # noqa
from im43_h_feats import paths_of  # noqa
L = H.L; names, DGB, DE = load_db(); stem = lambda s: Path(str(s).replace("\\", "/")).stem
grp = np.array(["_".join(n.split("_")[:3]) for n in names]); num = np.array([int(re.sub(r"\D", "", n.split("_")[-1]) or 0) for n in names]); idx_of = {n: i for i, n in enumerate(names)}
gidx = {}
for i, g in enumerate(grp): gidx.setdefault(g, []).append(i)
gord = {g: np.array(sorted(ix, key=lambda i: num[i])) for g, ix in gidx.items()}
def excl(src, mode):
    i = idx_of.get(src)
    if i is None or mode == "none": return np.array([], int)
    if mode == "a": return np.array([i])
    g = gord[grp[i]]; pos = int(np.where(g == i)[0][0]); return g[max(0, pos - 3): pos + 4]
GAP, LO, HI = 3.4739584922790527, 1.0110872983932495, 15.0
def ffx(imgs, srcs=None, mode="none"):
    out = []
    for k in range(0, len(imgs), 64):
        ge = [finefp_img(im) for im in imgs[k:k + 64]]; gs = np.stack([a for a, _ in ge]); q = torch.from_numpy(gs).to(DEV); d = torch.cdist(q, DGB, p=1) / q.shape[1]
        if srcs is not None:
            for j, s in enumerate(srcs[k:k + 64]):
                ex = excl(s, mode)
                if len(ex): d[j, torch.as_tensor(ex, device=DEV)] = 1e9
        v, ix = torch.topk(d, 2, largest=False); v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for j in range(len(gs)):
            i1 = ix[j, 0]; dg = np.abs(gs[j] - DGB[i1].cpu().numpy()); lg = float(np.sort(dg - np.median(dg))[-int(len(dg) * .02):].mean())
            out.append((v[j, 0], v[j, 1] - v[j, 0], lg))
    return np.array(out)
zone = lambda X: (X[:, 1] >= GAP) & (X[:, 0] > LO) & (X[:, 0] <= HI)
def make(raw, kern=Image.LANCZOS, scale=1.0, sh=0, gamma=1.0, bright=1.0, q=92):
    w, h = raw.size; tw, th = (768, 1024) if h >= w else (1024, 768); s = max(tw / w, th / h) * scale
    nw, nh = max(tw, round(w * s)), max(th, round(h * s)); r = raw.resize((nw, nh), kern)
    x0 = int(np.clip((nw - tw) // 2 + sh, 0, nw - tw)); y0 = int(np.clip((nh - th) // 2 + sh, 0, nh - th)); a = np.asarray(r.crop((x0, y0, x0 + tw, y0 + th)), np.float32)
    if gamma != 1.0: a = 255 * (a / 255) ** gamma
    a = np.clip(a * bright, 0, 255).astype(np.uint8); b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=2); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
REAL = {}
rr5 = [r for r in csv.DictReader(open(L / "im_set_r5/manifest.csv", encoding="utf-8")) if r["label"] == "0"]
raws = [Image.open(L / "im_r5_raw" / r["tar"] / r["source"]).convert("RGB") for r in rr5]
for nm, kw in [("기준", {}), ("이동2", dict(sh=2)), ("이동4", dict(sh=4)), ("이동8", dict(sh=8)), ("이동-2", dict(sh=-2)), ("이동-4", dict(sh=-4)), ("이동-8", dict(sh=-8)), ("확대1.03", dict(scale=1.03)), ("확대1.06", dict(scale=1.06)),
               ("감마0.9", dict(gamma=.9)), ("감마1.1", dict(gamma=1.1)), ("밝기+5%", dict(bright=1.05)), ("밝기-5%", dict(bright=.95)), ("Q70", dict(q=70)), ("Q85", dict(q=85)), ("bicubic", dict(kern=Image.BICUBIC)), ("bilinear", dict(kern=Image.BILINEAR))]:
    REAL["R5원본_" + nm] = ffx([make(im, **kw) for im in raws]); print(nm, flush=True)
for sn in ("im_set_newreal", "im_set_r3", "im_set_r4", "im_set_r5", "sealed_real", "sealed_brown", "sealed_bright", "im_set_realg", "im_set_realn", "jig500", "im_set_r3p_sharp", "im_set_r3p_area", "im_set_r3p_png", "im_set_r3p_webp", "im_set_r3p_c444", "pub21"):
    ps = paths_of(sn)
    if sn == "pub21": y = np.array([int(r["label"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))]); src = [""] * len(ps)
    elif sn == "jig500": y = np.zeros(len(ps), int); src = [""] * len(ps)
    else:
        rr = list(csv.DictReader(open((L / sn / "manifest.csv") if sn.startswith("im_set") else (L / "sealed_c3" / sn[7:] / "manifest.csv"), encoding="utf-8"))); y = np.array([int(r["label"]) for r in rr]); src = [stem(r.get("source", "")) for r in rr]
    m = y == 0; P = [p for p, k in zip(ps, m) if k]; S = [s for s, k in zip(src, m) if k]; ims = [Image.open(p).convert("RGB") for p in P]
    REAL[sn] = ffx(ims)
    if sn in ("im_set_newreal", "im_set_r5", "sealed_real"):
        for md in ("a", "b"): REAL[f"{sn}_제외{md}"] = ffx(ims, S, md)
    print(sn, flush=True)
allz = np.concatenate([X[zone(X), 2] for X in REAL.values()]); T = float(allz.max()) if len(allz) else np.inf
print(f"\n진짜 영역 안 {len(allz)}장 lg 분위 50/99/99.9/최대 {np.round(np.quantile(allz, [.5, .99, .999]), 2).tolist()} / {T:.2f} → T = {T:.2f}", flush=True)
for k, X in REAL.items(): print(f"  [{k:22s}] 영역 안 {int(zone(X).sum()):4d}/{len(X)} · lg 최대 {X[zone(X), 2].max() if zone(X).any() else 0:.2f}", flush=True)
print("\n== 가짜 ==", flush=True)
for sn in ("im_set_gpt1", "im_set_gpt2", "im_set_gpt3", "im_set_gptr", "im_set_r2", "im_set_r3", "im_set_r4", "im_set_r5", "im_set_sd2", "sealed_qwen", "pub21"):
    G = H.load(sn); ff = G["ff"]; ok = np.isfinite(ff).all(1); X = np.c_[ff[:, 0], ff[:, 1], ff[:, 2]]
    if sn == "pub21": y = np.array([int(r["label"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))])
    else: rr = list(csv.DictReader(open((L / sn / "manifest.csv") if sn.startswith("im_set") else (L / "sealed_c3" / sn[7:] / "manifest.csv"), encoding="utf-8"))); y = np.array([int(r["label"]) for r in rr])
    f = ok & zone(np.nan_to_num(X)) & (np.nan_to_num(X[:, 2]) > T)
    print(f"  [{sn:14s}] 가짜 {int((y == 1).sum())} 중 규칙 발동 {int((f & (y == 1)).sum())} | 진짜 발동 {int((f & (y == 0)).sum())}", flush=True)
np.save(L / "preds/im73_T.npy", np.array([T]))
print("IM73_DONE", flush=True)
