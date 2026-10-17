"""IM70: 나단10 근거 점검 2·3·4(IM_나단10_점검234_v1.md, 실행 전 고정). 서버(nadan, GPU 0).
2) 공개 진짜 dg1 여유  3) R5 진짜 원본에 처리 변형 → band 경보율  4) 공개 가짜 band 경보·나단9 정답 여부."""
import sys, io, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im66_meta_v2 as M  # noqa (나단9 전문가·θ·형식 게이트)
from im36_finefp_eval import finefp_img, load_db, DEV  # noqa
H = M.H; L = H.L; names, DGB, DE = load_db()
GAP, DGT, U, TC = 3.47395849, 1.09114587, 2.3891, .55
def top_share(d, q=.1):
    k = max(1, int(len(d) * q)); s = np.sort(d)[::-1]; return float(s[:k].sum() / (s.sum() + 1e-6))
def ffq(imgs):
    out = []
    for k in range(0, len(imgs), 64):
        gs = np.stack([finefp_img(im)[0] for im in imgs[k:k + 64]]); q = torch.from_numpy(gs).to(DEV); d = torch.cdist(q, DGB, p=1) / q.shape[1]
        v, ix = torch.topk(d, 2, largest=False); v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for j in range(len(gs)): out.append((v[j, 0], v[j, 1] - v[j, 0], top_share(np.abs(gs[j] - DGB[ix[j, 0]].cpu().numpy()))))
    return np.array(out)
band = lambda X: (X[:, 1] >= GAP) & (X[:, 0] > DGT) & (X[:, 0] < U)
r10b = lambda X: band(X) | ((X[:, 1] >= GAP) & (X[:, 0] >= U) & (X[:, 2] > TC))
guard = lambda X: band(X) & (X[:, 2] > TC)
# 2·4) 공개 21
P = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; pr = list(csv.DictReader(open(P / "sample_labels.csv")))
G = M.load_g("pub21"); p9 = np.where(G["std"], M.p9_of(G, M.score_with(M.M9, G)) if hasattr(M, "p9_of") else H.apply(G["p6"], M.score_with(M.M9, G), M.T9["theta"], M.T9["theta_rc"], G["rc"]), G["p6"])
X = ffq([Image.open(P / "images" / r["file"]).convert("RGB") for r in pr]); y = np.array([int(r["label"]) for r in pr])
print("== 점검 2: 공개 진짜 ==", flush=True)
for i in np.where(y == 0)[0]: print(f"  {pr[i]['file']}: dg1 {X[i, 0]:.3f} gap {X[i, 1]:.2f} 집중도 {X[i, 2]:.3f} | 원본 잡힘 {X[i, 1] >= GAP} | band {bool(band(X[i:i + 1])[0])} | 나단9 p {p9[i]:.3f}", flush=True)
m = (y == 0) & (X[:, 1] >= GAP); c2 = bool((X[m, 0] <= .8).all()); print(f"  [판정 2] 원본 잡힌 공개 진짜 {m.sum()}장 모두 dg1 ≤ 0.8: {c2}", flush=True)
print("== 점검 4: 공개 가짜 ==", flush=True)
for i in np.where(y == 1)[0]: print(f"  {pr[i]['file']}: dg1 {X[i, 0]:.2f} gap {X[i, 1]:.2f} 집중도 {X[i, 2]:.3f} | band {bool(band(X[i:i + 1])[0])} · 10b {bool(r10b(X[i:i + 1])[0])} | 나단9 p {p9[i]:.3f}", flush=True)
fk = y == 1; print(f"  band 경보 가짜 {int(band(X)[fk].sum())}장 (그중 나단9가 이미 맞힘 {int((band(X) & fk & (p9 >= .5)).sum())}) · 새로 고침 {int((band(X) & fk & (p9 < .5)).sum())} | 10b 새로 고침 {int((r10b(X) & fk & (p9 < .5)).sum())}", flush=True)
# 3) 처리 내성
rr = [r for r in M.man("im_set_r5") if r["label"] == "0"]; RAW = L / "im_r5_raw"
def make(raw, kern=Image.LANCZOS, scale=1.0, sh=0, gamma=1.0, bright=1.0, q=92):
    w, h = raw.size; tw, th = (768, 1024) if h >= w else (1024, 768); s = max(tw / w, th / h) * scale
    nw, nh = max(tw, round(w * s)), max(th, round(h * s)); r = raw.resize((nw, nh), kern)
    x0 = int(np.clip((nw - tw) // 2 + sh, 0, nw - tw)); y0 = int(np.clip((nh - th) // 2 + sh, 0, nh - th)); a = np.asarray(r.crop((x0, y0, x0 + tw, y0 + th)), np.float32)
    if gamma != 1.0: a = 255 * (a / 255) ** gamma
    a = np.clip(a * bright, 0, 255).astype(np.uint8); b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=2); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
VAR = [("기준", {}), ("이동2", dict(sh=2)), ("이동4", dict(sh=4)), ("이동8", dict(sh=8)), ("이동-4", dict(sh=-4)), ("확대1.03", dict(scale=1.03)), ("확대1.06", dict(scale=1.06)),
       ("감마0.9", dict(gamma=.9)), ("감마1.1", dict(gamma=1.1)), ("밝기+5%", dict(bright=1.05)), ("밝기-5%", dict(bright=.95)), ("Q70", dict(q=70)), ("Q85", dict(q=85)),
       ("bicubic", dict(kern=Image.BICUBIC)), ("bilinear", dict(kern=Image.BILINEAR))]
raws = [Image.open(RAW / r["tar"] / r["source"]).convert("RGB") for r in rr]; print(f"== 점검 3: R5 진짜 원본 {len(raws)}장 ==", flush=True)
c3 = True; SAVE = {}
for nm, kw in VAR:
    X = ffq([make(im, **kw) for im in raws]); SAVE[nm] = X; b_ = band(X)
    c3 &= bool(b_.mean() <= .01)
    print(f"  [{nm:9s}] band 경보 {int(b_.sum()):3d} ({b_.mean():.4f}) · 가드 {int(guard(X).sum())} · 10b {int(r10b(X).sum())} | dg1 중앙 {np.median(X[:, 0]):.3f} q90 {np.quantile(X[:, 0], .9):.3f} | gap≥3.47 {np.mean(X[:, 1] >= GAP):.3f}", flush=True)
np.savez(L / "preds/im70_proc.npz", **SAVE)
print(f"\n== 판정 == 점검 2 {c2} · 점검 3(모든 변형 ≤1%) {c3}", flush=True)
print("IM70_DONE", flush=True)
