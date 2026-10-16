"""IM71: VAE 재구성 오차(AEROBLADE 계열) 탐색 — 기준 IM_VAE경보_V_v1.md(실행 전 고정). 서버(nadan, GPU 0).
사진을 384×512로 줄여 VAE 인코드→디코드, 재구성 오차(L1)가 낮을수록 '확산 모델을 거친 사진'으로 본다. VAE: SD1.5 vae, TAESD(용량 작음).
문턱 = 보정 진짜(newreal + R5 진짜)의 오차 0.25% 분위(낮은 쪽). 경보 → p = max(p9, .6)."""
import sys, io, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im66_meta_v2 as M  # noqa
from im43_h_feats import paths_of  # noqa
H = M.H; L = H.L; DEV = "cuda"
from diffusers import AutoencoderKL, AutoencoderTiny
V = {"sd15": AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16).to(DEV),
     "taesd": AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(DEV)}
@torch.no_grad()
def err(paths):
    out = {k: [] for k in V}
    for i in range(0, len(paths), 16):
        ims = []
        for p in paths[i:i + 16]:
            im = Image.open(p).convert("RGB"); im = im.resize((384, 512) if im.height >= im.width else (512, 384), Image.BICUBIC)
            a = np.asarray(im, np.float32) / 127.5 - 1; ims.append(a if im.height >= im.width else a.transpose(1, 0, 2))
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(DEV, torch.float16)
        for k, v in V.items():
            r = v.decode(v.encode(x).latents if k == "taesd" else v.encode(x).latent_dist.mean).sample.clamp(-1, 1)
            out[k] += (r - x).abs().float().mean((1, 2, 3)).cpu().tolist()
    return {k: np.array(v) for k, v in out.items()}
def p9(sn):
    G = M.load_g(sn); sc = M.score_with(M.M9, G); return np.where(G["std"], H.apply(G["p6"], sc, M.T9["theta"], M.T9["theta_rc"], G["rc"]), G["p6"])
def lab(sn):
    if sn == "pub21": return np.array([int(r["label"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))]), None
    if sn == "jig500": return np.zeros(500, int), None
    rr = list(csv.DictReader(open(L / "sealed_c3" / sn[7:] / "manifest.csv", encoding="utf-8"))) if sn.startswith("sealed_") else M.man(sn)
    return np.array([int(r["label"]) for r in rr]), np.array([r["kind"] for r in rr])
SETS = ["im_set_newreal", "im_set_r5", "sealed_real", "sealed_brown", "sealed_bright", "jig500", "pub21", "im_set_gptr", "im_set_gpt3", "im_set_gpt1", "im_set_sd2", "sealed_qwen"]
E, P, Y, K = {}, {}, {}, {}
for sn in SETS:
    E[sn] = err(paths_of(sn)); P[sn] = p9(sn); Y[sn], K[sn] = lab(sn); print(sn, "done", len(P[sn]), flush=True)
np.savez(L / "preds/im71_vae.npz", **{f"{sn}__{k}": v for sn in SETS for k, v in E[sn].items()})
for k in V:
    cal = np.concatenate([E["im_set_newreal"][k], E["im_set_r5"][k][Y["im_set_r5"] == 0]]); T = float(np.quantile(cal, .0025))
    print(f"\n== {k} == 문턱 {T:.5f} (보정 진짜 {len(cal)}장 0.25% 분위) | 진짜 오차 중앙 {np.median(cal):.4f}", flush=True)
    newfix = 0
    for sn in SETS:
        e, y, p = E[sn][k], Y[sn], P[sn]; a = e < T
        s = f"  [{sn:14s}] 진짜 경보 {int((a & (y == 0)).sum())}/{int((y == 0).sum())}" if (y == 0).any() else f"  [{sn:14s}]"
        if (y == 1).any():
            nf = int((a & (y == 1) & (p < .5)).sum()); s += f" | 가짜 경보 {int((a & (y == 1)).sum())}/{int((y == 1).sum())} · 나단9 놓친 {int(((y == 1) & (p < .5)).sum())} 중 새로 잡음 {nf} | 가짜 오차 중앙 {np.median(e[y == 1]):.4f}"
            if sn not in ("pub21", "im_set_r5"): newfix += nf
        print(s, flush=True)
    pub_ok = int((((P["pub21"] >= .5) | (E["pub21"][k] < T)) == (Y["pub21"] == 1)).sum())
    fp_ok = all(((E[s][k] < T) & (Y[s] == 0)).mean() <= .005 for s in ("sealed_real", "sealed_brown", "sealed_bright", "jig500"))
    print(f"  [판정 {k}] (a) 보호 진짜 각 ≤0.5% {fp_ok} · (b) 외부 가짜 새로 잡음 {newfix} ≥ 5 {newfix >= 5} · (c) 공개 {pub_ok}/21 {pub_ok == 21}", flush=True)
print("IM71_DONE", flush=True)
