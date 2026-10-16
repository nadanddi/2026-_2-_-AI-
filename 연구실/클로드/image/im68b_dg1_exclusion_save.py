"""IM68b(im68 복사): 시나리오별 dg1·gap 배열 저장(상한 문턱 산정용). IM68: 나단10 후보(dg1 경보: gap≥3.474 & dg1>1.0911 → 가짜) 위험 점검 — im52 복사. 진짜 사진의 원본을 DB에서 빼고 대조했을 때(a 원본만 / b ±3프레임 / c 촬영 묶음 전체) 경보율.
시험 진짜가 DB에 없는 사진일 때의 오탐 위험을 본다. 서버(nadan, GPU 0). 탐색·판정 아님."""
import sys, csv, re, json, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im36_finefp_eval import finefp_img, load_db, DEV  # noqa
from im43_h_feats import paths_of  # noqa
L = H.L; z = np.load(L / "preds/h_model_h2.npz"); M = {}
for k, v in z.items(): a, c = k.split("__"); M.setdefault(a, {})[c] = v
th = json.load(open(L / "preds/h_theta_h2.json"))
names, DG, DE = load_db(); stem = lambda s: Path(str(s).replace("\\", "/")).stem
grp = np.array(["_".join(n.split("_")[:3]) for n in names]); num = np.array([int(re.sub(r"\D", "", n.split("_")[-1]) or 0) for n in names])
gidx = {}
for i, g in enumerate(grp): gidx.setdefault(g, []).append(i)
gord = {g: np.array(sorted(ix, key=lambda i: num[i])) for g, ix in gidx.items()}; idx_of = {n: i for i, n in enumerate(names)}
def excl(src, mode):
    i = idx_of.get(src)
    if i is None: return np.array([], int)
    g = gord[grp[i]]
    if mode == "a": return np.array([i])
    if mode == "c": return g
    pos = int(np.where(g == i)[0][0]); return g[max(0, pos - 3): pos + 4]
def match(paths, srcs, mode):
    out = []
    for k in range(0, len(paths), 64):
        gs = np.stack([finefp_img(Image.open(p))[0] for p in paths[k:k + 64]]); q = torch.from_numpy(gs).to(DEV); d = torch.cdist(q, DG, p=1) / q.shape[1]
        for j, s in enumerate(srcs[k:k + 64]):
            ex = excl(s, mode)
            if len(ex): d[j, torch.as_tensor(ex, device=DEV)] = 1e9
        v, _ = torch.topk(d, 2, largest=False); v = v.cpu().numpy(); out += [(a, b - a) for a, b in v]
    return np.array(out)
GAP, DGT = 3.47395849, 1.09114587
fire = lambda X: (X[:, 1] >= GAP) & (X[:, 0] > DGT)
R = []
for sn in ("im_set_r3", "im_set_r4", "im_set_r5", "im_set_newreal", "sealed_real", "sealed_brown", "sealed_bright"):
    F = H.load(sn); ps = paths_of(sn)
    rr = list(csv.DictReader(open((L / sn / "manifest.csv") if sn.startswith("im_set") else (L / "sealed_c3" / sn[7:] / "manifest.csv"), encoding="utf-8")))
    m = np.array([int(r["label"]) == 0 for r in rr])
    R.append((np.array(ps)[m].tolist(), [stem(r["source"]) for r, mm in zip(rr, m) if mm], sn))
tot = sum(len(x[0]) for x in R); print("진짜", tot, "장", flush=True)
SAVE = {}
for mode in ("none", "a", "b", "c"):
    out = []
    for ps, ss, sn in R:
        X = match(ps, ss if mode != "none" else [""] * len(ss), mode if mode != "none" else "a"); f = fire(X); out.append(f); SAVE[f"{mode}__{sn}"] = X
        print(f"  [{mode}] {sn:16s} n={len(ps):4d} 경보 {int(f.sum()):3d} ({f.mean():.4f}) | dg1 중앙 {np.median(X[:, 0]):.2f} · gap 중앙 {np.median(X[:, 1]):.2f} · gap≥3.47 {np.mean(X[:, 1] >= GAP):.3f}", flush=True)
    a = np.concatenate(out); print(f"[시나리오 {mode}] 진짜 경보 {int(a.sum())}/{tot} = {a.mean():.4f}", flush=True)
np.savez(L / "preds/im68b_excl.npz", **SAVE); print("IM68_DONE", flush=True)
