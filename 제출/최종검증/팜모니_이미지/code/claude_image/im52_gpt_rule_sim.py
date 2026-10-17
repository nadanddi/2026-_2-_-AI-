"""IM52: 후보 I2(IM_GPT경계규칙_I2_v1.md) — 원본·이웃 프레임 제외 대조로 진짜 발동률 시뮬레이션. 서버(nadan, GPU 0)."""
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
fire = lambda dg1, p: (dg1 > 5) & (dg1 < 20) & (p >= .35) & (p < .5)
R = []
for sn in ("im_set_r3", "sealed_real", "sealed_brown", "sealed_bright"):
    F = H.load(sn); ps = paths_of(sn)
    rr = list(csv.DictReader(open((L / sn / "manifest.csv") if sn.startswith("im_set") else (L / "sealed_c3" / sn[7:] / "manifest.csv"), encoding="utf-8")))
    m = np.array([int(r["label"]) == 0 for r in rr]); pn = H.apply(F["ppre"], H.score(M, F), th["theta"])   # 정밀 대조 끈 나단8 p
    R.append((np.array(ps)[m].tolist(), [stem(r["source"]) for r, mm in zip(rr, m) if mm], pn[m], sn))
tot = sum(len(x[0]) for x in R); print("진짜", tot, "장", flush=True)
for mode in ("a", "b", "c"):
    f_all, dg = [], []
    for ps, ss, pn, sn in R:
        X = match(ps, ss, mode); f_all.append(fire(X[:, 0], pn)); dg.append(X[:, 0])
    f_all = np.concatenate(f_all); dg = np.concatenate(dg)
    print(f"[시나리오 {mode}] 진짜 발동 {f_all.sum()}/{tot} = {f_all.mean():.4f} | dg1 사분위 {np.percentile(dg, [10, 25, 50, 75, 90]).round(2).tolist()} | 5<dg1<20 비율 {((dg > 5) & (dg < 20)).mean():.3f}", flush=True)
F = H.load("im_set_gpt3"); rr = list(csv.DictReader(open(L / "im_set_gpt3/manifest.csv", encoding="utf-8"))); k = np.array([r["kind"] for r in rr])
p8 = H.apply(F["p6"], H.score(M, F), th["theta"], th["theta_rc"], F["rc"]); f = fire(F["ff"][:, 0], p8)
for kk in ("gpt3_s1", "gpt3_pgen", "gpt3_pstyle"):
    m = k == kk; print(f"[gpt3 {kk}] n={m.sum()} 나단8 탐지 {(p8[m] >= .5).mean():.3f} → 규칙 후 {((p8[m] >= .5) | f[m]).mean():.3f} (고침 {int((f[m] & (p8[m] < .5)).sum())}장, 표본 안 — 과적합)", flush=True)
print("I2_DONE")
