"""IM69: 남은 오차 ①② 탐색 — 원본(최근접) 지문과의 칸별 차이 '집중도'(상위 10% 칸 몫)로 이웃 컷 진짜 vs 부분 편집 가짜 가르기. 서버(nadan, GPU 0). 탐색·판정 아님.
진짜: R3·R4·R5 진짜, newreal, 봉인 진짜 3종 × 시나리오 none/a/b(im52 방식 제외). 가짜: 나단9가 놓친 R2~R5 가짜 181장(코덱스 진단표), 제외 없음.
문턱은 진짜만으로: gap≥3.474 & dg1≥U(2.3891)인 진짜(none∪a∪b)에서 집중도 99.75% 분위."""
import sys, csv, re, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im36_finefp_eval import finefp_img, load_db, DEV  # noqa
from im43_h_feats import paths_of  # noqa
L = H.L; names, DGB, DE = load_db(); stem = lambda s: Path(str(s).replace("\\", "/")).stem
grp = np.array(["_".join(n.split("_")[:3]) for n in names]); num = np.array([int(re.sub(r"\D", "", n.split("_")[-1]) or 0) for n in names])
gidx = {}
for i, g in enumerate(grp): gidx.setdefault(g, []).append(i)
gord = {g: np.array(sorted(ix, key=lambda i: num[i])) for g, ix in gidx.items()}; idx_of = {n: i for i, n in enumerate(names)}
def excl(src, mode):
    i = idx_of.get(src)
    if i is None or mode == "none": return np.array([], int)
    if mode == "a": return np.array([i])
    g = gord[grp[i]]; pos = int(np.where(g == i)[0][0]); return g[max(0, pos - 3): pos + 4]
def top_share(d, q=.1):
    k = max(1, int(len(d) * q)); s = np.sort(d)[::-1]; return float(s[:k].sum() / (s.sum() + 1e-6))
def feats(paths, srcs, mode):
    out = []
    for k in range(0, len(paths), 64):
        ge = [finefp_img(Image.open(p)) for p in paths[k:k + 64]]; gs = np.stack([a for a, _ in ge]); q = torch.from_numpy(gs).to(DEV); d = torch.cdist(q, DGB, p=1) / q.shape[1]
        for j, s in enumerate(srcs[k:k + 64]):
            ex = excl(s, mode)
            if len(ex): d[j, torch.as_tensor(ex, device=DEV)] = 1e9
        v, ix = torch.topk(d, 2, largest=False); v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for j in range(len(gs)):
            i1 = ix[j, 0]; dg = np.abs(gs[j] - DGB[i1].cpu().numpy()); de = np.abs(ge[j][1] - DE[i1])
            out.append((v[j, 0], v[j, 1] - v[j, 0], top_share(dg), top_share(de), float((dg > 3 * np.median(dg) + 2).mean())))
    return np.array(out)   # dg1, gap, conc_g, conc_e, frac_hot
R = {}
for sn in ("im_set_r3", "im_set_r4", "im_set_r5", "im_set_newreal", "sealed_real", "sealed_brown", "sealed_bright"):
    ps = paths_of(sn); rr = list(csv.DictReader(open((L / sn / "manifest.csv") if sn.startswith("im_set") else (L / "sealed_c3" / sn[7:] / "manifest.csv"), encoding="utf-8")))
    m = np.array([int(r["label"]) == 0 for r in rr]); P = np.array(ps)[m].tolist(); S = [stem(r["source"]) for r, mm in zip(rr, m) if mm]
    for mode in ("none", "a", "b"): R[(sn, mode)] = feats(P, S, mode)
    print(sn, "done", flush=True)
mr = list(csv.DictReader(open(L / "preds/missed181.csv", encoding="utf-8")))
FK = feats([str(L / f"im_set_{r['set']}" / "img" / r["file"]) for r in mr], [""] * len(mr), "none"); kind = np.array([r["kind"] for r in mr])
np.savez(L / "preds/im69_conc.npz", fk=FK, kind=kind, **{f"{a}__{b}": v for (a, b), v in R.items()})
GAP, DGT, U = 3.47395849, 1.09114587, 2.3891
RE = {md: np.concatenate([v for (a, b), v in R.items() if b == md]) for md in ("none", "a", "b")}
allr = np.concatenate(list(RE.values())); zone = lambda X: (X[:, 1] >= GAP) & (X[:, 0] >= U)
print(f"진짜(none∪a∪b) {len(allr)} 중 상한 위 영역 {zone(allr).sum()} | 놓친 가짜 {len(FK)} 중 상한 위 영역 {zone(FK).sum()}", flush=True)
for ci, nm in ((2, "밝기 칸 집중도"), (3, "질감 칸 집중도"), (4, "튀는 칸 비율")):
    rz = allr[zone(allr), ci]; T = float(np.quantile(rz, .9975)) if len(rz) else np.inf
    print(f"[{nm}] 진짜 영역 분위 50/90/99/99.75 {np.round(np.quantile(rz, [.5, .9, .99, .9975]), 3).tolist()} | 가짜 영역 분위 50/10 {np.round(np.quantile(FK[zone(FK), ci], [.5, .1]), 3).tolist()} → T {T:.3f}", flush=True)
    for md, X in RE.items(): print(f"     진짜 [{md}] 경보 {int((zone(X) & (X[:, ci] > T)).sum())}/{len(X)}", flush=True)
    f = zone(FK) & (FK[:, ci] > T); print(f"     놓친 가짜 회수 {int(f.sum())} " + str({k: int(f[kind == k].sum()) for k in sorted(set(kind))}), flush=True)
print("IM69_DONE", flush=True)
