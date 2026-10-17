"""IM36: 후보 F 정밀 원본 대조 — IM_정밀원본대조_F_v1.md. 서버(nadan, GPU 0).
입력: local/finefp/*.npz (Colab colab_정밀지문.py 결과), 각 세트 배포 이미지, 나단5 예측(R2: preds/_r2_나단5.npy, 봉인: preds/sealed_c3v2.npz, 공개: 새로 계산)
1) 정밀 지문 DB 적재(g 24×32, e 12×16)  2) 시험 사진 정밀 지문 → g L1 최소 1·2위 후보 → 특징(dg1, gap, dg_max, de_mean, de_max)
3) 문턱: 학습 진짜(원본 DB에 있음)·학습 가짜(원본 DB에 있음) 분포에서 사전 절차로  4) 나단5 출력에 규칙 덧씌워 판정
"""
import sys, csv, json, subprocess, numpy as np, cv2, torch
from pathlib import Path
from PIL import Image
B = Path.home() / "storage/nadan/farmai_img"; L = B / "data/local"; NB = 2000
DEV = torch.device("cuda")

def finefp_img(img):   # 시험 사진용(colab_정밀지문.finefp와 같은 계산; 이미 배포 크기)
    im = img.convert("RGB"); w, h = im.size; port = h >= w; tw, th = (768, 1024) if port else (1024, 768); s = max(tw / w, th / h)
    nw, nh = max(tw, round(w * s)), max(th, round(h * s))
    if (nw, nh) != (w, h): im = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2; a = np.asarray(im.crop((x0, y0, x0 + tw, y0 + th)).convert("L"), np.float32)
    if not port: a = np.ascontiguousarray(np.rot90(a))
    g = cv2.resize(a, (24, 32), interpolation=cv2.INTER_AREA); lap = cv2.Laplacian(a, cv2.CV_32F, ksize=3); v = lap.reshape(16, 64, 12, 64).var(axis=(1, 3))
    e = np.clip(np.round((np.log1p(v) / 10.0) * 255), 0, 255)
    return np.clip(np.round(g), 0, 255).astype(np.float32).ravel(), e.astype(np.float32).ravel()

def load_db():
    N, G, E = [], [], []
    for p in sorted((L / "finefp").glob("*.npz")):
        z = np.load(p); N += list(z["names"]); G.append(z["g"].reshape(len(z["g"]), -1)); E.append(z["e"].reshape(len(z["e"]), -1))
    return np.array(N), torch.from_numpy(np.concatenate(G).astype(np.float32)).to(DEV), np.concatenate(E).astype(np.float32)

def features(paths, names, DG, DE):
    out = []
    for k in range(0, len(paths), 64):
        gs, es = zip(*[finefp_img(Image.open(p)) for p in paths[k:k + 64]])
        q = torch.from_numpy(np.stack(gs)).to(DEV); d = torch.cdist(q, DG, p=1) / q.shape[1]; v, ix = torch.topk(d, 2, largest=False); v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for j in range(len(gs)):
            i1 = ix[j, 0]; dg = np.abs(gs[j] - DG[i1].cpu().numpy()); de = np.abs(es[j] - DE[i1])
            out.append([v[j, 0], v[j, 1] - v[j, 0], np.sort(dg)[-int(len(dg) * .02):].mean(), de.mean(), np.sort(de)[-int(len(de) * .05):].mean(), names[i1]])
    return out

def rows_of(sn, kinds=None):
    return [r for r in csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")) if kinds is None or r["kind"] in kinds]

def main():
    names, DG, DE = load_db(); print("정밀 DB", len(names), flush=True)
    stem = lambda s: Path(s.replace("\\", "/")).stem
    # 문턱 정하기용(학습 자료, 원본이 DB에 있는 사진)
    cal = []
    for sn in ("im_set_v4", "im_set_v5", "im_set_sd2", "im_set_gpt1", "im_set_gpt2"):
        for r in rows_of(sn):
            if r.get("source"): cal.append((str(L / sn / "img" / r["file"]), int(r["label"]), stem(r["source"]), r["kind"]))
    rng = np.random.default_rng(0); rng.shuffle(cal); cal = cal[:6000]
    F = features([c[0] for c in cal], names, DG, DE); y = np.array([c[1] for c in cal]); hitorig = np.array([f[5] == c[2] for f, c in zip(F, cal)])
    X = np.array([f[:5] for f in F], np.float64); print(f"문턱 자료 {len(cal)} (진짜 {int((y == 0).sum())}), 1위 후보=자기 원본 비율 진짜 {hitorig[y == 0].mean():.3f} 가짜 {hitorig[y == 1].mean():.3f}", flush=True)
    R = (y == 0) & hitorig
    T = dict(m=np.quantile(X[R, 0], .995), gap=np.quantile(X[R, 1], .01), r_dgmax=np.quantile(X[R, 2], .99), r_dem=np.quantile(X[R, 3], .99), r_demax=np.quantile(X[R, 4], .99),
             f_dgmax=np.quantile(X[R, 2], .999), f_dem=np.quantile(X[R, 3], .999), f_demax=np.quantile(X[R, 4], .999))
    print("문턱", {k: round(float(v), 3) for k, v in T.items()}, flush=True); json.dump({k: float(v) for k, v in T.items()}, open(L / "preds/finefp_thresholds.json", "w"))
    def apply(p, X):
        match = (X[:, 0] <= T["m"]) & (X[:, 1] >= T["gap"])
        real_ok = match & (X[:, 2] <= T["r_dgmax"]) & (X[:, 3] <= T["r_dem"]) & (X[:, 4] <= T["r_demax"])
        fake_ok = match & ((X[:, 2] > T["f_dgmax"]) | (X[:, 3] > T["f_dem"]) | (X[:, 4] > T["f_demax"]))
        q = p.copy(); q[real_ok] = np.minimum(q[real_ok], .05); q[fake_ok] = np.maximum(q[fake_ok], .95); return q, match, real_ok, fake_ok
    m0, r0, f0 = apply(np.full(len(y), .5), X)[1:]
    print(f"문턱 자료에서: 원본판정 진짜 {m0[y == 0].mean():.3f} 가짜 {m0[y == 1].mean():.3f} | 진짜확정 진짜 {r0[y == 0].mean():.3f} 가짜 {r0[y == 1].mean():.4f} | 가짜확정 진짜 {f0[y == 0].mean():.4f} 가짜 {f0[y == 1].mean():.3f}", flush=True)
    # 판정 자료
    def evalset(nm, paths, yy, p5, grp):
        Fx = features(paths, names, DG, DE); Xx = np.array([f[:5] for f in Fx], np.float64); q, m, ro, fo = apply(p5, Xx)
        acc = lambda p: ((p >= .5) == yy).mean() if len(set(yy)) == 1 else .5 * (p[yy == 0] < .5).mean() + .5 * (p[yy == 1] >= .5).mean()
        print(f"[{nm}] n={len(yy)} 원본판정 {m.mean():.3f} 진짜확정 {ro.mean():.3f} 가짜확정 {fo.mean():.3f} | 나단5 {acc(p5):.4f} → F {acc(q):.4f}", flush=True)
        if (yy == 0).any(): print(f"     진짜 오탐 {(p5[yy == 0] >= .5).mean():.4f} → {(q[yy == 0] >= .5).mean():.4f} (진짜확정 잘못 {(ro & (yy == 1)).sum()} · 가짜확정 잘못 {(fo & (yy == 0)).sum()})", flush=True)
        if (yy == 1).any(): print(f"     가짜 탐지 {(p5[yy == 1] >= .5).mean():.4f} → {(q[yy == 1] >= .5).mean():.4f}", flush=True)
        return q, Xx
    r2 = rows_of("im_set_r2"); yr = np.array([int(r["label"]) for r in r2]); p5 = np.load(L / "preds/_r2_나단5.npy"); gr = np.array([r["group"] for r in r2])
    qr, _ = evalset("R2 보류", [str(L / "im_set_r2/img" / r["file"]) for r in r2], yr, p5, gr)
    gs_ = np.array(sorted(set(gr))); gi = {g: np.where(gr == g)[0] for g in gs_}; rng = np.random.default_rng(7); bs = []
    acc2 = lambda p, idx: .5 * (p[idx][yr[idx] == 0] < .5).mean() + .5 * (p[idx][yr[idx] == 1] >= .5).mean()
    for _ in range(NB):
        idx = np.concatenate([gi[g] for g in rng.choice(gs_, len(gs_))])
        if (yr[idx] == 0).any() and (yr[idx] == 1).any(): bs.append(acc2(qr, idx) - acc2(p5, idx))
    pw = float((np.array(bs) <= 0).mean()); print(f"     R2 묶음 부트스트랩 p_worse {pw:.4f}", flush=True)
    z = np.load(L / "preds/sealed_c3v2.npz", allow_pickle=True); st, pp5, sp = z["set"], z["p5"], z["path"]
    res = {}
    for sn, lab in (("real", 0), ("qwen", 1), ("v7", 1), ("brown", 0), ("bright", 0)):
        m = st == sn; res[sn] = evalset("봉인 " + sn, list(sp[m]), np.full(m.sum(), lab), pp5[m], None)
    print(f"\n== 판정(IM_정밀원본대조_F_v1.md, 탐색적; 공개 21장은 별도 패키지 비교) ==")
    print(f"  R2 상승&p {acc2(qr, np.arange(len(yr))) > acc2(p5, np.arange(len(yr))) and pw < .025} · 봉인 진짜 오탐 비증가 {(res['real'][0] >= .5).mean() <= (pp5[st == 'real'] >= .5).mean()}"
          f" · 봉인 가짜 탐지 비감소 {((np.r_[res['qwen'][0], res['v7'][0]]) >= .5).mean() >= (pp5[np.isin(st, ['qwen', 'v7'])] >= .5).mean()}")

if __name__ == "__main__":
    main()
