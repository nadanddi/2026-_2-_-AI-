"""IM06b: 영역 불일치 특징 (중간 비평 제안 2, 후보 A의 일부) — 부분 편집·합성은 사진 일부만 통계가 다르다.
 (1) 잡음 수준 지도: 32px 블록마다 잔차(원본-중앙값3)의 MAD → 로그 지도의 분산·범위·상위/하위 분위 차·공간 기울기 평균·최대 이탈(z)
 (2) JPEG 8x8 격자 블록성 지도: 64px 블록마다 경계 불연속비 → 분산·최소·최대·최대 이탈
 (3) 선명도 지도: 32px 블록 라플라시안 분산(로그) → 같은 통계
 (4) 채도 지도: 32px 블록 평균 채도 → 최대 이탈
 (5) 이웃 블록 불일치: 잡음 지도와 선명도 지도에서 이웃 차의 99분위(국소 경계)
사용: python im06b_region_feats.py <name> <img_dir> <manifest.csv> → local/feats/<name>_reg.npz
"""
import sys, csv, time, numpy as np, cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

LOCAL = Path(__file__).resolve().parents[1] / "local"

def blockmap(x, B, fn):
    h, w = x.shape[0] // B * B, x.shape[1] // B * B
    v = x[:h, :w].reshape(h // B, B, w // B, B).swapaxes(1, 2)
    return fn(v)

def stats(m):
    sc = max(1.4826 * np.median(np.abs(m - np.median(m))), .5 * m.std(), 1e-3)   # MAD=0(양자화) 대비
    z = (m - np.median(m)) / sc
    dx = np.abs(np.diff(m, axis=1)); dy = np.abs(np.diff(m, axis=0)); nb = np.concatenate([dx.ravel(), dy.ravel()])
    return [m.std(), np.percentile(m, 95) - np.percentile(m, 5), np.abs(z).max(), np.mean(np.abs(z) > 3), nb.mean(), np.percentile(nb, 99)]

def feats(path):
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr.shape[0] < bgr.shape[1]: bgr = np.ascontiguousarray(np.rot90(bgr))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    noise = np.log(blockmap(np.abs(res), 32, lambda v: np.median(v, axis=(2, 3))) + .05)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3); sharp = np.log(blockmap(lap, 32, lambda v: v.var(axis=(2, 3))) + 1)
    dh = np.abs(np.diff(g, axis=1)); dv = np.abs(np.diff(g, axis=0))
    def blk(v):  # v: (H/64, W/64, 64, 64) 블록 안 8x8 경계 불연속비
        a = np.abs(np.diff(v, axis=3)); b = np.abs(np.diff(v, axis=2))
        return (a[..., 7::8].mean((2, 3)) + b[:, :, 7::8, :].mean((2, 3))) / (a.mean((2, 3)) + b.mean((2, 3)) + 1e-6)
    jb = blockmap(g, 64, blk)
    sat = blockmap(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32), 32, lambda v: v.mean(axis=(2, 3)))
    out = stats(noise) + stats(sharp) + stats(jb) + stats(sat)
    # 잡음-선명도 관계의 이탈(같은 선명도인데 잡음만 다른 영역)
    A = np.stack([sharp.ravel(), np.ones(sharp.size)], 1); coef, *_ = np.linalg.lstsq(A, noise.ravel(), rcond=None)
    r = noise.ravel() - A @ coef; out += [r.std(), np.abs(r).max(), np.percentile(np.abs(r), 99)]
    return np.nan_to_num(np.array(out, np.float32))

def main(name, img_dir, manifest):
    files = [r["file"] for r in csv.DictReader(open(manifest, encoding="utf-8"))]
    t = time.time()
    with ProcessPoolExecutor(4) as ex: X = list(ex.map(feats, [Path(img_dir) / f for f in files], chunksize=8))
    np.savez_compressed(LOCAL / "feats" / f"{name}_reg.npz", feats=np.stack(X), files=np.array(files)); print("saved", np.stack(X).shape, f"{time.time()-t:.0f}s")

if __name__ == "__main__":
    main(*sys.argv[1:4])
