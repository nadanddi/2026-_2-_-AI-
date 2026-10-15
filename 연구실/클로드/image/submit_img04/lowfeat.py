"""저수준 38개 통계 — 학습 때 쓴 im06_lowlevel.py의 feats()와 같은 코드(이름만 low_features)."""
import numpy as np, cv2

def tile_stats(x, T=128):
    h, w = x.shape; v = []
    for y in range(0, h - T + 1, T):
        for xx in range(0, w - T + 1, T): v.append(x[y:y + T, xx:xx + T])
    return v

def blockiness(g):
    d_h = np.abs(np.diff(g, axis=1)); d_v = np.abs(np.diff(g, axis=0))
    bh = d_h[:, 7::8].mean() / (d_h.mean() + 1e-6); bv = d_v[7::8, :].mean() / (d_v.mean() + 1e-6)
    return (bh + bv) / 2

def low_features(path):
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr.shape[0] < bgr.shape[1]: bgr = np.ascontiguousarray(np.rot90(bgr))   # 방향 통일(세로)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32); res = g - med
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))); hh, ww = F.shape; cy, cx = hh // 2, ww // 2
    yy, xx = np.ogrid[:hh, :ww]; rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(hh, ww) / 2)
    e = [F[(rr >= a) & (rr < b)].mean() for a, b in [(0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.0)]]
    e = np.log(np.array(e) + 1e-6); e = e - e[0]
    ext = np.mean(np.abs(res) > 60)
    r_t = np.array([t.std() for t in tile_stats(res)]); l_t = np.array([t.var() for t in tile_stats(lap)])
    s_t = np.array([t.mean() for t in tile_stats(hsv[..., 1])])
    x_t = np.array([np.mean(np.abs(t) > 60) for t in tile_stats(res)])
    def d(v): lv = np.log(v + 1e-6); return [lv.mean(), lv.std(), lv.min(), lv.max(), np.percentile(lv, 10), np.percentile(lv, 90)]
    k = ((res - res.mean()) ** 4).mean() / (res.var() ** 2 + 1e-6)
    out = [res.std(), np.log(k), ext, np.log(lap.var() + 1e-6), blockiness(g),
           hsv[..., 1].mean(), hsv[..., 1].std(), hsv[..., 2].mean(), hsv[..., 2].std(), np.mean(g >= 250), np.mean(g <= 5)]
    out += list(e[1:]) + d(r_t) + d(l_t) + d(s_t) + [x_t.mean(), x_t.max(), x_t.std()]
    # 색 채널 간 잔차 상관(생성 모델은 채널 상관이 다름)
    rc = [cv2.medianBlur(bgr[..., c], 3).astype(np.float32) for c in range(3)]
    rs = [bgr[..., c].astype(np.float32) - rc[c] for c in range(3)]
    out += [np.corrcoef(rs[0].ravel(), rs[1].ravel())[0, 1], np.corrcoef(rs[1].ravel(), rs[2].ravel())[0, 1]]
    return np.nan_to_num(np.array(out, np.float32))

