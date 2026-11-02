"""IM26: 국소 주파수 스펙트럼·잡음 불일치 이상치 점수(무학습, 사전 시험). im25와 같은 대상.
64px 타일(간격 32)마다: 잡음 잔차(원본-중앙값3) 표준편차, 고주파 에너지 비(FFT 반경 .5~1 / 전체), 라플라시안 분산, 8x8 블록 경계비.
각 지표를 사진 안 중앙값·MAD로 표준화한 |z|의 최댓값과 상위 1% 평균 → 사진 점수(조각이 '주변과 얼마나 튀나').
출력 local/feats/localspec_scores.npz (set, fname, 지표별 zmax·z99)
"""
import sys, csv, time, numpy as np, cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im25_trufor_score import jobs, LOCAL  # noqa
T, ST = 64, 32
yy, xx = np.ogrid[:T, :T]; RR = np.sqrt((yy - T / 2) ** 2 + (xx - T / 2) ** 2) / (T / 2); HI = RR >= .5

def score(p):
    g = cv2.cvtColor(cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32); lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    h, w = g.shape; M = []
    for y in range(0, h - T + 1, ST):
        for x in range(0, w - T + 1, ST):
            t = g[y:y + T, x:x + T]; F = np.abs(np.fft.fftshift(np.fft.fft2(t - t.mean()))) + 1e-6
            dh = np.abs(np.diff(t, axis=1)); dv = np.abs(np.diff(t, axis=0)); off = (x % 8, y % 8)
            blk = (dh[:, (7 - off[0]) % 8::8].mean() + dv[(7 - off[1]) % 8::8, :].mean()) / (dh.mean() + dv.mean() + 1e-6)
            M.append([np.log(res[y:y + T, x:x + T].std() + 1e-3), np.log(F[HI].sum() / F.sum()), np.log(lap[y:y + T, x:x + T].var() + 1e-3), blk])
    M = np.array(M); med = np.median(M, 0); mad = np.median(np.abs(M - med), 0) * 1.4826 + 1e-6; Z = np.abs(M - med) / mad
    Zs = np.sort(Z, 0); k = max(1, len(Z) // 100)
    return np.r_[Zs[-1], Zs[-k:].mean(0)]

def main():
    J = jobs(); t = time.time()
    with ProcessPoolExecutor(16) as ex: R = list(ex.map(score, [p for _, p in J], chunksize=8))
    R = np.array(R); names = ["noise", "hifreq", "lap", "block"]
    np.savez(LOCAL / "feats" / "localspec_scores.npz", set=np.array([s for s, _ in J]), fname=np.array([p.name for _, p in J]),
             **{f"{n}_zmax": R[:, i] for i, n in enumerate(names)}, **{f"{n}_z99": R[:, 4 + i] for i, n in enumerate(names)})
    print("saved", R.shape, f"{time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main()
