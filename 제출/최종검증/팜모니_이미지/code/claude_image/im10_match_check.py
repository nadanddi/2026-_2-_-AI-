"""IM10: 원본 대조 가설 점검 — 공개 사진과 AI허브 후보 원본을 정렬해 픽셀 차이를 본다.
 1) 배포 처리 재현(덮게 축소 → 가운데 크롭)으로 바로 비교
 2) 안 맞으면 ECC(유사변환)로 정렬 후 비교
출력: 정렬 오차(잔차 RMS), 잔차의 공간 분포(32px 블록 RMS 지도 → 상위 블록 비율), 차이 지도 PNG
사용: python im10_match_check.py <공개 파일> <원본 경로> [<원본 경로> ...]
"""
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image

PUB = Path(__file__).resolve().parents[3] / "공용/대회자료/이미지데이터/참가자_배포/공개샘플/images"
OUT = Path(__file__).resolve().parents[1] / "local/match"

def cover(img, size):
    w, h = img.size; tw, th = size; s = max(tw / w, th / h); nw, nh = max(tw, round(w * s)), max(th, round(h * s))
    r = img.resize((nw, nh), Image.LANCZOS); x0, y0 = (nw - tw) // 2, (nh - th) // 2; return r.crop((x0, y0, x0 + tw, y0 + th))

def report(tag, a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(2)
    H, W = d.shape; B = 32; m = d[:H // B * B, :W // B * B].reshape(H // B, B, W // B, B).mean((1, 3))
    print(f"  [{tag}] 잔차 평균 {d.mean():.2f}, 중앙 {np.median(d):.2f}, 99분위 {np.percentile(d, 99):.1f} | 블록 RMS 중앙 {np.median(m):.2f}, 최대 {m.max():.1f}, "
          f"중앙의 3배 넘는 블록 비율 {(m > 3 * np.median(m)).mean():.3f}")
    return d, m

def main(pub, *origs):
    P = Image.open(PUB / pub).convert("RGB"); a = np.asarray(P)
    for o in origs:
        O = Image.open(o).convert("RGB"); print(f"{pub} vs {Path(o).name}  (원본 {O.size})")
        b = np.asarray(cover(O, P.size)); d, m = report("배포처리 재현", a, b)
        # ECC 정렬(회색조, 유사변환)
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32); gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ga, gb, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
            b2 = cv2.warpAffine(b, warp, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            d2, m2 = report("ECC 정렬 후", a, b2); print("   warp", np.round(warp, 4).tolist())
        except cv2.error as e:
            print("   ECC 실패", str(e)[:80]); d2, m2 = d, m
        OUT.mkdir(parents=True, exist_ok=True)
        vis = np.clip(d2 * 4, 0, 255).astype(np.uint8)
        Image.fromarray(np.concatenate([a, np.stack([vis] * 3, 2)], 1)).resize((a.shape[1], a.shape[0] // 2)).save(OUT / f"diff_{pub[:-4]}_{Path(o).stem}.png")

if __name__ == "__main__":
    main(*sys.argv[1:])
