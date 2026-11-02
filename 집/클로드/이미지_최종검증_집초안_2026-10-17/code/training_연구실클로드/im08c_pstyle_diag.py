"""IM08c: r2 부분 스타일(프로그램) 가짜의 탐지 실패 원인 진단 — 짝 실제와의 픽셀 차이(강도·면적)·변형 종류별 탐지율. 모델 = B2+A(im08b와 같음)."""
import sys, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05c_eval_v3 as V3  # noqa
import im05d_eval_v4 as D  # noqa
from im05k_backbones import bb  # noqa
LOCAL = D.LOCAL

def main():
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2", "im_set_r2"]); keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Xl = X["low"][keep]
    fz = {}
    for sn in ("im_set_v4", "im_set_sd2", "im_set_r2"): fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R]); kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R])
    st = np.array([r["_set"] for r in R]); m = np.isin(st, ["im_set_v4", "im_set_sd2"]); w = np.zeros(len(R)); w[m] = V3.weights(kind[m]); idx = np.where(m)[0]
    ev = np.where((st == "im_set_r2") & (kind == "partial_style"))[0]
    p = (D.fit("clip", Xb, y, w, idx, 0).predict_proba(Xb[ev])[:, 1] + D.fit("low", Xl, y, w, idx, 0).predict_proba(Xl[ev])[:, 1]) / 2
    mad, area, ops = [], [], []
    for i in ev:
        f = R[i]["file"]; a = np.asarray(Image.open(LOCAL / "im_set_r2" / "img" / f), np.float32); b = np.asarray(Image.open(LOCAL / "im_set_r2" / "img" / f.replace("_fake", "_real")), np.float32)
        d = np.abs(a - b).mean(2); ch = d > 6; mad.append(d[ch].mean() if ch.any() else 0); area.append(ch.mean()); ops.append(R[i]["info"])
    mad, area = np.array(mad), np.array(area); det = p >= .5
    print(f"부분 스타일 {len(ev)}장, 탐지 {det.mean():.2f}")
    print("바뀐 면적(픽셀차>6 비율) 구간별 탐지:")
    for lo, hi in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1)]:
        s = (area >= lo) & (area < hi); print(f"   {lo:.2f}~{hi:.2f}: n={s.sum():4d} 탐지 {det[s].mean() if s.any() else float('nan'):.2f}")
    print("바뀐 곳 평균 차이 구간별 탐지:")
    for lo, hi in [(0, 8), (8, 12), (12, 20), (20, 40), (40, 255)]:
        s = (mad >= lo) & (mad < hi); print(f"   {lo}~{hi}: n={s.sum():4d} 탐지 {det[s].mean() if s.any() else float('nan'):.2f}")
    print("변형 종류(포함 여부)별 탐지:")
    for op in ["lab", "tone", "hsv", "detail", "smooth", "clahe"]:
        s = np.array([op in o.split("+") for o in ops]); print(f"   {op:6s}: 포함 n={s.sum():4d} 탐지 {det[s].mean():.2f} | 미포함 {det[~s].mean():.2f}")
    print(f"면적<.05 비율 {np.mean(area < .05):.2f}, 그중 탐지 {det[area < .05].mean():.2f}")

if __name__ == "__main__":
    main()
