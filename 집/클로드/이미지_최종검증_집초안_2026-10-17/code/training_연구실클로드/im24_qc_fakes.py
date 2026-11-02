"""IM24: 가짜 품질 점검(진단, 판정 없음) — 6판 부분생성·5판 partial_style_xl.
원본 배포형(같은 난수로 재현) vs 가짜: 마스크 안/밖 평균 |차이|(RGB 0~255).
- 안 차이가 아주 작으면 '편집이 사실상 안 됨'(가짜 라벨이지만 진짜와 같음 → 라벨 잡음)
- 밖 차이가 크면 되붙임 실패
출력: logs 요약 + local/qc/<set>_{random,low}.jpg (원본|가짜|마스크 표시 3단, 무작위 8장과 안 차이 최저 8장)
"""
import sys, csv, re, random, zlib, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import to_np, blob_mask, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
from im15_fakes_v5 import rawp, sources, object_mask, PEDIT, ART  # noqa

def recon(sn, i, man):
    if sn == "im_set_v6":
        lr = random.Random(zlib.crc32(f"v6|pgen|{i}".encode())); lr.randint(0, 2**31); src = man[lr.randrange(len(man))]
        a = to_np(dist_cover(Image.open(rawp(src["file"])).convert("RGB"), lr))
        return a, (object_mask(a, lr, .02, .15) if lr.random() < .5 else blob_mask(a.shape, lr, (.02, .15)))
    lr = random.Random(zlib.crc32(f"v5|partial_style_xl|{i}".encode())); lr.randint(0, 2**31); src = man[lr.randrange(len(man))]
    a = to_np(dist_cover(Image.open(rawp(src["file"])).convert("RGB"), lr)); lr.randint(25, 35); lr.uniform(5, 8)
    photo = lr.random() < .5; lr.uniform(.30, .45) if photo else lr.uniform(.40, .55); lr.choice(PEDIT if photo else ART)
    return a, object_mask(a, lr)

def sheet(items, path):
    tiles = []
    for a, f, m in items:
        vis = f.copy(); e = (m > 0) & ((np.roll(m, 3, 0) == 0) | (np.roll(m, 3, 1) == 0)); vis[e] = (255, 0, 0)
        tiles.append(np.concatenate([a, f, vis], 1)[::4, ::4])
    Image.fromarray(np.concatenate(tiles, 0)).save(path, quality=88)

def main():
    man = sources(); (LOCAL / "qc").mkdir(exist_ok=True)
    for sn, kind in [("im_set_v6", "partial_gen_v6"), ("im_set_v5", "partial_style_xl")]:
        rows = [r for r in csv.DictReader(open(LOCAL / sn / "manifest.csv", encoding="utf-8")) if r["kind"] == kind]
        stats, keep = [], {}
        for r in rows:
            i = int(re.search(r"_(\d{5})\.jpg", r["file"]).group(1)); a, m = recon(sn, i, man)
            f = np.asarray(Image.open(LOCAL / sn / "img" / r["file"]).convert("RGB"))
            if f.shape != a.shape: stats.append((r["file"], np.nan, np.nan, m.mean())); continue
            d = np.abs(f.astype(np.float32) - a.astype(np.float32)).mean(2); mi = d[m > 0].mean(); mo = d[m == 0].mean()
            stats.append((r["file"], mi, mo, m.mean())); keep[r["file"]] = (a, f, m)
        mi = np.array([s[1] for s in stats]); mo = np.array([s[2] for s in stats]); ar = np.array([s[3] for s in stats])
        print(f"{sn}: n={len(stats)} 크기불일치 {np.isnan(mi).sum()} | 마스크 안 차이 중앙값 {np.nanmedian(mi):.1f} 분위(5,25,75) {np.round(np.nanquantile(mi, [.05, .25, .75]), 1)}"
              f" | 밖 차이 중앙값 {np.nanmedian(mo):.2f} 최대 {np.nanmax(mo):.2f} | 안 차이<5 비율 {np.nanmean(mi < 5):.3f} <10 {np.nanmean(mi < 10):.3f}", flush=True)
        for lo, hi in [(0, .05), (.05, .10), (.10, .15), (.15, 1)]:
            s = (ar >= lo) & (ar < hi)
            if s.sum(): print(f"   넓이 {lo:.2f}~{hi:.2f}: n={s.sum()} 안 차이 중앙값 {np.nanmedian(mi[s]):.1f} <10 비율 {np.nanmean(mi[s] < 10):.2f}", flush=True)
        if sn == "im_set_v6":
            g = np.array([r["info"].split("|")[0] for r in rows])
            for gg in ("xl", "sd15"): print(f"   {gg}: 안 차이 중앙값 {np.nanmedian(mi[g == gg]):.1f} <10 비율 {np.nanmean(mi[g == gg] < 10):.2f}", flush=True)
        rng = np.random.default_rng(0); fs = [s[0] for s in stats if s[0] in keep and keep[s[0]][0].shape[0] > keep[s[0]][0].shape[1]]
        sheet([keep[f] for f in rng.choice(fs, 8, replace=False)], LOCAL / "qc" / f"{sn}_random.jpg")
        low = [s[0] for s in sorted([s for s in stats if s[0] in keep and keep[s[0]][0].shape[0] > keep[s[0]][0].shape[1]], key=lambda s: s[1])[:8]]
        sheet([keep[f] for f in low], LOCAL / "qc" / f"{sn}_low.jpg")
        np.savez(LOCAL / "qc" / f"{sn}_qc.npz", files=np.array([s[0] for s in stats]), mi=mi, mo=mo, area=ar)

if __name__ == "__main__":
    main()
