"""IM12: 공개 21장 → AI허브 지문 DB(IM11) 최근접 대조 + 최근접 원본의 원해상도 재확인.
 1) 지문(12x16 RGB, 같은 방향만) 평균 절대차로 상위 3개
 2) 1위 원본을 tar에서 꺼내 IM10 방식(덮게 축소·가운데 크롭 + ECC 정렬)으로 원해상도 잔차
출력: 표(파일·유형·1위 이름·지문차 1/2/3위·원해상도 잔차 평균/중앙·블록 이탈 비율) → logs, local/match/lookup_public.csv
"""
import sys, csv, tarfile, numpy as np, cv2
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im11_fingerprint_db as F  # noqa
from im10_match_check import cover, PUB  # noqa

LOCAL = Path(__file__).resolve().parents[1] / "local"; ROOT = F.ROOT

def load_db():
    names, tars, fps, ports = [], [], [], []
    for p in sorted((LOCAL / "fp_db").glob("*.npz")):
        z = np.load(p); n = len(z["names"])
        names += list(z["names"]); tars += [p.stem] * n; fps.append(z["fp"]); ports.append(z["portrait"])
    return np.array(names), np.array(tars), np.concatenate(fps).astype(np.float32), np.concatenate(ports)

def fetch(tarname, stem):
    sp = "2.Validation/원천데이터" if tarname.startswith("VS") else "1.Training/원천데이터_add_20260807"
    dst = LOCAL / "match" / f"{stem}.jpg"
    if dst.exists(): return dst
    with tarfile.open(ROOT / sp / f"{tarname}.tar", "r:") as tf:
        for m in tf:
            if Path(m.name).stem == stem: dst.write_bytes(tf.extractfile(m).read()); return dst

def fullres(pub, orig):
    P = Image.open(PUB / pub).convert("RGB"); a = np.asarray(P); b = np.asarray(cover(Image.open(orig).convert("RGB"), P.size))
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32) for x in (a, b)); warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(ga, gb, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        b = cv2.warpAffine(b, warp, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except cv2.error: pass
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(2); H, W = d.shape
    m = d[:H // 32 * 32, :W // 32 * 32].reshape(H // 32, 32, W // 32, 32).mean((1, 3))
    return d.mean(), np.median(d), (m > 3 * max(np.median(m), .5)).mean()

def main():
    names, tars, fps, ports = load_db(); print(f"DB {len(names)}장, tar {len(set(tars))}개", flush=True)
    lab = list(csv.DictReader(open(PUB.parent / "sample_labels.csv"))); rows = []
    for r in lab:
        q, port = F.fingerprint(Image.open(PUB / r["file"]))
        q = (q if port else q.transpose(1, 0, 2)).astype(np.float32)
        idx = np.where(ports == port)[0]; d = np.abs(fps[idx] - q).mean((1, 2, 3)); o = idx[np.argsort(d)[:3]]; ds = np.sort(d)[:3]
        src = fetch(tars[o[0]], names[o[0]]); fm, fmed, fblk = fullres(r["file"], src)
        rows.append(dict(file=r["file"], type=r["type"], top1=names[o[0]], tar=tars[o[0]], d1=round(float(ds[0]), 2), d2=round(float(ds[1]), 2), d3=round(float(ds[2]), 2),
                         full_mean=round(float(fm), 2), full_median=round(float(fmed), 2), block_out=round(float(fblk), 3)))
        x = rows[-1]; print(f"{x['file']:9s} {x['type']:13s} 1위 {x['top1']:24s} 지문차 {x['d1']:6.2f} / {x['d2']:6.2f} / {x['d3']:6.2f} | 원해상도 평균 {x['full_mean']:6.2f} 중앙 {x['full_median']:5.2f} 이탈블록 {x['block_out']:.3f}", flush=True)
    with open(LOCAL / "match" / "lookup_public.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
