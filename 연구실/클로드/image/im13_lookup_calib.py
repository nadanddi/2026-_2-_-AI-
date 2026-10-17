"""IM13: 원본 대조 규칙의 문턱을 학습 자료만으로 정하기 위한 분포 측정 (공개 21장 미사용).
질의: (a) 우리 실제 사진(v4·r2 세트의 real, 빨간 조명 제외) — 두 경우로 측정
        ① 원본 포함: DB 그대로(정확한 원본이 있는 실제)
        ② 원본 제외: DB에서 자기 원본(같은 파일명) 제거 → '정확한 원본은 없고 이웃 프레임만 있는 실제'
      (b) 우리 가짜(v4·sd2·r2) — 원본 포함 DB(가짜의 원본은 AI허브에 있음). 유형별.
지표: d1(1위 지문 차), d2(2위), gap = d2 − d1.
출력: 분위수 표 + local/match/calib_lookup.npz
"""
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im11_fingerprint_db as F  # noqa
from im12_lookup_public import load_db  # noqa
import im05c_eval_v3 as V3  # noqa

LOCAL = Path(__file__).resolve().parents[1] / "local"

def query(fps, ports, names, img, exclude=None):
    q, port = F.fingerprint(img); q = (q if port else q.transpose(1, 0, 2)).astype(np.float32)
    idx = np.where(ports == port)[0]
    if exclude is not None: idx = idx[names[idx] != exclude]
    d = np.abs(fps[idx] - q).mean((1, 2, 3)); s = np.partition(d, 1)[:2]; s.sort()
    return s[0], s[1]

def main(max_per=400):
    names, tars, fps, ports = load_db()
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    rng = np.random.default_rng(0); out = {}
    for sname in ["im_set_v4", "im_set_r2", "im_set_sd2"]:
        for r in csv.DictReader(open(LOCAL / sname / "manifest.csv", encoding="utf-8")):
            if r["source"] in red or r["kind"] not in V3.TYPE or "tile_whole=1" in r["info"]: continue
            out.setdefault(V3.TYPE[r["kind"]] if r["kind"] != "real_resampled" else "real_rs", []).append((sname, r))
    res = {}
    for k, lst in out.items():
        sel = [lst[i] for i in rng.permutation(len(lst))[:max_per]]
        a = []
        for sname, r in sel:
            img = Image.open(LOCAL / sname / "img" / r["file"])
            src = Path(r["source"]).stem if r["source"] else None
            if k == "real":
                a.append(query(fps, ports, names, img) + query(fps, ports, names, img, exclude=src))
            else:
                a.append(query(fps, ports, names, img) + (np.nan, np.nan))
        res[k] = np.array(a)
    np.savez(LOCAL / "match" / "calib_lookup.npz", **res)
    q = [5, 25, 50, 75, 95, 99]
    print("분위수", q)
    for k, a in res.items():
        print(f"{k:14s} n={len(a)} d1 {np.percentile(a[:, 0], q).round(2)} gap {np.percentile(a[:, 1] - a[:, 0], q).round(2)}")
        if k == "real":
            print(f"{'real(원본 제외)':14s}        d1 {np.percentile(a[:, 2], q).round(2)} gap {np.percentile(a[:, 3] - a[:, 2], q).round(2)}")

if __name__ == "__main__":
    main()
