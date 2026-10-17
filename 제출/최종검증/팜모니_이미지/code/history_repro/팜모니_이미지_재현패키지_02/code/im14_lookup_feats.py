"""IM14: 후보 L1용 원본 대조 특징 계산 (IM_후보L1_v1.md 정의).
대상: im_set_v4, im_set_sd2 전 행 + 빨간 진단(im_set_v2 red_list) + 공개 21장.
특징: d1, d2, gap=d2-d1 (지문 12x16 RGB, 같은 방향). 실제 행의 1/3(원본 파일 crc32 % 3 == 0)은 자기 원본을 DB에서 제외(excl=1).
출력: local/feats/<set>_look.npz (files, feats=(N,3), excl)
"""
import sys, csv, zlib, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im11_fingerprint_db as F  # noqa
from im12_lookup_public import load_db  # noqa

LOCAL = Path(__file__).resolve().parents[1] / "local"
PUBD = Path(__file__).resolve().parents[3] / "공용/대회자료/이미지데이터/참가자_배포/공개샘플"

class DB:
    def __init__(s):
        names, tars, fps, ports = load_db(); s.names = names
        s.by = {p: (np.where(ports == p)[0], fps[ports == p].reshape(-1, 576)) for p in (True, False)}
    def look(s, img, exclude=None):
        q, port = F.fingerprint(img); q = (q if port else q.transpose(1, 0, 2)).astype(np.float32).reshape(576)
        idx, M = s.by[port]; d = np.abs(M - q).mean(1)
        if exclude is not None:
            d = d.copy(); d[s.names[idx] == exclude] = np.inf
        p = np.argpartition(d, 1)[:2]; a, b = sorted(d[p]); return a, b, b - a

def run(db, name, img_dir, rows, excl_flags):
    X = np.array([db.look(Image.open(Path(img_dir) / r["file"]), Path(r["source"]).stem if e else None) for r, e in zip(rows, excl_flags)], np.float32)
    np.savez(LOCAL / "feats" / f"{name}_look.npz", files=np.array([r["file"] for r in rows]), feats=X, excl=np.array(excl_flags, int))
    print(name, X.shape, "excl", int(np.sum(excl_flags)), flush=True)

def main():
    db = DB()
    for s in ["im_set_v4", "im_set_sd2", "im_set_r2"]:
        rows = list(csv.DictReader(open(LOCAL / s / "manifest.csv", encoding="utf-8")))
        ex = [r["kind"] in ("real", "real_resampled") and r["source"] != "" and zlib.crc32(r["source"].encode()) % 3 == 0 for r in rows]
        run(db, s, LOCAL / s / "img", rows, ex)
    red = [l.strip() for l in open(LOCAL / "im_set_v2" / "red_list.txt", encoding="utf-8") if l.strip()]
    run(db, "im_set_v2red", LOCAL / "im_set_v2" / "img", [dict(file=f, source="") for f in red], [False] * len(red))
    pub = list(csv.DictReader(open(PUBD / "sample_labels.csv")))
    run(db, "public21", PUBD / "images", [dict(file=r["file"], source="") for r in pub], [False] * len(pub))

if __name__ == "__main__":
    main()
