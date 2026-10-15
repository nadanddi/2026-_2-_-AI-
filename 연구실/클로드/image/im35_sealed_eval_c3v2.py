"""IM35: C3v2 판정(IM_C3v2_v1.md) — 나단4 대 나단5 패키지 predict 그대로. 봉인 자료 두 번째 열람. 서버(nadan, GPU 0)."""
import sys, csv, zlib, importlib, numpy as np
from pathlib import Path
B = Path.home() / "storage/nadan/farmai_img"; S = B / "data/local/sealed_c3"

def run(pk, paths):
    sys.path.insert(0, str(pk)); I = importlib.import_module("inference"); m = I.load_model(str(pk / "model")); p = np.array(I.predict(m, paths))
    sys.path.remove(str(pk)); del sys.modules["inference"]; return p

def main():
    R = []
    for sn in ("real", "qwen", "v7", "brown", "bright"):
        for r in csv.DictReader(open(S / sn / "manifest.csv", encoding="utf-8")):
            if sn in ("brown", "bright") and zlib.crc32(r["file"].encode()) % 2 == 0: continue   # 학습에 쓴 절반 제외
            R.append(dict(r, set=sn, path=str(S / sn / "img" / r["file"])))
    paths = [r["path"] for r in R]; st = np.array([r["set"] for r in R])
    p4 = run(B / "submit_img04/팜모니_나단4", paths); p5 = run(B / "submit_img05/팜모니_나단5", paths)
    np.savez(B / "data/local/preds/sealed_c3v2.npz", p4=p4, p5=p5, set=st, path=np.array(paths))
    r = {s: ((p4[st == s] >= .5).mean(), (p5[st == s] >= .5).mean(), int((st == s).sum())) for s in ("real", "qwen", "v7", "brown", "bright")}
    for s, (a, b, n) in r.items(): print(f"{s:6s} n={n:3d} {'오탐' if s in ('real', 'brown', 'bright') else '탐지'} 나단4 {a:.4f} 나단5 {b:.4f}")
    c = [r["bright"][1] <= r["bright"][0], r["brown"][1] <= r["brown"][0] + .01, r["real"][1] <= r["real"][0] + .005, r["v7"][1] >= r["v7"][0] + .03, r["qwen"][1] >= r["qwen"][0] - .01]
    print(f"\n== 판정(IM_C3v2_v1.md) ==\n  밝음 {c[0]} · 갈변 {c[1]} · 봉인진짜 {c[2]} · 7판+.03 {c[3]} · Qwen {c[4]} → {'후보' if all(c) else '불채택'}")

if __name__ == "__main__":
    main()
