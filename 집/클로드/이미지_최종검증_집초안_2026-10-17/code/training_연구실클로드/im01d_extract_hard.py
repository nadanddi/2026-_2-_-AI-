"""IM01d: 후보 H1 — '어려운 실제'(자연 흐림·흔들림·센서 잡음) 추가 추출 (사용자 지시 10-11).
근거: 우리 실제 오탐 상위가 자연 흐림·흔들림·잡음 사진이고, 문제설명서가 이런 실제를 라벨 0이라고 명시.
방법:
 - 기준 분포: 이미 뽑은 실제(빨간 조명 제외) 600장을 dist(덮게 축소·가운데 크롭) 크기에서 측정 → 선명도(라플라시안 분산 로그) 15분위, 잡음(중앙값 잔차 MAD) 90분위를 문턱으로 고정.
 - 모든 tar(금실·설향, VS·TS)에서 아직 안 뽑은 사진을 tar당 최대 SCAN장 훑어, 문턱을 넘는 것(흐림 OR 잡음)만 보관. 빨간 조명(평균 R-G>25)은 제외.
 - 품종별 목표 TARGET장씩(5:5), tar당 최대 PER_TAR장.
출력: local/im_real_raw/<tar>/ (같은 폴더) + manifest_hard.csv (열: 기존 + hard=blur|noise|both, lapv, noise)
"""
import tarfile, random, csv, io, time, sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(r"G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터")
LOCAL = Path(__file__).resolve().parents[1] / "local"; OUT = LOCAL / "im_real_raw"
TARGET, PER_TAR, SCAN = 300, 12, 60

def measure(img):
    w, h = img.size; tw, th = (768, 1024) if h >= w else (1024, 768); s = max(tw / w, th / h)
    r = img.resize((max(tw, round(w * s)), max(th, round(h * s))), Image.BICUBIC)
    a = np.asarray(r.convert("RGB")); g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lapv = np.log(cv2.Laplacian(g, cv2.CV_32F).var() + 1)
    res = g - cv2.medianBlur(g.astype(np.uint8), 3); noise = np.median(np.abs(res - np.median(res))) * 1.4826
    red = (a[..., 0].astype(float) - a[..., 1]).mean() > 25
    return lapv, noise, red

def main():
    man = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8")))
    redset = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    have = {r["file"].replace("\\", "/") for r in man}
    base = [r for r in man if r["file"] not in redset]; rng = random.Random(7); rng.shuffle(base)
    M = np.array([measure(Image.open(OUT / r["file"]))[:2] for r in base[:600]])
    th_l, th_n = np.percentile(M[:, 0], 15), np.percentile(M[:, 1], 90)
    print(f"문턱: 선명도(log lapvar) < {th_l:.3f}, 잡음 MAD > {th_n:.3f}", flush=True)
    tars = sorted((ROOT / "2.Validation" / "원천데이터").glob("*.tar")) + sorted((ROOT / "1.Training" / "원천데이터_add_20260807").glob("*.tar"))
    rng.shuffle(tars); got = defaultdict(int)
    f = open(OUT / "manifest_hard.csv", "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["tar", "split", "file", "group", "farm_code", "size", "hard", "lapv", "noise"]); w.writeheader()
    fa = open(OUT / "manifest.csv", "a", newline="", encoding="utf-8"); wa = csv.DictWriter(fa, fieldnames=["tar", "split", "file", "group", "farm_code", "size"])
    for tp in tars:
        v = "금실" if "금실" in tp.stem else "설향"
        if got[v] >= TARGET: continue
        if all(got[x] >= TARGET for x in ("금실", "설향")): break
        t = time.time(); k = 0; scanned = 0
        try:
            tf = tarfile.open(tp, "r:"); ms = [m for m in tf if m.isfile() and m.name.lower().endswith(".jpg") and f"{tp.stem}/{Path(m.name).name}" not in have]
            random.Random(tp.stem).shuffle(ms)
            for m in ms[:SCAN]:
                scanned += 1; data = tf.extractfile(m).read(); img = Image.open(io.BytesIO(data))
                lapv, noise, red = measure(img)
                if red: continue
                hb, hn = lapv < th_l, noise > th_n
                if not (hb or hn): continue
                dst = OUT / tp.stem / Path(m.name).name; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
                stem = Path(m.name).stem
                row = dict(tar=tp.stem, split=tp.parent.parent.name, file=str(dst.relative_to(OUT)), group="_".join(stem.split("_")[:3]), farm_code=stem.split("_")[0], size=m.size)
                w.writerow(dict(row, hard="both" if hb and hn else ("blur" if hb else "noise"), lapv=f"{lapv:.3f}", noise=f"{noise:.3f}")); wa.writerow(row)
                k += 1; got[v] += 1
                if k >= PER_TAR or got[v] >= TARGET: break
            f.flush(); fa.flush()
            print(f"{tp.stem}: 훑음 {scanned}, 보관 {k} | 누적 금실 {got['금실']} 설향 {got['설향']} | {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            print("FAIL", tp.stem, e, flush=True)
    print("done", dict(got))

if __name__ == "__main__":
    main()
