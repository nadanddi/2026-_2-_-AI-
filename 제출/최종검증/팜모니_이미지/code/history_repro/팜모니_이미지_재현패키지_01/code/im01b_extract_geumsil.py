"""IM01b: 금실 추가 추출 → 금실:설향(빨간 조명 제외) = 5:5 (사용자 지시 10-10).
- 목표 금실 수 = 현재 설향(빨간 조명 제외) 수. 부족분을 금실 tar 전부(VS·TS)에 고르게 나눠, 이미 뽑은 파일은 건너뛰고 개체 라운드로빈으로 뽑음.
- 같은 manifest(im_real_raw/manifest.csv)에 이어 씀(열 동일). 이번 추가분은 manifest_geumsil_add.csv에도 따로 기록.
"""
import tarfile, random, csv, time, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터")
LOCAL = Path(__file__).resolve().parents[1] / "local"; OUT = LOCAL / "im_real_raw"
FIELDS = ["tar", "split", "file", "group", "farm_code", "size"]

def main():
    man = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8")))
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    have = {r["file"] for r in man}
    n_sy = sum(1 for r in man if "설향" in r["tar"] and r["file"] not in red)
    n_gs = sum(1 for r in man if "금실" in r["tar"])
    need = n_sy - n_gs
    tars = sorted((ROOT / "2.Validation" / "원천데이터").glob("*금실*.tar")) + sorted((ROOT / "1.Training" / "원천데이터_add_20260807").glob("*금실*.tar"))
    print(f"설향(빨강 제외) {n_sy}, 금실 {n_gs} → 추가 {need}장, 금실 tar {len(tars)}개", flush=True)
    per = math.ceil(need / len(tars)) + 1
    fa = open(OUT / "manifest.csv", "a", newline="", encoding="utf-8"); wa = csv.DictWriter(fa, fieldnames=FIELDS)
    fb = open(OUT / "manifest_geumsil_add.csv", "w", newline="", encoding="utf-8"); wb = csv.DictWriter(fb, fieldnames=FIELDS); wb.writeheader()
    got = 0
    for tp in tars:
        if got >= need: break
        t = time.time(); rng = random.Random(f"gs2|{tp.stem}")
        try:
            tf = tarfile.open(tp, "r:")
            groups = defaultdict(list)
            for m in tf:
                if m.isfile() and m.name.lower().endswith(".jpg"):
                    rel = f"{tp.stem}/{Path(m.name).name}"
                    if rel.replace("/", "\\") in have or rel in have: continue
                    groups["_".join(Path(m.name).stem.split("_")[:3])].append(m)
            for g in groups.values(): rng.shuffle(g)
            keys = sorted(groups); rng.shuffle(keys); chosen = []
            k = min(per, need - got)
            while len(chosen) < k and any(groups.values()):
                for key in keys:
                    if groups[key] and len(chosen) < k: chosen.append(groups[key].pop())
            odir = OUT / tp.stem; odir.mkdir(parents=True, exist_ok=True)
            for m in chosen:
                dst = odir / Path(m.name).name
                if not dst.exists(): dst.write_bytes(tf.extractfile(m).read())
                stem = Path(m.name).stem
                row = dict(tar=tp.stem, split=tp.parent.parent.name, file=str(dst.relative_to(OUT)), group="_".join(stem.split("_")[:3]),
                           farm_code=stem.split("_")[0], size=m.size)
                wa.writerow(row); wb.writerow(row)
            fa.flush(); fb.flush(); got += len(chosen)
            print(f"{tp.stem}: +{len(chosen)} (누적 {got}/{need}) {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            print("FAIL", tp.stem, e, flush=True)
    print("done", got)

if __name__ == "__main__":
    main()
