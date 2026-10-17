"""IM01: AI허브 원천 tar에서 실제 사진을 고르게 뽑아 로컬로 복사 (원본 바이트 그대로).
- Validation(VS) 36개 tar 전부 × K_VS장, Training(TS) tar는 간격을 두고 N_TS개 × K_TS장.
- 같은 tar 안에서 개체(파일명 앞 C??_B??_???)가 고르게 나오도록 개체별 라운드로빈으로 뽑음.
- 출력: 연구실/클로드/local/im_real_raw/<tar이름>/<파일명>, manifest csv
"""
import tarfile, random, csv, sys, time, os
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터")
OUT = Path(__file__).resolve().parents[1] / "local" / "im_real_raw"
K_VS, K_TS, N_TS = 24, 12, 60
SEED = 20261010

def pick(tar_path, k, rng):
    t = time.time()
    tf = tarfile.open(tar_path, "r:")
    members = [m for m in tf if m.isfile() and m.name.lower().endswith(".jpg")]
    groups = defaultdict(list)
    for m in members:
        stem = Path(m.name).stem
        groups["_".join(stem.split("_")[:3])].append(m)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups); rng.shuffle(keys)
    chosen = []
    while len(chosen) < k and any(groups.values()):
        for key in keys:
            if groups[key] and len(chosen) < k:
                chosen.append(groups[key].pop())
    odir = OUT / tar_path.stem; odir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in chosen:
        dst = odir / Path(m.name).name
        if not dst.exists():
            dst.write_bytes(tf.extractfile(m).read())
        stem = Path(m.name).stem
        rows.append(dict(tar=tar_path.stem, split=tar_path.parent.parent.name, file=str(dst.relative_to(OUT)),
                         group="_".join(stem.split("_")[:3]), farm_code=stem.split("_")[0], size=m.size))
    print(f"{tar_path.stem}: {len(members)} jpg, {len(groups)} groups, took {len(chosen)} in {time.time()-t:.0f}s", flush=True)
    return rows

def main():
    rng = random.Random(SEED)
    vs = sorted((ROOT / "2.Validation" / "원천데이터").glob("*.tar"))
    ts_all = sorted((ROOT / "1.Training" / "원천데이터_add_20260807").glob("*.tar"))
    step = max(1, len(ts_all) // N_TS)
    ts = ts_all[::step][:N_TS]
    jobs = [(p, K_VS) for p in vs] + [(p, K_TS) for p in ts]
    man = OUT / "manifest.csv"; OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if man.exists():
        done = {r["tar"] for r in csv.DictReader(open(man, encoding="utf-8"))}
    with open(man, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tar", "split", "file", "group", "farm_code", "size"])
        if not done: w.writeheader()
        for p, k in jobs:
            if p.stem in done: continue
            try:
                for r in pick(p, k, rng): w.writerow(r)
                f.flush()
            except Exception as e:
                print("FAIL", p.stem, e, flush=True)

if __name__ == "__main__":
    main()
