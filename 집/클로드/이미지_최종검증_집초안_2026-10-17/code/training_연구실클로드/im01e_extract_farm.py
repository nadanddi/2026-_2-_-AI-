"""IM01e: 후보 R2 — 농장 균형 + 원본 해상도(축소 배율) 고른 실제 사진 추출 (사용자 결정 10-11: '농장 균형').
근거: 팀원 v5 분석(학습에 적은 농장·많이 축소된 사진에서 실제 오판), 우리 자료 확인(배율<.4 오탐 7.3% vs ≥.6 3.7%),
      우리 실제의 AIF005 쏠림(63%, 금실=AIF005 단일 농장)과 AIF006·007·012 결손.
선택: 집 코덱스 라벨 메타데이터(Training 135,189 + Validation 16,854; farm_id·width·height·archive·member).
 - 농장 9곳 × 목표 PER_FARM장, 농장 안에서 배율 구간 [<.4, .4-.5, .5-.6, ≥.6] 균등(없는 구간은 다른 구간으로), 개체(prefix) 라운드로빈.
 - 농장마다 무작위 tar N_TAR개에서만 고름(tar 색인 시간 절약). 이미 뽑은 사진은 건너뜀. 빨간 조명(평균 R-G>25)은 버리고 같은 tar에서 대체.
출력: local/im_real_raw/<tar>/ + manifest_r2.csv (열: 기존 + farm_id, width, height, scale, bin)
"""
import csv, io, random, tarfile, time, sys
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

ROOT = Path(r"G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터")
REPO = Path(__file__).resolve().parents[3]; LOCAL = Path(__file__).resolve().parents[1] / "local"; OUT = LOCAL / "im_real_raw"
META_T = REPO / "집/코덱스/analysis/image_ingestion_20261010_v1/metadata_all_v1.csv"
META_V = REPO / "집/코덱스/analysis/image_download_complete_20261010_v1/validation_metadata_v1.csv"
PER_FARM, N_TAR = 260, 8; BINS = [(0, .4), (.4, .5), (.5, .6), (.6, 9)]

def load_meta():
    rows = []
    for fn, sp in [(META_T, "1.Training"), (META_V, "2.Validation")]:
        for r in csv.DictReader(open(fn, encoding="utf-8-sig")):
            stem = Path(r["member"]).stem   # 라벨 member 이름 = 사진 파일 이름(확장자 제외)
            src_tar = r["archive"].replace("TL_", "TS_").replace("VL_", "VS_").replace(".tar", "")
            sc = 768 / min(int(r["width"]), int(r["height"]))
            rows.append(dict(farm=r["farm_id"], tar=src_tar, split=sp, stem=stem, w=int(r["width"]), h=int(r["height"]), scale=sc,
                             prefix=r.get("prefix_candidate") or "_".join(stem.split("_")[:3])))
    return rows

def main():
    rng = random.Random(20261011)
    meta = load_meta()
    have = {Path(r["file"]).stem for r in csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8"))}
    by_farm_tar = defaultdict(lambda: defaultdict(list))
    for m in meta:
        if m["stem"] not in have: by_farm_tar[m["farm"]][m["tar"]].append(m)
    fo = open(OUT / "manifest_r2.csv", "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fo, fieldnames=["tar", "split", "file", "group", "farm_code", "size", "farm_id", "width", "height", "scale", "bin"]); w.writeheader()
    fa = open(OUT / "manifest.csv", "a", newline="", encoding="utf-8"); wa = csv.DictWriter(fa, fieldnames=["tar", "split", "file", "group", "farm_code", "size"])
    for farm in sorted(by_farm_tar):
        tars = sorted(by_farm_tar[farm]); rng.shuffle(tars); tars = tars[:N_TAR]
        pool = [m for t in tars for m in by_farm_tar[farm][t]]
        # 배율 구간별 목표: 균등, 부족분은 다른 구간으로
        bins = [[m for m in pool if lo <= m["scale"] < hi] for lo, hi in BINS]
        quota = [PER_FARM // 4] * 4; avail = [len(b) for b in bins]
        for _ in range(4):
            short = sum(max(0, q - a) for q, a in zip(quota, avail)); quota = [min(q, a) for q, a in zip(quota, avail)]
            room = [i for i in range(4) if avail[i] > quota[i]]
            for i in room[:]:
                add = min(short // max(1, len(room)), avail[i] - quota[i]); quota[i] += add
        chosen = []
        for bi, b in enumerate(bins):
            g = defaultdict(list)
            for m in b: g[m["prefix"]].append(m)
            for v in g.values(): rng.shuffle(v)
            keys = sorted(g); rng.shuffle(keys); pick = []
            while len(pick) < quota[bi] * 2 and any(g.values()):   # 빨간 조명 대체분까지 2배 후보
                for k in keys:
                    if g[k] and len(pick) < quota[bi] * 2: pick.append(dict(g[k].pop(), bin=bi))
            chosen.append(pick)
        if len(sys.argv) > 1 and sys.argv[1] == "dry":
            print(f"{farm}: tar {len(tars)}개 후보 {len(pool)}장, 구간 가용 {avail} → 목표 {quota} (합 {sum(quota)})"); continue
        need = {bi: quota[bi] for bi in range(4)}; got = defaultdict(int); t0 = time.time()
        by_tar = defaultdict(list)
        for pick in chosen:
            for m in pick: by_tar[m["tar"]].append(m)
        for tname, ms in by_tar.items():
            tp = ROOT / ms[0]["split"] / ("원천데이터_add_20260807" if ms[0]["split"] == "1.Training" else "원천데이터") / f"{tname}.tar"
            if not tp.exists(): print("없음", tp, flush=True); continue
            want = {m["stem"]: m for m in ms}
            try:
                tf = tarfile.open(tp, "r:")
                for mem in tf:
                    st = Path(mem.name).stem
                    if st not in want: continue
                    m = want[st]
                    if got[m["bin"]] >= need[m["bin"]]: continue
                    data = tf.extractfile(mem).read(); a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB").reduce(16), np.float32)
                    if (a[..., 0] - a[..., 1]).mean() > 25: continue
                    dst = OUT / tname / Path(mem.name).name; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
                    row = dict(tar=tname, split=m["split"], file=str(dst.relative_to(OUT)), group="_".join(st.split("_")[:3]), farm_code=st.split("_")[0], size=mem.size)
                    wa.writerow(row); w.writerow(dict(row, farm_id=farm, width=m["w"], height=m["h"], scale=f"{m['scale']:.3f}", bin=m["bin"])); got[m["bin"]] += 1
                fo.flush(); fa.flush()
            except Exception as e:
                print("FAIL", tname, e, flush=True)
        print(f"{farm}: tar {len(by_tar)}개, 구간별 목표 {quota} → 확보 {[got[i] for i in range(4)]} | {time.time()-t0:.0f}s", flush=True)
    print("done")

if __name__ == "__main__":
    main()
