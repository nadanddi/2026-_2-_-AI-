"""IM30: 새 AI Hub 사진 추출(IM_필요이미지_정리_v1.md 0·3번). 로컬(구글 드라이브 tar) 실행.
- 후보: 지문 DB(fp_db, tar별 사진 이름·12x16 색)에서 '아직 안 쓴 파일'만(촬영 묶음은 이미 거의 다 쓰여 묶음 제외는 불가 → 새 역할끼리만 묶음 겹침 없음; 봉인 진짜도 학습 진짜와 같은 포기를 공유할 수 있어 약간 낙관적). 빨간 조명(지문 평균 R−G>25) 제외.
- tar 읽기 시간을 줄이려 tar N_TAR개만 고름(이름 앞부분(품종)·TS/VS 고르게).
- 역할(서로 다른 촬영 묶음, 겹침 없음)
    gen 1000(생성·편집 원본) / sealed 400(봉인 진짜) / train 600(추가 학습 진짜) / brown 200(갈변 후보 상위) / bright 150(밝음 후보 상위)
  갈변 점수 = 지문 칸 중 (R>G, G−B>8, 밝기 50~200) 비율, 밝음 점수 = 칸 밝기>232 비율.
출력: local/im_new_raw/<tar>/<file>, local/im_new_raw/manifest.csv (role, tar, file, group, farm_code, brown, bright)
"""
import csv, glob, random, tarfile, time
import numpy as np
from pathlib import Path
from collections import defaultdict
ROOT = Path(r"G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터")
LOCAL = Path(__file__).resolve().parents[1] / "local"; OUT = LOCAL / "im_new_raw"
N_TAR = 48; ROLES = [("gen", 1000), ("sealed", 400), ("train", 600)]; SPECIAL = [("brown", 200), ("bright", 150)]

def used_sets():
    used = set()
    for m in glob.glob(str(LOCAL / "im_set_*" / "manifest*.csv")) + [str(LOCAL / "im_real_raw/manifest.csv")]:
        for r in csv.DictReader(open(m, encoding="utf-8")):
            for c in ("source", "other", "file"):
                v = (r.get(c) or "").replace("/", "\\")
                if v.endswith(".jpg"): used.add(Path(v.replace("\\", "/")).name)
    for lst in ("원본_목록.csv", "원본2_목록.csv", "원본3_목록.csv"):
        used |= {Path(r["AI허브_원천"].replace("\\", "/")).name for r in csv.DictReader(open(LOCAL / "ai_gen_work" / lst, encoding="utf-8-sig"))}
    groups = {"_".join(Path(u).stem.split("_")[:3]) for u in used}
    return used, groups

def main():
    rng = random.Random(20261013); used, ugroups = used_sets()
    cands = defaultdict(list)
    for p in sorted((LOCAL / "fp_db").glob("*.npz")):
        z = np.load(p); tar = p.stem
        for nm, fp in zip(z["names"], z["fp"].astype(np.float32)):
            nm = Path(str(nm)).name; nm = nm if nm.lower().endswith(".jpg") else nm + ".jpg"; g = "_".join(Path(nm).stem.split("_")[:3])
            if nm in used: continue
            R, G, B = fp[..., 0], fp[..., 1], fp[..., 2]
            if (R - G).mean() > 25: continue
            lum = (R + G + B) / 3
            cands[tar].append(dict(file=nm, group=g, farm_code=nm.split("_")[0], brown=float(((R > G) & (G - B > 8) & (lum > 50) & (lum < 200)).mean()), bright=float((lum > 232).mean())))
    tars = sorted(cands); key = lambda t: (t.split(".")[0], "".join(c for c in t.split(".")[1] if not c.isdigit()))
    byk = defaultdict(list)
    for t in tars: byk[key(t)].append(t)
    for v in byk.values(): rng.shuffle(v)
    pick = []
    while len(pick) < N_TAR and any(byk.values()):
        for k in sorted(byk):
            if byk[k] and len(pick) < N_TAR: pick.append(byk[k].pop())
    pool = [dict(c, tar=t) for t in pick for c in cands[t]]
    print("후보 tar", len(tars), "선택", len(pick), "후보 사진", len(pool), "촬영묶음", len({c['group'] for c in pool}), flush=True)
    bygrp = defaultdict(list)
    for c in pool: bygrp[c["group"]].append(c)
    for v in bygrp.values(): rng.shuffle(v)
    # 촬영 묶음 단위로 봉인/학습쪽을 나눔(농장 고르게, 봉인 약 25%) → 봉인 진짜는 학습쪽과 같은 묶음을 공유하지 않음
    groups = list(bygrp); rng.shuffle(groups); byfarm = defaultdict(list)
    for g in groups: byfarm[g.split("_")[0]].append(g)
    order = []
    while any(byfarm.values()):
        for f in sorted(byfarm):
            if byfarm[f]: order.append(byfarm[f].pop())
    sealed_g = set(order[::4]); train_g = [g for g in order if g not in sealed_g]
    taken, chosen = set(), []
    def draw(gs, n, role):   # 묶음을 돌아가며 한 장씩
        got, ptr = 0, defaultdict(int)
        while got < n:
            moved = False
            for g in gs:
                lst = bygrp[g]
                while ptr[g] < len(lst) and lst[ptr[g]]["file"] in taken: ptr[g] += 1
                if ptr[g] < len(lst) and got < n:
                    c = lst[ptr[g]]; chosen.append(dict(c, role=role)); taken.add(c["file"]); got += 1; moved = True
            if not moved: break
        print(role, got, "묶음", len(gs), flush=True)
    draw(sorted(sealed_g), dict(ROLES)["sealed"], "sealed")
    for role, n in SPECIAL:   # 특수 후보: 학습쪽 묶음에서 점수 상위(묶음당 최대 10장)
        cnt = defaultdict(int); got = 0
        for c in sorted((c for c in pool if c["group"] not in sealed_g), key=lambda c: -c[role]):
            if got >= n: break
            if c["file"] in taken or cnt[c["group"]] >= 10: continue
            chosen.append(dict(c, role=role)); taken.add(c["file"]); cnt[c["group"]] += 1; got += 1
        print(role, got, flush=True)
    draw(train_g, dict(ROLES)["gen"], "gen"); draw(train_g, dict(ROLES)["train"], "train")
    need = defaultdict(dict)
    for c in chosen: need[c["tar"]][c["file"]] = c
    OUT.mkdir(exist_ok=True); fo = open(OUT / "manifest.csv", "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fo, fieldnames=["role", "tar", "file", "group", "farm_code", "brown", "bright"]); w.writeheader(); t0 = time.time(); n_ok = 0
    for k, (tar, want) in enumerate(sorted(need.items())):
        sp = "2.Validation/원천데이터" if tar.startswith("VS_") else "1.Training/원천데이터_add_20260807"
        tp = ROOT / sp / f"{tar}.tar"; (OUT / tar).mkdir(exist_ok=True)
        try:
            with tarfile.open(tp, "r|") as tf:   # 순차 스트림(드라이브 건너뛰기 읽기가 매우 느림, 순차 6.7MB/s)
                for mem in tf:
                    nm = Path(mem.name).name
                    if nm in want and mem.isfile():
                        (OUT / tar / nm).write_bytes(tf.extractfile(mem).read()); c = want.pop(nm)
                        w.writerow({x: c[x] for x in ("role", "tar", "file", "group", "farm_code", "brown", "bright")}); n_ok += 1
                    if not want: break
        except Exception as e:
            print("FAIL", tar, repr(e)[:120], flush=True)
        fo.flush(); print(f"[{k + 1}/{len(need)}] {tar} 누적 {n_ok}장 {time.time() - t0:.0f}s 남은 {len(want)}", flush=True)
    print("EXTRACT_DONE", n_ok)

if __name__ == "__main__":
    main()
