"""ABL2 이미지 '대회 규칙성 장치' 끄기(abl2_PLAN.md, 실행 전 고정, 서술용). 집 노트북 GPU.
나단11 제출 ZIP의 inference.py·models를 수정 없이 불러와, 사진마다 _batch_gpu·_safe_low를 한 번 계산하고
info만 바꿔 _final을 판별로 부른다: A 그대로 / D 지문 DB 끔 / E 형식 게이트 끔(std=False) / F D+E / G 게이트 늘 엶(std=True).
실행: PYTHONPATH="" python -u abl2_image_field_switches_v1.py <묶음 폴더 이름...>   (local/abl2/data/<묶음>/{img,manifest.csv})
출력: local/abl2/preds_abl2_<묶음>.csv, 표준출력 로그."""
import sys, os, csv, hashlib, importlib.util, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
W = HERE / "local" / "abl2"
PK = W / "pkg"
PUB = HERE.parents[2] / "공용" / "대회자료" / "이미지데이터" / "참가자_배포" / "공개샘플"
OLD = HERE.parents[2] / "제출" / "최종검증" / "팜모니_이미지" / "logs" / "ablation_원본대조_2026-10-20" / "preds_ablation.csv"

print("inference.py sha256", hashlib.sha256((PK / "inference.py").read_bytes()).hexdigest(), flush=True)
sys.path.insert(0, str(PK))
sp = importlib.util.spec_from_file_location("inference", str(PK / "inference.py"))
I = importlib.util.module_from_spec(sp); sp.loader.exec_module(I)
m = I.load_model(str(PK / "model"))
print("device", m["dev"], m["dt"], flush=True)
VAR = "ADEFG"


def variants(info):
    base = dict(info)
    nodb = {k: x for k, x in base.items() if k not in ("d", "ff")}
    return {"A": base, "D": nodb, "E": dict(base, std=False), "F": dict(nodb, std=False), "G": dict(base, std=True)}


def run(paths):
    order = sorted(range(len(paths)), key=lambda i: paths[i]); res = {v: [0.5] * len(paths) for v in VAR}
    std = [False] * len(paths)
    with ThreadPoolExecutor(8) as ex:
        lows = {i: ex.submit(I._safe_low, paths[i]) for i in order}
        for b in range(0, len(order), I.BATCH):
            idx = order[b:b + I.BATCH]; imgs = []
            for i in idx:
                try: imgs.append(Image.open(paths[i]).convert("RGB"))
                except Exception: imgs.append(None)
            infos = I._batch_gpu(m, imgs)
            for i, info in zip(idx, infos):
                info["std"] = I._std_format(paths[i]); std[i] = info["std"]; low = lows[i].result()
                for v, inf in variants(info).items():
                    res[v][i] = I._final(m, inf, low)
    return {v: np.array(x) for v, x in res.items()}, np.array(std)


# 재현 검사 1: 공개 21 판 A == 패키지 predict (같은 기계, 1e-6)
pr = list(csv.DictReader(open(PUB / "sample_labels.csv", encoding="utf-8")))
pp = [str(PUB / "images" / r["file"]) for r in pr]
a = run(pp)[0]["A"]; b = np.array(I.predict(m, pp)); md = float(np.abs(a - b).max())
print(f"[재현 검사 1] 공개21 판A vs predict 최대 차 {md:.2e}", flush=True)
if md > 1e-6:
    print("재현 실패 → 중단", flush=True); sys.exit(1)
old = {}
for r in csv.DictReader(open(OLD, encoding="utf-8")):
    old[(r["set"], r["file"])] = (float(r["pA"]), float(r["pD"]))

for sn in sys.argv[1:]:
    rr = list(csv.DictReader(open(W / "data" / sn / "manifest.csv", encoding="utf-8")))
    ps = [str(W / "data" / sn / "img" / r["file"]) for r in rr]
    t = time.time(); r, std = run(ps)
    print(f"{sn} {len(ps)}장 {time.time() - t:.0f}s, 배포 형식 비율 {std.mean():.4f}", flush=True)
    # 재현 검사 2: 10-20 서버 결과와 대조(있는 묶음만)
    ko = [(j, old[(sn, x['file'])]) for j, x in enumerate(rr) if (sn, x["file"]) in old]
    if ko:
        for v, k in (("A", 0), ("D", 1)):
            o = np.array([q[k] for _, q in ko]); n = r[v][[j for j, _ in ko]]
            dmax = float(np.abs(o - n).max()); agree = float(((o >= .5) == (n >= .5)).mean())
            ok = dmax <= 0.02 and agree >= 0.995
            print(f"[재현 검사 2] {sn} 판{v} vs 10-20: n {len(ko)}, 최대 차 {dmax:.4f}, 판정 일치 {agree:.4f} → {'통과' if ok else '실패'}", flush=True)
            if not ok:
                print("재현 실패 → 이 묶음 결과는 보고에 쓰지 않음(원인 기록)", flush=True)
    with open(W / f"preds_abl2_{sn}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["set", "file", "label", "kind", "std"] + [f"p{v}" for v in VAR])
        for j, x in enumerate(rr):
            w.writerow([sn, x["file"], x["label"], x["kind"], int(std[j])] + [f"{r[v][j]:.6f}" for v in VAR])
print("ABL2_DONE", flush=True)
