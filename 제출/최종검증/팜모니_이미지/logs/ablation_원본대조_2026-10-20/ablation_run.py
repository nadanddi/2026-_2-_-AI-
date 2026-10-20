"""원본 대조 끄기 실험(PLAN.md, 실행 전 고정). 서버(nadan, GPU0).
나단11 패키지 inference.py를 수정 없이 불러와, 사진마다 _batch_gpu·_safe_low를 한 번 계산하고
info에서 키만 빼서 _final을 4번 부른다: A 그대로 / B "d" 제거(R1 끔) / C "ff" 제거(정밀 대조 끔) / D 둘 다.
출력: preds_ablation.csv(묶음, 파일, 라벨, kind, pA~pD), 표준출력 로그."""
import sys, csv, hashlib, importlib.util, time, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
PK = Path.home() / "storage/nadan/farmai_img/submit_img11/팜모니_나단11"
L = Path.home() / "storage/nadan/farmai_img/data/local"
PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/ablation_out")
OUT.mkdir(parents=True, exist_ok=True)
print("inference.py sha256", hashlib.sha256((PK / "inference.py").read_bytes()).hexdigest(), flush=True)
sys.path.insert(0, str(PK)); sp = importlib.util.spec_from_file_location("inference", str(PK / "inference.py")); I = importlib.util.module_from_spec(sp); sp.loader.exec_module(I)
m = I.load_model(str(PK / "model"))
VAR = {"A": (), "B": ("d",), "C": ("ff",), "D": ("d", "ff")}

def run(paths):
    order = sorted(range(len(paths)), key=lambda i: paths[i]); res = {v: [0.5] * len(paths) for v in VAR}
    with ThreadPoolExecutor(8) as ex:
        lows = {i: ex.submit(I._safe_low, paths[i]) for i in order}
        for b in range(0, len(order), I.BATCH):
            idx = order[b:b + I.BATCH]; imgs = []
            for i in idx:
                try: imgs.append(Image.open(paths[i]).convert("RGB"))
                except Exception: imgs.append(None)
            infos = I._batch_gpu(m, imgs)
            for i, info in zip(idx, infos):
                info["std"] = I._std_format(paths[i]); low = lows[i].result()
                for v, drop in VAR.items():
                    res[v][i] = I._final(m, {k: x for k, x in info.items() if k not in drop}, low)
    return {v: np.array(x) for v, x in res.items()}

def sets():
    pr = list(csv.DictReader(open(PUB / "sample_labels.csv")))
    yield "pub21", [str(PUB / "images" / r["file"]) for r in pr], [r["file"] for r in pr], [int(r["label"]) for r in pr], [r["type"] for r in pr]
    for sn in ("im_set_r2", "im_set_r3", "im_set_r4", "im_set_r5", "im_set_realg", "im_set_realn",
               "im_set_r3p_sharp", "im_set_r3p_area", "im_set_r3p_png", "im_set_r3p_webp", "im_set_r3p_c444",
               "im_set_gpt1", "im_set_gpt2", "im_set_gpt3", "im_set_gptr"):
        rr = list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")))
        yield sn, [str(L / sn / "img" / r["file"]) for r in rr], [r["file"] for r in rr], [int(r["label"]) for r in rr], [r["kind"] for r in rr]

# 재현 검사: 판 A == 패키지 predict (공개 21)
s0 = next(sets()); a = run(s0[1])["A"]; b = np.array(I.predict(m, s0[1])); md = float(np.abs(a - b).max())
print(f"[재현 검사] 공개21 판A vs predict 최대 차 {md:.2e}", flush=True)
if md > 1e-6: print("재현 실패 → 중단", flush=True); sys.exit(1)
with open(OUT / "preds_ablation.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["set", "file", "label", "kind", "pA", "pB", "pC", "pD"])
    for sn, ps, fs, ys, ks in sets():
        t = time.time(); r = run(ps)
        for j in range(len(ps)): w.writerow([sn, fs[j], ys[j], ks[j]] + [f"{r[v][j]:.6f}" for v in "ABCD"])
        f.flush(); print(f"{sn} {len(ps)}장 {time.time() - t:.0f}s", flush=True)
print("ABLATION_DONE", flush=True)
