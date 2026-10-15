"""팀 모델 비교 도구 — 여러 제출 ZIP(대회 규격: inference.py의 load_model/predict)을 같은 라벨 사진 묶음에 돌려 비교한다.
사용(Colab T4 권장):
  python compare_models.py --zips 팜모니_나단8.zip 내팀폴더.zip --images_dir <사진 폴더> --labels <labels.csv> [--base /content/drive/MyDrive/farmai_t4]
labels.csv 열: file,label[,type]   (label 0=진짜, 1=가짜; type은 있으면 유형별 정답률 출력)
출력: 모델별 정확도·진짜 오판률·가짜 탐지율·유형별 정답률·524장 환산 시간, 두 모델 판정이 갈린 사진 수와 승패.
각 모델은 별도 프로세스로 실행한다(모듈 이름 충돌·GPU 메모리 방지). 네트워크 접근 없음.
"""
import argparse, csv, json, subprocess, sys, time, zipfile, shutil, collections
from pathlib import Path

WORKER = r'''
import sys, json, time, importlib
team, model_dir, lst, out = sys.argv[1:5]
sys.path.insert(0, team); I = importlib.import_module("inference")
paths = json.load(open(lst)); t0 = time.time(); m = I.load_model(model_dir); t1 = time.time(); p = I.predict(m, paths); t2 = time.time()
json.dump(dict(p=[float(x) for x in p], load=t1 - t0, pred=t2 - t1), open(out, "w"))
'''

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zips", nargs="+", required=True); ap.add_argument("--images_dir", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--base", default="/content/drive/MyDrive/farmai_t4"); ap.add_argument("--work", default="/content/cmpwork"); a = ap.parse_args()
    W = Path(a.work); shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)
    rows = list(csv.DictReader(open(a.labels, encoding="utf-8-sig"))); paths = [str(Path(a.images_dir) / r["file"]) for r in rows]
    y = [int(r["label"]) for r in rows]; ty = [r.get("type", "") or ("real" if int(r["label"]) == 0 else "fake") for r in rows]
    json.dump(paths, open(W / "paths.json", "w")); (W / "worker.py").write_text(WORKER, encoding="utf-8"); res = {}
    for z in a.zips:
        zp = Path(z) if Path(z).exists() else Path(a.base) / z; d = W / zp.stem; zipfile.ZipFile(zp).extractall(d)
        team = next(p for p in [d, *d.rglob("*")] if p.is_dir() and (p / "inference.py").exists())
        out = W / f"{zp.stem}.json"; t = time.time()
        subprocess.run([sys.executable, "-W", "ignore", str(W / "worker.py"), str(team), str(team / "model"), str(W / "paths.json"), str(out)], check=True)
        r = json.load(open(out)); res[zp.stem] = r
        p = r["p"]; n = len(p); hit = [(pi >= .5) == (yi == 1) for pi, yi in zip(p, y)]
        real = [h for h, yi in zip(hit, y) if yi == 0]; fake = [h for h, yi in zip(hit, y) if yi == 1]
        bal = .5 * (sum(real) / max(1, len(real))) + .5 * (sum(fake) / max(1, len(fake)))
        print(f"\n== {zp.stem} == 정확도 {sum(hit) / n:.4f} | 균형 {bal:.4f} | 진짜 오판 {1 - sum(real) / max(1, len(real)):.4f} | 가짜 탐지 {sum(fake) / max(1, len(fake)):.4f}")
        print(f"   모델 로드 {r['load']:.1f}s + 예측 {r['pred']:.1f}s → 524장 환산 {r['load'] + r['pred'] / n * 524:.1f}s (한도 300s)")
        by = collections.defaultdict(list)
        for h, t_ in zip(hit, ty): by[t_].append(h)
        for t_ in sorted(by): print(f"   {t_:20s} n={len(by[t_]):4d} 정답률 {sum(by[t_]) / len(by[t_]):.3f}")
    names = list(res)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, Bm = res[names[i]]["p"], res[names[j]]["p"]; dif = [k for k in range(len(y)) if (A[k] >= .5) != (Bm[k] >= .5)]
            wa = sum((A[k] >= .5) == (y[k] == 1) for k in dif); print(f"\n[{names[i]} vs {names[j]}] 판정이 갈린 사진 {len(dif)}장: {names[i]} 정답 {wa} / {names[j]} 정답 {len(dif) - wa}")
            byt = collections.Counter(ty[k] for k in dif if (A[k] >= .5) == (y[k] == 1)); byt2 = collections.Counter(ty[k] for k in dif if (Bm[k] >= .5) == (y[k] == 1))
            print(f"   {names[i]}가 이긴 유형 {dict(byt)}\n   {names[j]}가 이긴 유형 {dict(byt2)}")

if __name__ == "__main__":
    main()
