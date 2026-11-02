# -*- coding: utf-8 -*-
"""나단11 저장 모델로 공개샘플 21장 추론 재현 (집 클로드 2026-10-17, CPU).  두 번(정순·역순) 돌려 순서 독립도 확인."""
import sys, os, time, csv, importlib.util
pkg, imgdir, labels, out = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("inference", os.path.join(pkg, "inference.py"))
inf = importlib.util.module_from_spec(spec); sys.path.insert(0, pkg); spec.loader.exec_module(inf)
lab = {}
with open(labels, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): lab[r[list(r)[0]]] = r
paths = sorted(os.path.join(imgdir, x) for x in os.listdir(imgdir))
t0 = time.time(); m = inf.load_model(os.path.join(pkg, "model")); t1 = time.time()
p1 = inf.predict(m, paths); t2 = time.time()
p2 = inf.predict(m, list(reversed(paths)))[::-1]
print("load %.1fs, predict 21 %.1fs (CPU)" % (t1 - t0, t2 - t1))
print("first label row:", next(iter(lab.values())))
rows = []
for p, a, b in zip(paths, p1, p2):
    rows.append((os.path.basename(p), float(a), float(b)))
print("max |forward - reversed| =", max(abs(a - b) for _, a, b in rows))
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "p", "p_reversed_order"]); w.writerows(rows)
print("wrote", out)
