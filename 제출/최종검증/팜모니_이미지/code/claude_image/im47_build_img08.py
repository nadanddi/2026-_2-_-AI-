"""IM47: 이미지 제출 후보 08(팜모니_나단8) 조립 — 나단7 패키지 + 전문가 H2(im44 tag h2). 서버(nadan).
- 나단7 폴더 복사 → inference.py 교체(submit_img08/inference.py) → model/h_experts.npz(preds/h_model_h2.npz) → config.json에 "h"(θ·θ_rc) 추가
- 확인: 패키지 출력 = im44 계산(R2 앞 400 + 공개 21 + R3 진짜 처리 sharp 앞 100) 차 < 1e-3, 용량
사용: python im47_build_img08.py"""
import sys, json, shutil, subprocess, numpy as np, csv
from pathlib import Path
B = Path.home() / "storage/nadan/farmai_img"; L = B / "data/local"; SRC = B / "submit_img07/팜모니_나단7"; DST = B / "submit_img08/팜모니_나단8"
if DST.exists(): raise SystemExit("이미 있음: " + str(DST))
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__")); shutil.copy(B / "data/image/submit_img08_inference.py", DST / "inference.py")
shutil.copy(L / "preds/h_model_h2.npz", DST / "model/h_experts.npz")
cfg = json.loads((DST / "model/config.json").read_text(encoding="utf-8")); th = json.load(open(L / "preds/h_theta_h2.json"))
cfg["h"] = dict(theta=th["theta"], theta_rc=th["theta_rc"], note="IM_유형1to5_H2_v1.md, im44 tag h2"); (DST / "model/config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()); print("용량", size, "바이트", size / 1e9, flush=True); assert size < 1e9
sys.path.insert(0, str(B / "data/image")); import im44_h_train_eval as H  # noqa
M = {k: {a: v for a, v in np.load(DST / "model/h_experts.npz").items() if a.startswith(k + "__")} for k in ("E_f", "E_ps2", "E_g7")}
M = {k: {a.split("__")[1]: v for a, v in d.items()} for k, d in M.items()}; TH, THrc = th["theta"], th["theta_rc"]
def expect(sn, n):
    F = H.load(sn); F = {k: v[:n] for k, v in F.items()}; return H.apply(F["p6"], H.score(M, F), TH, THrc, F["rc"])
r2 = [str(L / "im_set_r2/img" / r["file"]) for r in csv.DictReader(open(L / "im_set_r2/manifest.csv", encoding="utf-8"))][:400]
P = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; pub = [str(P / "images" / r["file"]) for r in csv.DictReader(open(P / "sample_labels.csv"))]
sh = [str(L / "im_set_r3p_sharp/img" / r["file"]) for r in csv.DictReader(open(L / "im_set_r3p_sharp/manifest.csv", encoding="utf-8"))][:100]
for nm, ps, sn in (("R2 앞 400", r2, "im_set_r2"), ("공개 21", pub, "pub21"), ("R3 선명화 앞 100", sh, "im_set_r3p_sharp")):
    lst = L / "preds/_b8.json"; out = L / "preds/_b8.npy"; json.dump(ps, open(lst, "w"))
    code = "import sys,json,numpy as np; sys.path.insert(0, %r); import inference as I; m=I.load_model(%r); np.save(%r, np.array(I.predict(m, json.load(open(%r)))))" % (str(DST), str(DST / "model"), str(out), str(lst))
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True); got = np.load(out); exp = expect(sn, len(ps))
    d = np.abs(got - exp); print(f"[{nm}] 패키지 vs 계산 최대 차 {d.max():.2e}, 판정 다름 {int(((got >= .5) != (exp >= .5)).sum())}장", flush=True)
print("BUILD8_DONE", flush=True)
