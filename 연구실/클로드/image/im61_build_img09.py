"""IM61: 이미지 제출 후보 09(팜모니_나단9) 조립 — 나단8 패키지 + E_gr(ip_model_v1.npz, θ ip_theta_v1.json = 짝 음성 판 .0311). 서버(nadan).
확인: 패키지 출력 = 계산(게이트 포함) — 공개 21, gpt3 200, R2 앞 400, GPT 재그리기 확인분 78."""
import sys, io, csv, json, shutil, subprocess, numpy as np
from pathlib import Path
from PIL import Image, JpegImagePlugin
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
B = Path.home() / "storage/nadan/farmai_img"; L = B / "data/local"; SRC = B / "submit_img08/팜모니_나단8"; DST = B / "submit_img09/팜모니_나단9"
if DST.exists(): raise SystemExit("이미 있음")
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__")); shutil.copy(B / "data/image/submit_img09_inference.py", DST / "inference.py"); shutil.copy(B / "data/image/submit_img09_README.md", DST / "README.md")
shutil.copy(L / "preds/ip_model_v1.npz", DST / "model/gr_expert.npz")
T9 = json.load(open(L / "preds/ip_theta_v1.json")); assert abs(T9["theta"]["E_gr"] - 0.031073) < 1e-5
cfg = json.loads((DST / "model/config.json").read_text(encoding="utf-8")); cfg["h"] = dict(theta=T9["theta"], theta_rc=T9["theta_rc"], note="나단8 H2 + E_gr(IM_GPT재그리기_Ip_v1.md, 짝 음성 판)")
(DST / "model/config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()); print("용량", size, size / 1e9, flush=True); assert size < 1e9
sys.path.insert(0, str(B / "data/image")); import im44_h_train_eval as H  # noqa
b = io.BytesIO(); Image.new("RGB", (64, 64), (120, 130, 140)).save(b, "JPEG", quality=92); REFQ = Image.open(io.BytesIO(b.getvalue())).quantization
def std(p):
    try: im = Image.open(p); return im.format == "JPEG" and JpegImagePlugin.get_sampling(im) == 2 and im.quantization == REFQ
    except Exception: return False
M8 = {}
for k, v in np.load(L / "preds/h_model_h2.npz").items(): a, c = k.split("__"); M8.setdefault(a, {})[c] = v
MG = {a.split("__")[1]: v for a, v in np.load(DST / "model/gr_expert.npz").items()}
def expect(sn, ps, n):
    G = H.load(sn); G = {k: v[:n] for k, v in G.items()}; s = np.array([std(p) for p in ps])
    sc = {**H.score(M8, G), "E_gr": H.lrp(MG, np.c_[G["low"], G["emb"]])}; return np.where(s, H.apply(G["p6"], sc, T9["theta"], T9["theta_rc"], G["rc"]), G["p6"])
man = lambda sn: [str(L / sn / "img" / r["file"]) for r in csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8"))]
P = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; pub = [str(P / "images" / r["file"]) for r in csv.DictReader(open(P / "sample_labels.csv"))]
gptr = man("im_set_gptr")
for nm, ps, sn in (("공개 21", pub, "pub21"), ("gpt3 200", man("im_set_gpt3"), "im_set_gpt3"), ("R2 앞 400", man("im_set_r2")[:400], "im_set_r2"), ("GPT 재그리기 150", gptr, "im_set_gptr")):
    lst = L / "preds/_b9.json"; out = L / "preds/_b9.npy"; json.dump(ps, open(lst, "w"))
    code = "import sys,json,numpy as np; sys.path.insert(0, %r); import inference as I; m=I.load_model(%r); np.save(%r, np.array(I.predict(m, json.load(open(%r)))))" % (str(DST), str(DST / "model"), str(out), str(lst))
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True); got = np.load(out); exp = expect(sn, ps, len(ps))
    print(f"[{nm}] 패키지 vs 계산 최대 차 {np.abs(got - exp).max():.2e}, 판정 다름 {int(((got >= .5) != (exp >= .5)).sum())}장, 패키지 가짜 판정 비율 {(got >= .5).mean():.3f}", flush=True)
print("BUILD9_DONE", flush=True)
