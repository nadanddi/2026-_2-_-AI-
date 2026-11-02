"""IM48: 나단8 게이트 — 파일이 운영진 배포 형식(JPEG, 4:2:0, PIL Q92 양자화표)일 때만 H2 전문가 적용, 아니면 나단6 출력. 저장 특징(feats_h)으로 재계산. 서버(nadan).
결과를 본 뒤의 설계 변경(R3 처리 5종 결과 이후)이므로 탐색적. webp/png/4:4:4는 구조상 나단6과 같아짐."""
import sys, io, csv, json, numpy as np
from pathlib import Path
from PIL import Image, JpegImagePlugin
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im43_h_feats import paths_of  # noqa
b = io.BytesIO(); Image.new("RGB", (64, 64), (120, 130, 140)).save(b, "JPEG", quality=92); REFQ = Image.open(io.BytesIO(b.getvalue())).quantization
def std(p):
    try:
        im = Image.open(p); return im.format == "JPEG" and JpegImagePlugin.get_sampling(im) == 2 and im.quantization == REFQ
    except Exception: return False
L = H.L; z = np.load(L / "preds/h_model_h2.npz"); M = {}
for k, v in z.items(): a, c = k.split("__"); M.setdefault(a, {})[c] = v
th = json.load(open(L / "preds/h_theta_h2.json")); TH, THrc = th["theta"], th["theta_rc"]
for sn in ["pub21", "im_set_r3", "im_set_r2", "sealed_real", "sealed_brown", "sealed_bright", "sealed_v7", "sealed_qwen", "jig500"] + [f"im_set_r3p_{k}" for k in ("sharp", "png", "area", "webp", "c444")]:
    F = H.load(sn); ps = paths_of(sn); s = np.array([std(p) for p in ps]); p6 = F["p6"]; ph = H.apply(p6, H.score(M, F), TH, THrc, F["rc"]); pg = np.where(s, ph, p6)
    if sn.startswith("im_set_r3") or sn.startswith("im_set_r2") or sn == "pub21":
        y = (np.array([int(r["label"]) for r in csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8"))]) if sn != "pub21"
             else np.array([int(r["label"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))]))
    else: y = np.full(len(p6), 0 if sn in ("sealed_real", "sealed_brown", "sealed_bright", "jig500") else 1)
    acc = lambda p: .5 * (p[y == 0] < .5).mean() + .5 * (p[y == 1] >= .5).mean() if (y == 0).any() and (y == 1).any() else ((p >= .5) == (y == 1)).mean()
    print(f"[{sn:18s}] 표준형식 {s.mean():.3f} | 정확도 나단6 {acc(p6):.4f} 나단8 {acc(ph):.4f} 게이트 {acc(pg):.4f}", flush=True)
print("GATE_DONE")
