"""IM60: R2(3,724)에서 나단8+게이트 vs 나단9(= + E_gr, ip_model_v1·ip_theta_v1) 비교. 서버(nadan). 탐색용(R2는 여러 번 본 자료)."""
import sys, io, csv, json, numpy as np
from pathlib import Path
from PIL import Image, JpegImagePlugin
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im43_h_feats import paths_of  # noqa
L = H.L
b = io.BytesIO(); Image.new("RGB", (64, 64), (120, 130, 140)).save(b, "JPEG", quality=92); REFQ = Image.open(io.BytesIO(b.getvalue())).quantization
def std(p):
    try: im = Image.open(p); return im.format == "JPEG" and JpegImagePlugin.get_sampling(im) == 2 and im.quantization == REFQ
    except Exception: return False
M8 = {}
for k, v in np.load(L / "preds/h_model_h2.npz").items(): a, c = k.split("__"); M8.setdefault(a, {})[c] = v
T8 = json.load(open(L / "preds/h_theta_h2.json")); T9 = json.load(open(L / "preds/ip_theta_v1.json")); assert abs(T9["theta"]["E_gr"] - .0311) < 1e-3
MG = {a.split("__")[1]: v for a, v in np.load(L / "preds/ip_model_v1.npz").items()}
G = H.load("im_set_r2"); rr = list(csv.DictReader(open(L / "im_set_r2/manifest.csv", encoding="utf-8"))); y = np.array([int(r["label"]) for r in rr]); k = np.array([r["kind"] for r in rr])
s = np.array([std(p) for p in paths_of("im_set_r2")]); s8 = H.score(M8, G); sg = H.lrp(MG, np.c_[G["low"], G["emb"]])
p8 = np.where(s, H.apply(G["p6"], s8, T8["theta"], T8["theta_rc"], G["rc"]), G["p6"]); p9 = np.where(s, H.apply(G["p6"], {**s8, "E_gr": sg}, T9["theta"], T9["theta_rc"], G["rc"]), G["p6"])
acc = lambda p: .5 * (p[y == 0] < .5).mean() + .5 * (p[y == 1] >= .5).mean()
print(f"[R2 n={len(y)}] 표준형식 {s.mean():.3f} | 균형 정확도 나단8 {acc(p8):.4f} → 나단9 {acc(p9):.4f}")
print(f"  진짜 오판 {(p8[y == 0] >= .5).mean():.4f} → {(p9[y == 0] >= .5).mean():.4f} (새 오판 {int(((p8 < .5) & (p9 >= .5) & (y == 0)).sum())}장) | 가짜 탐지 {(p8[y == 1] >= .5).mean():.4f} → {(p9[y == 1] >= .5).mean():.4f} (+{int(((p8 < .5) & (p9 >= .5) & (y == 1)).sum())}장)")
for kk in sorted(set(k)):
    m = k == kk; print(f"  {kk:14s} n={m.sum():4d} 정답률 나단8 {((p8[m] >= .5) == (y[m] == 1)).mean():.3f} → 나단9 {((p9[m] >= .5) == (y[m] == 1)).mean():.3f}")
print("R2N9_DONE")
