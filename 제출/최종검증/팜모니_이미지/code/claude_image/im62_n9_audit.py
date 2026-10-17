"""IM62: 나단9 점검(탐색) — ① 세트별 남은 오답 목록·특성 ② 부분 스타일 조각 색 통계 가능성(h3 학습 자료 묶음 5폴드 OOF만, 봉인 판정 아님). 서버(nadan).
색 통계: 54조각(224, 간격 112)마다 Lab 평균(L,a,b)·채도 → 사진 중앙값과의 차(조명 차이 상쇄) + 이웃 조각 평균과의 차 = 8개."""
import sys, io, csv, json, numpy as np, cv2
from pathlib import Path
from PIL import Image, JpegImagePlugin
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
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
MG = {a.split("__")[1]: v for a, v in np.load(L / "preds/ip_model_v1.npz").items()}; T9 = json.load(open(L / "preds/ip_theta_v1.json"))
def p9(sn):
    G = H.load(sn); s = np.array([std(p) for p in paths_of(sn)]); sc = {**H.score(M8, G), "E_gr": H.lrp(MG, np.c_[G["low"], G["emb"]])}
    return G, np.where(s, H.apply(G["p6"], sc, T9["theta"], T9["theta_rc"], G["rc"]), G["p6"]), sc
def man(sn): return list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")))
# ① 남은 오답
print("== ① 나단9 남은 오답 ==", flush=True)
for sn in ("im_set_r2", "im_set_r3", "im_set_r4"):
    G, p, sc = p9(sn); rr = man(sn); y = np.array([int(r["label"]) for r in rr]); k = np.array([r["kind"] for r in rr]); wrong = (p >= .5) != (y == 1)
    print(f"[{sn}] 오답 {wrong.sum()}/{len(y)} | 진짜 오판 {(wrong & (y == 0)).sum()}(rc {int((wrong & (y == 0) & G['rc']).sum())}) · 가짜 놓침 {(wrong & (y == 1)).sum()}", flush=True)
    for kk in sorted(set(k[wrong])):
        m = wrong & (k == kk); print(f"     {kk:16s} 오답 {m.sum():3d} | p 중앙 {np.median(p[m]):.3f} | p∈[.4,.5) {int(((p >= .4) & (p < .5) & m).sum())} · p∈[.5,.6) {int(((p >= .5) & (p < .6) & m).sum())}", flush=True)
# ② 조각 색 통계
S, ST = 224, 112
def grid(w, h):
    xs = np.linspace(0, w - S, int(np.ceil((w - S) / ST)) + 1).round().astype(int); ys = np.linspace(0, h - S, int(np.ceil((h - S) / ST)) + 1).round().astype(int)
    return [(int(x), int(y)) for y in ys for x in xs], len(xs), len(ys)
def colfeat(path):
    a = np.asarray(Image.open(path).convert("RGB")); h, w = a.shape[:2]; lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    g, nx, ny = grid(w, h); v = np.array([[*lab[y:y + S, x:x + S].reshape(-1, 3).mean(0), np.hypot(lab[y:y + S, x:x + S, 1] - 128, lab[y:y + S, x:x + S, 2] - 128).mean()] for x, y in g])
    d_img = v - np.median(v, 0); V = v.reshape(ny, nx, 4); pad = np.pad(V, ((1, 1), (1, 1), (0, 0)), mode="edge")
    nb = np.stack([pad[i:i + ny, j:j + nx] for i in range(3) for j in range(3) if (i, j) != (1, 1)]).mean(0).reshape(-1, 4)
    return np.c_[d_img, v - nb].astype(np.float32)   # (54, 8)
print("\n== ② 부분 스타일 조각 색 통계(h3 학습 자료, 묶음 5폴드 OOF, 탐색) ==", flush=True)
F = H.load("im_set_h3"); rr = man("im_set_h3"); kd = np.array([r["kind"] for r in rr]); grp = np.array([r["group"] for r in rr])
use = np.where(((kd == "real") | (kd == "partial_style")) & F["ok"])[0]
CF = np.stack([colfeat(L / "im_set_h3/img" / rr[i]["file"]) for i in use]); ES = F["es"][use].astype(np.float32); yk = kd[use]
OVp = {j: H.ov(L / "im_set_h3/mask" / rr[i]["file"].replace(".jpg", ".png")) for j, i in enumerate(use) if kd[i] == "partial_style"}
res = {"emb": np.zeros(len(use)), "emb+col": np.zeros(len(use)), "col": np.zeros(len(use))}
for tr, te in GroupKFold(5).split(use, groups=grp[use]):
    rng = np.random.default_rng(62); Xs = {k: [] for k in res}; ys = []
    for j in tr:
        if yk[j] == "real": pk = rng.choice(54, 12, replace=False); lab_ = [0] * 12
        else:
            pk = np.where(OVp[j] >= .3)[0]; lab_ = [1] * len(pk)
        if not len(pk): continue
        Xs["emb"].append(ES[j][pk]); Xs["col"].append(CF[j][pk]); Xs["emb+col"].append(np.c_[ES[j][pk], CF[j][pk]]); ys += lab_
    for k in res:
        m = H.lrfit(np.concatenate(Xs[k]), np.array(ys)); Xt = {"emb": ES[te], "col": CF[te], "emb+col": np.concatenate([ES[te], CF[te]], -1)}[k]
        res[k][te] = H.lrp(m, Xt.reshape(len(te) * 54, -1)).reshape(len(te), 54).max(1)
y = (yk == "partial_style").astype(int)
for k, s in res.items():
    th = np.quantile(s[y == 0], 1 - .0025); print(f"   [{k:8s}] 사진 AUC {roc_auc_score(y, s):.4f} | 진짜 0.25% 문턱에서 부분 스타일 탐지 {(s[y == 1] >= th).mean():.3f} (n={y.sum()})", flush=True)
ops = np.array([("tone" in rr[i]["info"] or "hsv" in rr[i]["info"] or "lab" in rr[i]["info"]) and not any(o in rr[i]["info"] for o in ("detail", "smooth", "clahe")) for i in use])
for k, s in res.items():
    th = np.quantile(s[y == 0], 1 - .0025); m = (y == 1) & ops; print(f"   [{k:8s}] 색·톤만 바꾼 부분 스타일 n={m.sum()} 탐지 {(s[m] >= th).mean():.3f}", flush=True)
print("AUDIT_DONE", flush=True)
