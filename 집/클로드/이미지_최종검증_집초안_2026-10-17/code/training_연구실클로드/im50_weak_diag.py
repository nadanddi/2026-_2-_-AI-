"""IM50: 약점 3종 진단(나단8 = H2+게이트 기준 놓친 사진의 특성). 서버(nadan). 탐색용."""
import sys, csv, re, json, numpy as np
from pathlib import Path
from PIL import Image
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
L = H.L; z = np.load(L / "preds/h_model_h2.npz"); M = {}
for k, v in z.items(): a, c = k.split("__"); M.setdefault(a, {})[c] = v
th = json.load(open(L / "preds/h_theta_h2.json"))
def run(sn):
    F = H.load(sn); rr = list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8"))); sc = H.score(M, F)
    ph = H.apply(F["p6"], sc, th["theta"], th["theta_rc"], F["rc"]); return F, rr, sc, ph
def q(x): return np.percentile(x, [25, 50, 75]).round(3).tolist() if len(x) else []
# 1) 부분 스타일: R3(마스크 넓이) + R2(스타일 연산)
F, rr, sc, ph = run("im_set_r3"); k = np.array([r["kind"] for r in rr]); m = k == "partial_style"
area = np.array([(np.asarray(Image.open(L / "im_set_r3/mask" / r["file"].replace(".jpg", ".png")).convert("L")) > 127).mean() if mm else np.nan for r, mm in zip(rr, m)])
hit = ph >= .5
print(f"[부분스타일 R3] n={m.sum()} 탐지 {hit[m].mean():.3f} | 넓이 사분위 탐지 {q(area[m & hit])} 놓침 {q(area[m & ~hit])} | E_ps2 놓침 중앙 {np.median(sc['E_ps2'][m & ~hit]):.4f} (θ {th['theta']['E_ps2']:.4f})")
for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, 1)):
    s = m & (area >= lo) & (area < hi); print(f"     넓이 {lo:.1f}~{hi:.1f} n={s.sum():3d} 탐지 {hit[s].mean() if s.any() else float('nan'):.3f}")
F, rr, sc, ph = run("im_set_r2"); k = np.array([r["kind"] for r in rr]); m = k == "partial_style"; hit = ph >= .5
ops = [set(re.findall(r"(lab|tone|hsv|detail|smooth|clahe)", r["info"])) for r in rr]
for op in ("lab", "tone", "hsv", "detail", "smooth", "clahe"):
    s = m & np.array([op in o for o in ops]); print(f"[부분스타일 R2] 연산 {op:6s} 포함 n={s.sum():3d} 탐지 {hit[s].mean():.3f}")
s = m & np.array([o <= {"tone", "hsv", "lab"} for o in ops]); print(f"[부분스타일 R2] 색·톤 연산만(질감 연산 없음) n={s.sum()} 탐지 {hit[s].mean():.3f}")
# 2) SD 부분 생성: 넓이
F, rr, sc, ph = run("im_set_sd2"); k = np.array([r["kind"] for r in rr]); hit = ph >= .5; m = k == "partial_gen_sd"
area = np.array([float(re.search(r"area=([\d.]+)", r["info"]).group(1)) if mm else np.nan for r, mm in zip(rr, m)])
for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, 1)):
    s = m & (area >= lo) & (area < hi); print(f"[SD 부분생성] 넓이 {lo:.2f}~{hi:.2f} n={s.sum():3d} 탐지 {hit[s].mean() if s.any() else float('nan'):.3f}")
# 3) GPT 전체 재생성(gpt3_s1): 원본 대조
F, rr, sc, ph = run("im_set_gpt3"); k = np.array([r["kind"] for r in rr]); hit = ph >= .5
for kk in ("gpt3_s1", "gpt3_pgen", "gpt3_pstyle"):
    m = k == kk; ff = F["ff"][m]
    print(f"[GPT3 {kk}] n={m.sum()} 탐지 {hit[m].mean():.3f} | 놓침 {int((~hit[m]).sum())}장: p 중앙 {np.median(ph[m][~hit[m]]) if (~hit[m]).any() else 0:.3f}, dg1 {q(ff[~hit[m], 0])}, gap {q(ff[~hit[m], 1])} | 탐지된 것 dg1 {q(ff[hit[m], 0])} gap {q(ff[hit[m], 1])}")
R = H.load("im_set_r3"); rr = list(csv.DictReader(open(L / "im_set_r3/manifest.csv", encoding="utf-8"))); y = np.array([int(r["label"]) for r in rr])
print(f"[참고 R3 진짜] dg1 {q(R['ff'][y == 0, 0])} gap {q(R['ff'][y == 0, 1])}")
print("DIAG_DONE")
