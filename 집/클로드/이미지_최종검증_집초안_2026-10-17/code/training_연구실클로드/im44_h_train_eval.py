"""IM44: 후보 H 학습·평가 — IM_유형1to5_H_v1.md(+ 비평 반영판). 서버(nadan). 입력 local/feats_h/*.npz(im43).
E_f(저수준38+emb 로지스틱), E_ps2·E_g7(B/16 조각 로지스틱, 마스크 겹침 라벨, 최댓값). θ = 진짜 풀 99.75% 분위. p_H = max(p6, g_k(s_k)).
비평 반영(v2): 진짜 확정(rc) 사진은 θ_rc = rc 진짜 풀 99.9% 분위(더 엄격), 오탐은 세 전문가 합집합으로 셈, R3는 정밀 대조 끈 상태(ppre 기준)도 보고, R3 진짜 오탐 +3장 이하 · 고친 가짜 > 새 오탐.
사용: python im44_h_train_eval.py [open_r3|no] [flip=.0025] [flip_rc=.001] [tag=v1] [jig]  (jig: im_set_h3j 흔들기 진짜를 학습·θ 풀에 진짜로 추가 = 후보 H2)   (open_r3 없으면 R3는 열지 않고 학습·θ·탐색 지표만)
"""
import sys, csv, json, numpy as np
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
B = Path.home() / "storage/nadan/farmai_img"; L = B / "data/local"; FH = L / "feats_h"
S, ST, NB = 224, 112, 2000
FLIP = float(sys.argv[2]) if len(sys.argv) > 2 else .0025; FLIP_RC = float(sys.argv[3]) if len(sys.argv) > 3 else .001; TAG = sys.argv[4] if len(sys.argv) > 4 else "v1"
FILT = {"filter_jpeg", "filter_blur", "filter_noise"}

def grid(w, h):
    xs = np.linspace(0, w - S, int(np.ceil((w - S) / ST)) + 1).round().astype(int); ys = np.linspace(0, h - S, int(np.ceil((h - S) / ST)) + 1).round().astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]

def ov(mp):
    m = np.asarray(Image.open(mp).convert("L"), np.float32) / 255; h, w = m.shape; return np.array([(m[y:y + S, x:x + S] > .5).mean() for x, y in grid(w, h)], np.float32)

def load(sn):
    z = np.load(FH / f"{sn}.npz"); return {k: z[k] for k in z.files}

def man(sn): return list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")))

def lrfit(X, y, C=1.0):
    sc = StandardScaler().fit(X); lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
    return dict(mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32), coef=lr.coef_.ravel().astype(np.float32), intercept=lr.intercept_.astype(np.float32))

def lrp(m, X): return 1 / (1 + np.exp(-(((X - m["mean"]) / m["scale"]) @ m["coef"] + m["intercept"][0])))
def g(s, th): return np.where(s < th, .5 * s / th, .5 + .5 * (s - th) / (1 - th))
fX = lambda F: np.c_[F["low"], F["emb"]]

def train(F, rows, idx, OV):
    y = np.array([int(rows[i]["label"]) for i in idx]); k = np.array([rows[i]["kind"] for i in idx]); M = {}
    sel = (k == "real") | np.isin(k, list(FILT)); M["E_f"] = lrfit(fX(F)[idx][sel], y[sel], C=.5)
    rng = np.random.default_rng(44)
    for name, kind in (("E_ps2", "partial_style"), ("E_g7", "partial_gen_v7")):
        Xp, yp = [], []
        for j, i in enumerate(idx):
            if k[j] == "real": pick = rng.choice(54, 12, replace=False); Xp.append(F["es"][i][pick].astype(np.float32)); yp += [0] * 12
            elif k[j] == kind and i in OV:
                pos = OV[i] >= .3
                if pos.any(): Xp.append(F["es"][i][pos].astype(np.float32)); yp += [1] * int(pos.sum())
        M[name] = lrfit(np.concatenate(Xp), np.array(yp), C=1.0)
    return M

def score(M, F):
    es = F["es"].astype(np.float32); n = len(es)
    return {"E_f": lrp(M["E_f"], fX(F)), "E_ps2": lrp(M["E_ps2"], es.reshape(n * 54, -1)).reshape(n, 54).max(1), "E_g7": lrp(M["E_g7"], es.reshape(n * 54, -1)).reshape(n, 54).max(1)}

def apply(p6, Sc, TH, THrc=None, rc=None):
    p = p6.copy()
    for k in TH:
        th = TH[k] if THrc is None else np.where(rc, THrc[k], TH[k]); p = np.maximum(p, g(Sc[k], th))
    return p

def main(open_r3):
    H = load("im_set_h3"); rows = man("im_set_h3"); assert len(rows) == len(H["files"]) and all(r["file"] == f for r, f in zip(rows, H["files"]))
    if len(sys.argv) > 5 and sys.argv[5] == "jig":
        J = load("im_set_h3j"); rj = man("im_set_h3j"); assert all(r["file"] == f for r, f in zip(rj, J["files"]))
        for r in rj: r["kind"] = "real"   # 흔들기 진짜 = 진짜
        g_of = {r["source"]: r["group"] for r in rows if r["kind"] == "real"}; assert all(g_of[r["source"]] == r["group"] for r in rj)   # 흔들기판 = 원본과 같은 묶음 → 같은 폴드
        H = {k: np.concatenate([H[k], J[k]]) for k in ("files", "emb", "es", "low", "p6", "ok", "ppre", "rc")}; rows = rows + rj; print("흔들기 진짜 추가", len(rj), "(묶음 일치 assert 통과)", flush=True)
    ok = H["ok"]; grp = np.array([r["group"] for r in rows]); kd = np.array([r["kind"] for r in rows])
    OV = {i: ov(L / "im_set_h3/mask" / rows[i]["file"].replace(".jpg", ".png")) for i in range(len(rows)) if kd[i] in ("partial_style", "partial_gen_v7") and ok[i]}
    allidx = np.where(ok)[0]; print("H 학습", len(allidx), dict(zip(*np.unique(kd[allidx], return_counts=True))), flush=True)
    # 묶음 5폴드 OOF(진짜 θ 풀용 + 학습 내 탐지율 참고)
    oof = {k: np.full(len(rows), np.nan) for k in ("E_f", "E_ps2", "E_g7")}
    for tr, te in GroupKFold(5).split(allidx, groups=grp[allidx]):
        M = train(H, rows, allidx[tr], OV); sc = score(M, {k: H[k][allidx[te]] for k in ("low", "emb", "es")})
        for k in oof: oof[k][allidx[te]] = sc[k]
    M = train(H, rows, allidx, OV)
    rmask = kd[allidx] == "real"; pool = {k: [oof[k][allidx][rmask]] for k in oof}; prc = [H["rc"][allidx][rmask]]
    for sn in ("im_set_newreal", "im_set_realg", "im_set_realn"):
        F = load(sn); sc = score(M, F); prc.append(F["rc"][F["ok"]])
        for k in pool: pool[k].append(sc[k][F["ok"]])
    prc = np.concatenate(prc); TH = {k: float(np.quantile(np.concatenate(v), 1 - FLIP)) for k, v in pool.items()}
    THrc = {k: float(np.quantile(np.concatenate(v)[prc], 1 - FLIP_RC)) for k, v in pool.items()}
    print(f"[설정 {TAG}] FLIP {FLIP} FLIP_RC {FLIP_RC}", flush=True); print("θ", {k: round(v, 4) for k, v in TH.items()}, "θ_rc", {k: round(v, 4) for k, v in THrc.items()}, "| 진짜 풀", len(prc), "중 진짜확정", int(prc.sum()), flush=True)
    un = np.zeros(len(prc), bool)
    for k, v in pool.items(): un |= np.concatenate(v) >= np.where(prc, THrc[k], TH[k])
    print(f"   진짜 풀 합집합 오탐률(전문가 셋 중 하나라도) {un.mean():.4f}", flush=True)
    for k, kk in (("E_f", FILT), ("E_ps2", {"partial_style"}), ("E_g7", {"partial_gen_v7"})):
        m = np.isin(kd, list(kk)) & ok; print(f"   [H 학습 OOF] {k} 해당 유형 θ 이상 {(oof[k][m] >= TH[k]).mean():.3f} (n={m.sum()}) | 진짜 {(oof[k][(kd == 'real') & ok] >= TH[k]).mean():.4f}", flush=True)
    np.savez(L / f"preds/h_model_{TAG}.npz", **{f"{k}__{a}": v for k, d in M.items() for a, v in d.items()}); json.dump(dict(theta=TH, theta_rc=THrc), open(L / f"preds/h_theta_{TAG}.json", "w"))
    acc = lambda p, y, ix: .5 * (p[ix][y[ix] == 0] < .5).mean() + .5 * (p[ix][y[ix] == 1] >= .5).mean()
    R = {}
    def ev(sn, y, grp_=None, kinds=None):
        F = load(sn); sc = score(M, F); p6 = F["p6"]; ph = apply(p6, sc, TH, THrc, F["rc"]); R[sn] = (p6, ph, y, F["ppre"], apply(F["ppre"], sc, TH))
        s = f"[{sn}] n={len(p6)}"
        if (y == 0).any(): s += f" 진짜 오탐 {(p6[y == 0] >= .5).mean():.4f} → {(ph[y == 0] >= .5).mean():.4f} ({int((ph[y == 0] >= .5).sum() - (p6[y == 0] >= .5).sum()):+d}장)"
        if (y == 1).any(): s += f" 탐지 {(p6[y == 1] >= .5).mean():.4f} → {(ph[y == 1] >= .5).mean():.4f}"
        if (y == 0).any() and (y == 1).any(): s += f" | 균형 정확도 {acc(p6, y, np.arange(len(y))):.4f} → {acc(ph, y, np.arange(len(y))):.4f}"
        print(s, flush=True)
        if kinds is not None:
            for k in sorted(set(kinds)):
                m = kinds == k; print(f"     {k:16s} n={m.sum():4d} 정답률 {((p6[m] >= .5) == (y[m] == 1)).mean():.3f} → {((ph[m] >= .5) == (y[m] == 1)).mean():.3f}", flush=True)
        return p6, ph
    r2 = man("im_set_r2"); yr = np.array([int(r["label"]) for r in r2]); ev("im_set_r2", yr, kinds=np.array([r["kind"] for r in r2]))
    fpd = 0
    for sn, lb in (("sealed_real", 0), ("sealed_brown", 0), ("sealed_bright", 0), ("sealed_qwen", 1), ("sealed_v7", 1)):
        F = load(sn); a, b = ev(sn, np.full(len(F["p6"]), lb))
        if lb == 0: fpd += int((b >= .5).sum() - (a >= .5).sum())
    P = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; py = np.array([int(r["label"]) for r in csv.DictReader(open(P / "sample_labels.csv"))])
    a, b = ev("pub21", py); pub_ok = int(((b >= .5) == py).sum())
    F = load("jig500"); a, b = ev("jig500", np.zeros(len(F["p6"]), int)); jig_d = (b >= .5).mean() - (a >= .5).mean()
    r2ok = acc(R["im_set_r2"][1], yr, np.arange(len(yr))) >= acc(R["im_set_r2"][0], yr, np.arange(len(yr)))
    print(f"\n[탐색 지표] 봉인 진짜 {fpd:+d}장(≤3) · 공개 {pub_ok}/21 · 흔들기 {jig_d:+.4f}(≤.03) · R2 비하락 {r2ok}", flush=True)
    if not open_r3: print("H_TRAIN_DONE (R3 미개봉)", flush=True); return
    r3 = man("im_set_r3"); assert not ({r["source"] for r in r3} & {r["source"] for r in rows}) and not ({r["group"] for r in r3} & {r["group"] for r in rows})   # R3 ↔ 학습 원본·묶음 겹침 없음
    y3 = np.array([int(r["label"]) for r in r3]); g3 = np.array([r["group"] for r in r3]); p6, ph = ev("im_set_r3", y3, kinds=np.array([r["kind"] for r in r3]))
    gs = np.array(sorted(set(g3))); gi = {x: np.where(g3 == x)[0] for x in gs}; rng = np.random.default_rng(15); d = []
    for _ in range(NB):
        ix = np.concatenate([gi[x] for x in rng.choice(gs, len(gs))])
        if (y3[ix] == 0).any() and (y3[ix] == 1).any(): d.append(acc(ph, y3, ix) - acc(p6, y3, ix))
    pw = float((np.array(d) <= 0).mean()); fp3 = int((ph[y3 == 0] >= .5).sum() - (p6[y3 == 0] >= .5).sum()); fix3 = int((ph[y3 == 1] >= .5).sum() - (p6[y3 == 1] >= .5).sum())
    print(f"[R3] 균형 정확도 {acc(p6, y3, np.arange(len(y3))):.4f} → {acc(ph, y3, np.arange(len(y3))):.4f}, p_worse {pw:.4f}, 고친 가짜 {fix3}장 · 새 진짜 오탐 {fp3}장", flush=True)
    _, _, _, pp, pq = R["im_set_r3"]; k3 = np.array([r["kind"] for r in r3])
    print(f"[R3, 정밀 대조 끈 상태(참고)] 균형 정확도 {acc(pp, y3, np.arange(len(y3))):.4f} → {acc(pq, y3, np.arange(len(y3))):.4f} | " + ", ".join(f"{k} {((pp[k3 == k] >= .5) == (y3[k3 == k] == 1)).mean():.3f}→{((pq[k3 == k] >= .5) == (y3[k3 == k] == 1)).mean():.3f}" for k in sorted(set(k3))), flush=True)
    proc_ok = True
    for kind in ("sharp", "png", "area", "webp", "c444"):   # 중간 비평 추가 기준(개봉 전 고정): 처리별 진짜 오탐 ≤ 나단6 + 1%p
        a, b = ev(f"im_set_r3p_{kind}", np.zeros(len(load(f"im_set_r3p_{kind}")["p6"]), int)); proc_ok &= bool((b >= .5).mean() <= (a >= .5).mean() + .01)
    c = [acc(ph, y3, np.arange(len(y3))) > acc(p6, y3, np.arange(len(y3))) and pw < .025 and fix3 > fp3, fp3 <= 3, fpd <= 3, pub_ok == 21, jig_d <= .03, r2ok, proc_ok]
    print(f"\n== 판정(IM_유형1to5_H) == R3↑&p {c[0]} · R3진짜 {c[1]} · 봉인진짜 {c[2]} · 공개 {c[3]} · 흔들기 {c[4]} · R2 {c[5]} · R3처리5종 {c[6]} → {'채택 후보' if all(c) else '불채택'}", flush=True)
    print("H_EVAL_DONE", flush=True)

if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "open_r3")
