"""IM56: 후보 I 학습·평가 — IM_약점보완_I_v2.md. 서버(nadan). 입력 feats_h(im43): im_set_h3, im_set_h3j, im_set_h3s, im_set_h3g, 평가 세트들.
나단8 = 나단6 + H2(E_f·E_ps2·E_g7, θ h2) + 형식 게이트. I = 나단6 + E_f·E_ps2(H2 그대로) + E_g7'(v7+v7s 재학습) + E_rg(신설) + 같은 게이트.
사용: python im56_i_train_eval.py [open_r4]"""
import sys, io, csv, json, numpy as np
from pathlib import Path
from PIL import Image, JpegImagePlugin
from sklearn.model_selection import GroupKFold
OPEN = len(sys.argv) > 1 and sys.argv[1] == "open_r4"
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im43_h_feats import paths_of  # noqa
L = H.L; NB = 2000
b = io.BytesIO(); Image.new("RGB", (64, 64), (120, 130, 140)).save(b, "JPEG", quality=92); REFQ = Image.open(io.BytesIO(b.getvalue())).quantization
def std(p):
    try: im = Image.open(p); return im.format == "JPEG" and JpegImagePlugin.get_sampling(im) == 2 and im.quantization == REFQ
    except Exception: return False
z = np.load(L / "preds/h_model_h2.npz"); M8 = {}
for k, v in z.items(): a, c = k.split("__"); M8.setdefault(a, {})[c] = v
T8 = json.load(open(L / "preds/h_theta_h2.json"))
def man(sn): return list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")))
def lowemb(F): return np.c_[F["low"], F["emb"]]
def patch_scores(m, F):
    es = F["es"].astype(np.float32); n = len(es); return H.lrp(m, es.reshape(n * 54, -1)).reshape(n, 54).max(1)

def main():
    # 학습 자료: h3(진짜·가짜) + h3j(흔들기 진짜) + h3s(작은 부분 생성) + h3g(SDXL 재그리기)
    parts = []
    for sn in ("im_set_h3", "im_set_h3j", "im_set_h3s", "im_set_h3g"):
        F = H.load(sn); rr = man(sn); assert len(rr) == len(F["files"]) and all(r["file"] == f for r, f in zip(rr, F["files"]))
        parts.append((sn, F, rr))
    cat = lambda key: np.concatenate([F[key] for _, F, _ in parts])
    F = {k: cat(k) for k in ("emb", "es", "low", "ok", "rc")}
    rows = [dict(r, _sn=sn) for sn, _, rr in parts for r in rr]
    kd = np.array(["real" if r["kind"] in ("real", "real_jig") else r["kind"] for r in rows]); grp = np.array([r["group"] for r in rows]); ok = F["ok"]
    print("I 학습", {k: int((kd == k).sum()) for k in sorted(set(kd))}, flush=True)
    OV = {}
    for i, r in enumerate(rows):
        if kd[i] in ("partial_gen_v7", "partial_gen_v7s") and ok[i]: OV[i] = H.ov(L / r["_sn"] / "mask" / r["file"].replace(".jpg", ".png"))
    def train(idx):
        rng = np.random.default_rng(56); Xp, yp = [], []
        for i in idx:
            if kd[i] == "real" and rows[i]["_sn"] == "im_set_h3": pk = rng.choice(54, 12, replace=False); Xp.append(F["es"][i][pk].astype(np.float32)); yp += [0] * 12
            elif kd[i] == "real": pk = rng.choice(54, 6, replace=False); Xp.append(F["es"][i][pk].astype(np.float32)); yp += [0] * 6
            elif i in OV:
                pos = OV[i] >= .3
                if pos.any(): Xp.append(F["es"][i][pos].astype(np.float32)); yp += [1] * int(pos.sum())
        sel = np.array([kd[i] in ("real", "regen_xl") for i in idx])
        return {"E_g7": H.lrfit(np.concatenate(Xp), np.array(yp)), "E_rg": H.lrfit(lowemb(F)[idx][sel], (kd[idx][sel] == "regen_xl").astype(int), C=.5)}
    def score(M, G): return {"E_g7": patch_scores(M["E_g7"], G), "E_rg": H.lrp(M["E_rg"], lowemb(G))}
    allidx = np.where(ok)[0]; oof = {k: np.full(len(rows), np.nan) for k in ("E_g7", "E_rg")}
    for tr, te in GroupKFold(5).split(allidx, groups=grp[allidx]):
        M = train(allidx[tr]); sc = score(M, {k: F[k][allidx[te]] for k in ("emb", "es", "low")})
        for k in oof: oof[k][allidx[te]] = sc[k]
    MI = train(allidx)
    # 진짜 풀: 학습 진짜 OOF + newreal·realg·realn(전체 모델) — 기존 E_f·E_ps2 점수도 함께(합집합)
    rm = (kd == "real") & ok; P = {"E_g7": [oof["E_g7"][rm]], "E_rg": [oof["E_rg"][rm]]}; prc = [F["rc"][rm]]
    G8 = {"E_f": [H.lrp(M8["E_f"], lowemb(F)[rm])], "E_ps2": [patch_scores(M8["E_ps2"], {"es": F["es"][rm]})]}
    for sn in ("im_set_newreal", "im_set_realg", "im_set_realn"):
        G = H.load(sn); m = G["ok"]; s = score(MI, G); prc.append(G["rc"][m])
        for k in P: P[k].append(s[k][m])
        G8["E_f"].append(H.lrp(M8["E_f"], lowemb(G))[m]); G8["E_ps2"].append(patch_scores(M8["E_ps2"], G)[m])
    prc = np.concatenate(prc); P = {k: np.concatenate(v) for k, v in P.items()}; G8 = {k: np.concatenate(v) for k, v in G8.items()}
    old = np.zeros(len(prc), bool)
    for k in ("E_f", "E_ps2"): old |= G8[k] >= np.where(prc, T8["theta_rc"][k], T8["theta"][k])
    for fl, flrc in ((.0025, .001), (.0015, .00075), (.001, .0005), (.0005, .0005)):
        TH = {k: float(np.quantile(P[k], 1 - fl)) for k in P}; THrc = {k: float(np.quantile(P[k][prc], 1 - flrc)) for k in P}
        un = old.copy()
        for k in P: un |= P[k] >= np.where(prc, THrc[k], TH[k])
        print(f"[θ 단계 {fl}/{flrc}] θ {({k: round(v, 4) for k, v in TH.items()})} θ_rc {({k: round(v, 4) for k, v in THrc.items()})} | 진짜 풀 N={len(prc)} 합집합 오탐 {un.mean():.4f}", flush=True)
        if un.mean() <= .005: break
    else:
        print("== 합집합 예산 0.5% 초과 → 불채택 ==", flush=True); return
    for k, kk in (("E_g7", ("partial_gen_v7", "partial_gen_v7s")), ("E_rg", ("regen_xl",))):
        m = np.isin(kd, kk) & ok; print(f"   [OOF] {k} 해당 유형 θ 이상 {(oof[k][m] >= TH[k]).mean():.3f} (n={m.sum()})", flush=True)
    np.savez(L / "preds/i_model_v1.npz", **{f"{k}__{a}": v for k, d in MI.items() for a, v in d.items()})
    json.dump(dict(theta={**T8["theta"], **TH, "E_g7": TH["E_g7"]}, theta_rc={**T8["theta_rc"], **THrc}), open(L / "preds/i_theta_v1.json", "w"))
    def outputs(sn):
        G = H.load(sn); s = np.array([std(p) for p in paths_of(sn)]); p6 = G["p6"]; rc = G["rc"]
        s8 = H.score(M8, G); p8 = np.where(s, H.apply(p6, s8, T8["theta"], T8["theta_rc"], rc), p6)
        sI = {"E_f": s8["E_f"], "E_ps2": s8["E_ps2"], **score(MI, G)}; thI = {**T8["theta"], **TH}; thIrc = {**T8["theta_rc"], **THrc}
        pI = np.where(s, H.apply(p6, sI, thI, thIrc, rc), p6); return p8, pI
    def labels(sn):
        if sn == "pub21": return np.array([int(r["label"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))]), None, None
        if sn.startswith("sealed_"): rr = list(csv.DictReader(open(L / "sealed_c3" / sn[7:] / "manifest.csv", encoding="utf-8")))
        elif sn == "jig500": return np.zeros(500, int), None, None
        else: rr = man(sn)
        return np.array([int(r["label"]) for r in rr]), np.array([r["kind"] for r in rr]), np.array([r["group"] for r in rr])
    C = {}
    def rep(sn):
        y, k, g = labels(sn); p8, pI = outputs(sn); new_fp = int(((p8 < .5) & (pI >= .5) & (y == 0)).sum()) - int(((p8 >= .5) & (pI < .5) & (y == 0)).sum())
        fixed = int(((p8 < .5) & (pI >= .5) & (y == 1)).sum()) - int(((p8 >= .5) & (pI < .5) & (y == 1)).sum())
        print(f"[{sn}] n={len(y)} 진짜 오탐 증가 {new_fp:+d} · 고친 가짜 {fixed:+d}", flush=True)
        if k is not None:
            for kk in sorted(set(k[y == 1])):
                m = (k == kk) & (y == 1); print(f"     {kk:18s} n={m.sum():4d} 나단8 {(p8[m] >= .5).mean():.3f} → I {(pI[m] >= .5).mean():.3f}", flush=True)
        C[sn] = (y, k, g, p8, pI, new_fp, fixed); return y, k, g, p8, pI
    for sn in ("pub21", "jig500", "sealed_real", "sealed_brown", "sealed_bright", "im_set_r3", "im_set_gpt3", "im_set_sd2"): rep(sn)
    y, k, _, p8, pI = C["im_set_gpt3"][:5]; m = k == "gpt3_s1"; g_ok = (pI[m] >= .5).mean() > (p8[m] >= .5).mean()
    y, k, _, p8, pI = C["im_set_sd2"][:5]; m = (k == "partial_gen_sd"); sd_ok = (pI[m] >= .5).mean() >= (p8[m] >= .5).mean()
    y, k, _, p8, pI = C["im_set_r3"][:5]; r3_ok = (pI[y == 1] >= .5).sum() >= (p8[y == 1] >= .5).sum()
    pub_ok = int(((C["pub21"][4] >= .5) == C["pub21"][0]).sum()) == 21
    fp_ok = C["im_set_r3"][5] <= 2 and C["jig500"][5] <= 2 and sum(C[s][5] for s in ("sealed_real", "sealed_brown", "sealed_bright")) <= 2
    print(f"\n[탐색 지표] gpt3_s1 {g_ok} · sd2 {sd_ok} · R3 가짜 {r3_ok} · 공개 {pub_ok} · 진짜 오탐(R3·jig·봉인) {fp_ok}", flush=True)
    if not OPEN: print("I_TRAIN_DONE (R4 미개봉)", flush=True); return
    y, k, g, p8, pI = rep("im_set_r4")
    acc = lambda p, ix: .5 * (p[ix][y[ix] == 0] < .5).mean() + .5 * (p[ix][y[ix] == 1] >= .5).mean(); al = np.arange(len(y)); rng = np.random.default_rng(56)
    d = [acc(pI, ix) - acc(p8, ix) for ix in (rng.integers(0, len(y), len(y)) for _ in range(NB))]; pw = float((np.array(d) <= 0).mean())
    gs = np.array(sorted(set(g))); gi = {x: np.where(g == x)[0] for x in gs}; dg = []
    for _ in range(NB):
        ix = np.concatenate([gi[x] for x in rng.choice(gs, len(gs))])
        if (y[ix] == 0).any() and (y[ix] == 1).any(): dg.append(acc(pI, ix) - acc(p8, ix))
    pwg = float((np.array(dg) <= 0).mean()); same = sum(acc(pI, gi[x]) >= acc(p8, gi[x]) for x in gs if (y[gi[x]] == 0).any() and (y[gi[x]] == 1).any())
    print(f"[R4] 균형 정확도 나단8 {acc(p8, al):.4f} → I {acc(pI, al):.4f} | p_worse 사진 {pw:.4f} 묶음 {pwg:.4f} | 묶음 같은 방향 {same}/{len(gs)}", flush=True)
    new2 = all((pI[(k == kk) & (y == 1)] >= .5).mean() > (p8[(k == kk) & (y == 1)] >= .5).mean() for kk in ("partial_gen_v7s", "regen_xl"))
    old5 = all((pI[(k == kk) & (y == 1)] >= .5).sum() >= (p8[(k == kk) & (y == 1)] >= .5).sum() - 1 for kk in ("filter_jpeg", "filter_blur", "filter_noise", "partial_style", "partial_gen_v7"))
    c = [new2, old5, acc(pI, al) > acc(p8, al) and pw < .025 and same >= len(gs) - 1, C["im_set_r4"][5] <= 2 and fp_ok, g_ok, sd_ok and r3_ok and pub_ok]
    print(f"\n== 판정(IM_약점보완_I_v2) == 새2종↑ {c[0]} · 기존5종 무손해 {c[1]} · R4↑&p&묶음 {c[2]} · 진짜 오탐 {c[3]} · gpt3_s1 {c[4]} · 기타 무하락 {c[5]} → {'채택 후보' if all(c) else '불채택'}", flush=True)
    print("I_EVAL_DONE", flush=True)

if __name__ == "__main__":
    main()
