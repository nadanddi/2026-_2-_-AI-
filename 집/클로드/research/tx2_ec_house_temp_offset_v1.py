# -*- coding: utf-8 -*-
"""TX2 (2026-10-13 집 클로드; 사용자: '온도쪽을 더 보자').  사전 고정(커밋 후 실행).  온도.
근거: HX9 근권(6.797) — 고EC 동은 배지가 입력 모델보다 +.25℃ 따뜻(학습 정답으로 잰 차이); TX1(서술) — 13회차 구성 잔차가 고EC 동에서 양수, 추운 외삽(EXT10 +.64·EXT12 +.44)에서 크고,
  EC 모델 예측 sg(대상 정답 미사용)로도 일부 가려짐(EXT10 잔차–ecsg r .41).
규칙: pred_h = q_h + .25 · w_h · g_h,  w_h = clip((cmEC_h − .8)/.4, 0, 1), cmEC_h = EC 모델 예측(DIAG10 OOF 시드 평균, 평가에선 제출 EC 예측) 0..h 누적평균.
  안 COLD: g_h = clip((10 − in_temp_h)/2, 0, 1)  (실내 ≤ 8℃ 에서 1, 10℃ 이상 0 — G_C2 게이트와 같은 모양).   안 ALL: g_h = 1.
  .25 는 고정(검증 잔차로 맞추지 않음).  기준 = 13회차 구성(TC1c 셀 + v7 앵커, 측정 감쇠).
판정(온도 범위 규칙 6.485, 안 2개 α .0125):  COLD → 주 EXT10: 4칸 모두 개선 & 농장×5일 블록 부트스트랩 P < .0125;  ALL → 주 DIAG10 과 EXT10 둘 다 같은 조건.
  나머지(DIAG10/EXT12/EL1/F~L 중 주 아닌 것): 유의한 악화 없음(칸 평균 악화이면서 블록 부트스트랩 '후보가 나음' 비율 < .0125 이면 FAIL).
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s_ = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A7)
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261121)
R = A7.A2.A1.frames()
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
E = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); E = E[(E.tag == "CUR") & (E.validator == "DIAG10")].copy()
E["rec"] = E.row_id.str[:7]; Eh = E.groupby(["rec", "hour"]).sg.mean().reset_index().sort_values(["rec", "hour"])
Eh["cm"] = Eh.groupby("rec").sg.transform(lambda s: s.expanding().mean()); CM = {(r, int(h)): c for r, h, c in zip(Eh.rec, Eh.hour, Eh.cm)}
ARMS = ("COLD", "ALL"); res = []; blocks = {}
for L, (M, cells) in R.items():
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values; inT = M.in_temp.values; y = M.sub_temp.values
    kb, Ab = A7.build(M, L7f); cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, h)]); w = np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1)
    gate = {"COLD": np.nan_to_num(np.clip((10 - inT) / 2, 0, 1), nan=0.0), "ALL": np.ones(len(M))}
    for ci, c in enumerate(cells):
        base = A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5)
        for a in ARMS:
            new = base + .25 * w * gate[a]
            res.append(dict(layout=L, cell=ci, arm=a, pct=100 * (rm(new - y) / rm(base - y) - 1), p2=100 * (rm((new - y)[M.day.values >= 179]) / rm((base - y)[M.day.values >= 179]) - 1), changed=int((np.abs(new - base) > 1e-9).sum())))
            if ci == 0:
                bk = M.farm + "_" + (M.day // 5).astype(str); blocks[(L, a)] = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(bk.values).mean()
S = pd.DataFrame(res); pd.set_option("display.width", 220)
print(S.pivot_table(index=["arm", "layout"], columns="cell", values="pct").round(3).to_string())
print(S.groupby(["arm", "layout"]).agg(p2=("p2", "mean"), changed=("changed", "mean")).round(3).to_string())
def bootP(L, a):
    v = blocks[(L, a)].values; ix = rng.integers(0, len(v), (20000, len(v))); m = v[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for a, prim in (("COLD", ["EXT10"]), ("ALL", ["DIAG10", "EXT10"])):
    ok = True; msg = []
    for L in R:
        cells = S[(S.arm == a) & (S.layout == L)].pct; pw, pb = bootP(L, a)
        if L in prim:
            c = bool((cells < 0).all()) and pw < .0125; ok &= c; msg.append(f"{L}[primary] cells {int((cells < 0).sum())}/4 P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}")
        else:
            bad = cells.mean() > 0 and pb < .0125; ok &= not bad; msg.append(f"{L} mean {cells.mean():+.3f}% P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}")
    print(f"\n== TX2 {a}:"); print("\n".join("   " + m for m in msg)); print(f"TX2 {a}:", "ADOPT" if ok else "FAIL")
