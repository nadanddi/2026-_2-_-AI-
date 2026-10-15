# -*- coding: utf-8 -*-
"""AN1 (2026-10-15 집 클로드; CMX1(6.944) 손익분기 분석 후속, 사용자 '해봐').  사전 고정(커밋 후 실행).  온도.  학습 없음.
CMX1: 앵커는 링크 정밀도가 손익분기(온도 ≈ .76~.82)를 넘을 때만 이득. FL p ≥ .8 정밀도 .91(위), FL .5~.8 ≈ .72(아래);
  FLN(6.932, '전날 없음' 선택지) p ≥ .5 교차검증 정밀도 .938(위, 포괄 .561 > FL .8 의 .526).
안: 27번 온도의 FL K8 앵커(fl3 FL p ≥ .8)를 FLN 앵커로 교체 — FLN 학습표(local/fln1_train_links.csv) p ≥ .5, 전날이 그 폴드 학습 쪽 정답 기록, |dT_air| ≤ 1.5,
  식은 그대로 q_h + BETA[h]·clip(A23 − .36 − q0, ±2) (새 상수 없음).
기준 R27 = W40G-S 셀 → FL K8 → TX3,  후보 AN = W40G-S 셀 → FLN 앵커 → TX3.
배치: N·CK2(OS1 v2 저장 셀), EL1(yl1_ref_ckpt — YL1 에서 학습한 W40G-S 기준 셀, common.split_mask 폴드).  거친 날 제외.
판정(단일 안 α .025): 세 배치 12셀 모두 개선 & 합친(배치 x 농장 x 5일 블록) P(악화) < .025.  서술: 앵커 수(FL / FLN / 겹침), 2차·추운 행.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u an1_fln_temp_anchor_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1, B, BETA, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "an1_fln_temp_anchor_v1.py")
import common
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
ELF = dict(W.worlds()[7])["EL1"]; CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
rng = np.random.default_rng(20270126); ok = True; diffs = []
for L, ckdir, folds, mk in (("N", O1.CKB, LF["N"], O1.masks), ("CK2", O1.CKB, LF["CK2"], O1.masks), ("EL1", CKR, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckdir, f)) for f in sorted(os.listdir(ckdir)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; pos = {r: np.where(key == r)[0] for r in set(key)}
    cg = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0); cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, h)])
    tx3 = .25 * np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1) * cg
    TRS = {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
    def anchors(LK, pmin):
        return {r: float(ST[IDX[LK[r][0]], 23]) for r in pos if r in LK and r in TRS and LK[r][1] >= pmin and LK[r][0] in TRS[r] and np.isfinite(ST[IDX[LK[r][0]], 23])
                and abs(IN_[IDX[r], 0, 0] - IN_[IDX[LK[r][0]], 23, 0]) <= 1.5}
    AF, AN = anchors(FLK, .8), anchors(LINK, .5)
    def apply(c, A):
        out = c + tx3
        for r, a in A.items():
            ix = pos[r]; q0 = c[ix][h[ix] == 0]
            if len(q0): out[ix] += BETA[h[ix]] * np.clip(a - .36 - q0[0], -2, 2)
        return out
    cells = [B.w40(M, "base_R_%d" % s, "codex_R", fam) for s in O1.SEEDS for fam in "AB"]
    PB = [apply(c, AF) for c in cells]; PN = [apply(c, AN) for c in cells]
    ch = [rm(n - y) / rm(b - y) - 1 for n, b in zip(PN, PB)]; ok &= all(x < 0 for x in ch)
    kb = (L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values; diffs.append(pd.Series(np.mean([(n - y) ** 2 - (b - y) ** 2 for n, b in zip(PN, PB)], 0)).groupby(kb).mean())
    cold = (M.in_temp <= 8).values; p2 = M.day.values >= 179
    sub = lambda m: 100 * np.mean([rm((n - y)[m]) / rm((b - y)[m]) - 1 for n, b in zip(PN, PB)]) if m.any() else np.nan
    same = sum(1 for r in set(AF) & set(AN) if FLK[r][0] == LINK[r][0])
    print(f"{L:3s} AN vs R27 {' '.join('%+.3f' % (100 * x) for x in ch)} | p2 {sub(p2):+.2f}% cold {sub(cold):+.2f}% | anchors FL {len(AF)} FLN {len(AN)} both {len(set(AF) & set(AN))} (same prev {same}) of {len(pos)}", flush=True)
S = pd.concat(diffs).values; P_ = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"pooled P(worse) {P_:.4f} | all 12 cells better {ok} -> AN1: {'ADOPT' if ok and P_ < .025 else 'FAIL'}")
