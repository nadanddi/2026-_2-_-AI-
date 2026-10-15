# -*- coding: utf-8 -*-
"""ASM1 (2026-10-15 집 클로드; 사용자 '조립 검증 진행해봐').  사전 고정(커밋 후 실행).  온도.  학습 없음(저장 예측).
5점모델_설계_v1 에서 각자 통과한 온도 부품을 한꺼번에 올려 겹침·상쇄를 확인. 배치 = 처음 쓰는 N·CK2(OS1 v2 저장 셀·V3LF), 거친 날 제외.
  q   = W40G-S 셀(BASE 1616/1717 x PFN A/B)
  G   = 3층 C1: q + .10·g'·h'·(V3LF − q), g' = clip((in_temp − 10)/2), h' = clip((그날 실내 최저 − 12)/2) (결측 1)  (6.935)
  K8  = 2층 FL 어제 앵커: fl3 학습표 FL p ≥ .8, 전날이 그 폴드 학습 쪽 정답 기록, |dT_air| ≤ 1.5: + BETA[h]·clip(A23 − .36 − x0, ±2) (x = 앵커 걸기 전 예측)
  TX3 = 4층: + .25·clip((cmNH − .8)/.4)·clip((10 − in_temp)/2)  (6.931)
팔: R27 = q → K8 → TX3 (27번 온도),  ASM = q → G → K8 → TX3.   서술: q 만, q → K8, q → TX3, q → G, ASM 에서 TX3 뺌, ASM 에서 K8 뺌.
판정(α .025): ASM vs R27 — N·CK2 각각 4셀 모두 개선 & 각각 농장 x 5일 블록 P(악화) < .025.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u asm1_temp_assembly_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, IDX, ST, IN_, RD, CM, lab, LF, rm
__file__ = os.path.join(HERE, "asm1_temp_assembly_v1.py")
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
rng = np.random.default_rng(20270122); ok = True
for L in O1.LAYS:
    M = pd.concat([pd.read_csv(os.path.join(O1.CKB, f)) for f in sorted(os.listdir(O1.CKB)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(O1.CKG, f)) for f in sorted(os.listdir(O1.CKG)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; pos = {r: np.where(key == r)[0] for r in set(key)}
    cg = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0); cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, h)])
    tx3 = .25 * np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1) * cg
    gg = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 10) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
    mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); hg = np.where(mm.isna(), 1, np.clip((mm - 12) / 2, 0, 1)); gate = gg * hg
    v = M.V3LF.values; miss = np.isnan(v)
    TRS = {}
    for k, fd in enumerate(LF[L]):
        tm, vm = O1.masks(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
    K8 = {r: float(ST[IDX[FLK[r][0]], 23]) for r in pos if r in FLK and FLK[r][1] >= .8 and FLK[r][0] in TRS[r] and np.isfinite(ST[IDX[FLK[r][0]], 23])
          and abs(IN_[IDX[r], 0, 0] - IN_[IDX[FLK[r][0]], 23, 0]) <= 1.5}
    def k8(x):
        out = x.copy()
        for r, A in K8.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] = x[ix] + BETA[h[ix]] * np.clip(A - .36 - x0[0], -2, 2)
        return out
    P = {k: [] for k in ("q", "qK8", "qTX3", "qG", "R27", "ASM", "ASM_noTX3", "ASM_noK8")}
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); gq = q + .10 * gate * (~miss) * (np.nan_to_num(v) - q)
            P["q"].append(q); P["qK8"].append(k8(q)); P["qTX3"].append(q + tx3); P["qG"].append(gq)
            P["R27"].append(k8(q) + tx3); P["ASM"].append(k8(gq) + tx3); P["ASM_noTX3"].append(k8(gq)); P["ASM_noK8"].append(gq + tx3)
    cold = (M.in_temp <= 8).values; p2 = M.day.values >= 179; bk = (M.farm + "_" + (M.day // 5).astype(str)).values
    rel = lambda a, b, m=None: 100 * np.mean([rm((x - y)[m if m is not None else slice(None)]) / rm((z - y)[m if m is not None else slice(None)]) - 1 for x, z in zip(P[a], P[b])])
    ch = [rm(a - y) / rm(b - y) - 1 for a, b in zip(P["ASM"], P["R27"])]
    dd = pd.Series(np.mean([(a - y) ** 2 - (b - y) ** 2 for a, b in zip(P["ASM"], P["R27"])], 0)).groupby(bk).mean().values
    mb = dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1); pw = float((mb >= 0).mean()); c_ok = all(x < 0 for x in ch) and pw < .025; ok &= c_ok
    print(f"{L:3s} ASM vs R27 {' '.join('%+.3f' % (100 * x) for x in ch)} | P(worse) {pw:.4f} -> {'ok' if c_ok else 'FAIL'} | cold {rel('ASM', 'R27', cold):+.2f}% p2 {rel('ASM', 'R27', p2):+.2f}%", flush=True)
    print(f"     RMSE q {np.mean([rm(x - y) for x in P['q']]):.4f} -> R27 {np.mean([rm(x - y) for x in P['R27']]):.4f} -> ASM {np.mean([rm(x - y) for x in P['ASM']]):.4f} | K8 anchors {len(K8)} of {len(pos)} records", flush=True)
    print(f"     parts vs q: K8 {rel('qK8', 'q'):+.2f}%  TX3 {rel('qTX3', 'q'):+.2f}%  G {rel('qG', 'q'):+.2f}%  R27 {rel('R27', 'q'):+.2f}%  ASM {rel('ASM', 'q'):+.2f}% | "
          f"leave-one-out vs ASM: -TX3 {rel('ASM_noTX3', 'ASM'):+.2f}%  -K8 {rel('ASM_noK8', 'ASM'):+.2f}%  -G {rel('R27', 'ASM'):+.2f}%", flush=True)
print("ASM1:", "ADOPT" if ok else "FAIL")
