# -*- coding: utf-8 -*-
"""B4 (2026-10-13 집 클로드; 사용자: '리스트 전부 다 해봐' — B4: 사슬 새벽 보정(11회차 TC1c)을 v15 짝으로).  사전 고정(커밋 후 실행).  온도.
근거: 6.540 사슬 보정 이득의 원천은 '어느 기록이 전날인가'(사슬 이력), 6.558 '다음 개선은 전날 식별 정확도'.  TC1c 사슬 = 같은 ID 앞 1~3기록 중 자정 상태 최근접.
새: 전날 = v15(없음 선택지, PI .15, 거울 G0 기하) 1순위가 p0 ≥ .5 이면 그 기록(학습 기록 또는 같은 ID 앞 번호 — 입력만 사용), 아니면 TC1c 사슬 그대로.
  바꾸는 특징: ch_in23(전날 23시 실내온도), ch_eve(18~23시 평균), jump(= in_temp − ch_in23).  ch_cost·lit1/lit2 는 그대로.  보정기·학습원·시각(0~7시)·0.5배는 TC1c 그대로.
판정: B4 가 TC1c(사슬판)를 DIAG10·EL1·F~K 모든 칸에서 이기고, 같은 날 중복 없이 합친(DIAG10+F~K 평균) 5일 블록 부트스트랩 P(악화) < .025 → ADOPT.  EXT10·EXT12 서술.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
s = importlib.util.spec_from_file_location("tc1b", os.path.join(HERE, "tc1b_chain_history_common_train_v1.py")); B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
s = importlib.util.spec_from_file_location("tc1c", os.path.join(HERE, "tc1c_chain_ablation_v1.py")); C = importlib.util.module_from_spec(s); s.loader.exec_module(C)
rm = T.rm
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
IN23 = _X[_X.h == 23].set_index("rec").in_temp; EVE = _X[_X.h >= 18].groupby("rec").in_temp.mean()
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_mirror.csv")).set_index("rec")
CT = T.chain_table(); CT2 = CT.copy(); rep = 0
for i, r in CT2.iterrows():
    k = f"{r.farm}_{int(r.day):03d}"
    if k in PM.index and isinstance(PM.at[k, "prev0"], str) and PM.at[k, "p0"] >= .5:
        pv = PM.at[k, "prev0"]
        if pv in IN23.index: CT2.at[i, "ch_in23"] = IN23[pv]; CT2.at[i, "ch_eve"] = EVE[pv]; rep += 1
print(f"chain table rows {len(CT2)} | replaced by v15 pick {rep} | changed ch_in23 {(np.abs(CT2.ch_in23 - CT.ch_in23) > 1e-9).sum()}", flush=True)
D1 = T.layout_frame("DIAG10", CT); D2 = T.layout_frame("DIAG10", CT2)
rng = np.random.default_rng(20261118); ok = True; diffs = []
for L, (d, pre, bs, cx) in T.LAY.items():
    M1 = T.layout_frame(L, CT); M2 = T.layout_frame(L, CT2); y = M1.sub_temp.values; cells = []
    for bi, b in enumerate(bs):
        for fam in "AB":
            p0 = B.w40(M1, b, cx, fam); pc = C.corrected(M1, p0, D1, bi, fam, C.FB); pn = C.corrected(M2, p0, D2, bi, fam, C.FB)
            cells.append(rm(pn - y) / rm(pc - y) - 1)
            if bi == 0 and fam == "A" and L not in ("EL1", "EXT10", "EXT12"):
                key = M1.farm + "_" + (M1.day // 5).astype(str); diffs.append(pd.Series((pn - y) ** 2 - (pc - y) ** 2).groupby(key.values).mean())
    p2 = M1.day.values >= 179
    print("  %-6s B4 vs TC1c: %s" % (L, " ".join("%+.2f%%" % (100 * x) for x in cells)), flush=True)
    if L not in ("EXT10", "EXT12"): ok &= all(x < 0 for x in cells)
S = pd.concat(diffs, axis=1).mean(axis=1).values; idx = rng.integers(0, len(S), (20000, len(S))); P = float((S[idx].mean(1) >= 0).mean())
print(f"B4: all cells better {ok} | pooled P(worse) {P:.4f} ({len(S)} blocks) -> {'ADOPT' if ok and P < .025 else 'FAIL'}")
