# -*- coding: utf-8 -*-
"""TD7 (2026-10-12 집 클로드; 사용자: '계속 찾아보고 실험' — TD4 의 P .094 는 바뀐 2차 기록 10개라 검정력 부족 → 숨김 위치 10가지로 재판정).  사전 고정(이 파일 커밋 후 실행).
기하 o = 0..9: 거울 CV 덩어리 시작을 o 만큼 밀어 숨김(같은 규칙: 10기록 + 양옆 제거 + 다른 ID 거울 덩어리). 각 기하에서 2차 학습 기록(번호 ≥ 179)만 대상으로
  v7 짝(feat7 조건부 로짓, 덩어리마다 재학습, 0시)과 v11 짝(ID 특징 제거 + 달력 특징)을 새로 고르고, TD4 규칙으로 앵커 집합:
  BASE = v7 (p ≥ .8, 정답, |dT| ≤ 1.5);  TD4 = BASE ∪ v11(p ≥ .6, 정답, v7 없는 기록; |dT| > 1.5 면 차단 해제) − 날짜 불일치(대상 날짜 = 보이는 정답 기록 0시 외기 일치, 차 ≠ 1).
점수: DIAG10 틀(ta7.correct, 13회차 공식·BETA) 4칸, 2차 행. 기하마다 기록별 ΔSSE(4칸 평균).
판정(단일 안 α .025): 10기하 평균 2차 RMSE 변화 4칸 모두 음수 AND 기록 단위 부트스트랩(기록마다 10기하 평균 ΔSSE, 20000) P(악화) < .025 AND 잭나이프-1 같은 방향 AND 개선 기하 수 ≥ 8/10.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s); s.loader.exec_module(A7)
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
sys.path.insert(0, os.path.join(HERE, "linker_v10"))
_cwd = os.getcwd(); os.chdir(os.path.join(HERE, "linker_v10")); exec(open("v11_calendar_no_id_v1.py", encoding="utf-8").read().split("rows = []\nfor o in range(10):")[0])
os.chdir(_cwd)
import feat7 as F7m
TRG7 = F7m.train_groups7()
R = A7.A2.A1.frames(); M, cells = R["DIAG10"]; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; y = M.sub_temp.values; p2 = M.day.values >= 179
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def table(rows):
    return pd.DataFrame(rows, columns=["rec", "prev", "p", "prev_is_labelled", "anchor_st23", "dT_air", "steps_to_anchor"]) if rows else pd.DataFrame(columns=["rec", "prev", "p", "prev_is_labelled", "anchor_st23", "dT_air", "steps_to_anchor"])
per_off = []; rec_d = {}
for o in range(10):
    b7, t4 = [], []
    for f in ["F13", "F47"]:
        g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
        lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
        for s in range(lo - 10 + o, hi + 1, 10):
            blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
            tg = [b for b in blk if TRAIN[b] and NUM[b] >= 179]
            if not tg: continue
            rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
            m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
            touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
            w7 = fit_clogit([(X_, y_) for b_, a_, X_, y_ in TRG7 if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
            w11 = fit_clogit([(d[F11].values.astype(float), y_) for b_, a_, d, y_ in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
            lab = TRAIN.copy(); lab[touch] = False
            for b in tg:
                a0, p0 = F7m.pick7(b, 0, lab, rem2, w7)
                v7ok = a0 is not None and lab[a0] and p0 >= .8 and abs(IN_[b, 0, 0] - IN_[a0, 23, 0]) <= 1.5
                if v7ok:
                    row = (rid(b), rid(a0), 1.0, True, float(ST[a0, 23]), float(IN_[b, 0, 0] - IN_[a0, 23, 0]), 1); b7.append(row); t4.append(row); continue
                D = feats11(b, lab, rem2)
                if len(D) == 0: continue
                a = D.a.values.astype(int); sc = D[F11].values.astype(float) @ w11; p = np.exp(sc - sc.max()); p /= p.sum(); k = int(np.argmax(p)); pk = a[k]
                if not lab[pk] or p[k] < .6: continue
                db = date_of(b, lab & (np.arange(N) != b))
                if np.isfinite(db) and np.isfinite(DAY[pk]) and DAY[pk] != db - 1: continue
                dT = float(IN_[b, 0, 0] - IN_[pk, 23, 0])
                t4.append((rid(b), rid(pk), 1.0, True, float(ST[pk, 23]), 0.0 if abs(dT) > 1.5 else dT, 1))
    LB, LT = table(b7), table(t4)
    kb, Ab = A7.build(M, LB); kt, At = A7.build(M, LT); dp, ds = [], []
    for q in cells:
        bb = A7.correct(M, q, kb, Ab, .8, "const", "meas", 2, 1.5); tt = A7.correct(M, q, kt, At, .8, "const", "meas", 2, 1.5)
        dp.append(100 * (rm((tt - y)[p2]) / rm((bb - y)[p2]) - 1)); ds.append(pd.Series(((tt - y) ** 2 - (bb - y) ** 2)[p2]).groupby(key[p2]).sum())
    D = pd.concat(ds, axis=1).mean(axis=1)
    for r, v in D.items(): rec_d.setdefault(r, []).append(v)
    per_off.append(dict(o=o, base_anchors=len(LB), td4_anchors=len(LT), p2_pct=np.mean(dp), cells=" ".join(f"{x:+.2f}" for x in dp), changed=int((D.abs() > 1e-12).sum())))
    print(per_off[-1], flush=True)
P = pd.DataFrame(per_off); RD = pd.Series({r: np.mean(v) for r, v in rec_d.items()})
rng = np.random.default_rng(20261023); v = RD.values; Pw = float((v[rng.integers(0, len(v), (20000, len(v)))].sum(1) >= 0).mean()); jk = RD.drop(RD.idxmin()).sum()
cell_means = np.mean([[float(c) for c in s.split()] for s in P.cells], axis=0)
ok = (cell_means < 0).all() and Pw < .025 and jk < 0 and (P.p2_pct < 0).sum() >= 8
print(f"\nmean pass-2 change over offsets per cell {np.round(cell_means, 2).tolist()} | offsets improved {(P.p2_pct < 0).sum()}/10 | record bootstrap P {Pw:.4f} | jackknife {jk:+.3f} | records {len(RD)} (changed {(RD.abs() > 1e-12).sum()}) -> {'ADOPT' if ok else 'FAIL'}")
