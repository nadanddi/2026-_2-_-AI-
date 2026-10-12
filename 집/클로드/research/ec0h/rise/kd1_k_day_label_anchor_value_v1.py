# -*- coding: utf-8 -*-
"""KD1 (2026-10-12 집 클로드; 사용자: '순수 2차 창 사슬의 끝 연결 강화').  서술(같은 동 식별 완벽 가정 = 상한).
순수 창 F13 사슬: F13_217(80일, 학습, 23시 1.663) → [81일 지워짐] → F13_221(82, 평가) → F13_223(83) → F13_225(84).  끝 연결은 GR2(6.676) 지워진 날 건너뛰기.
F13_223·225 는 정답에서 3·4일 거리 — CH3(6.694)는 2일만 검증.  질문: 같은 동 k 일 전 23시 정답을 오늘 기록 전체 수준(EA2 공식)으로 쓰면 모델보다 나은가, k = 1..5.
학습 기록마다 진단용 연결로 k 단계 뒤 같은 동 기록(날짜 정확히 d−k, 정답 있음)을 찾고, A = 그 기록 23시 정답.
  q = DIAG10 OOF(3시드 평균), 안 = q_h + β(A − q 의 0..h 누적평균), β = 1 과 최소제곱 β.  보고: 전체·고EC(하루 평균 ≥ 1)별 기록 RMSE 평균, 개선 비율, β.
이어서 평가 적용 미리보기(서술): 12회차 예측에 k 별 β 로 F13_221(k 2)·223(k 3)·225(k 4) 를 당겼을 때 하루 평균 변화.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
TL = pd.read_pickle(S3 + r"\truth.pkl"); pre = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
Q = O.groupby(["rec", "hour"]).sg.mean().unstack()
rows = []
for r in Q.index:
    d = DAY.get(r); x = r
    if d is None or not np.isfinite(d): continue
    y = EC[idx[r]]; q = Q.loc[r].values; cm = np.cumsum(q) / np.arange(1, 25)
    for k in range(1, 6):
        x = pre.get(x) if x else None
        if x is None or x not in idx: break
        if DAY.get(x) != d - k or not TRAIN[idx[x]]: continue
        A = float(EC[idx[x], 23]); rows.append(dict(rec=r, k=k, hi=float(np.nanmean(y)) >= 1, y=y, q=q, g=A - cm))
D = pd.DataFrame(rows); BETA = {}
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
pd.set_option("display.width", 200)
for (k, hi), G in D.groupby(["k", "hi"]):
    Y = np.stack(G.y); Qm = np.stack(G.q); Gm = np.stack(G.g); R = Y - Qm
    b = float(np.nansum(R * Gm) / np.nansum(Gm * Gm)); BETA[(k, hi)] = b
    e0 = np.array([rm(R[i]) for i in range(len(G))]); e1 = np.array([rm(R[i] - Gm[i]) for i in range(len(G))]); eb = np.array([rm(R[i] - b * Gm[i]) for i in range(len(G))])
    print(f"k {k} {'high' if hi else 'norm'} n {len(G):3d} | record RMSE mean model {e0.mean():.3f} | full anchor (beta 1) {e1.mean():.3f} better {np.mean(e1 < e0):.2f} | LS beta {b:.2f} -> {eb.mean():.3f}")
P = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int)
PP = P.pivot(index="rec", columns="h", values="sub_ec"); A = float(EC[idx["F13_217"], 23])
print(f"\neval preview (A = F13_217 23h {A:.3f}; beta = high-EC LS beta for that k, else 'n/a'):")
for r, k in (("F13_221", 2), ("F13_223", 3), ("F13_225", 4)):
    q = PP.loc[r].values; cm = np.cumsum(q) / np.arange(1, 25); b = BETA.get((k, True), np.nan)
    new = q + b * (A - cm) if np.isfinite(b) else q
    print(f"   {r} (k {k}): 12th day mean {q.mean():.3f}, 0h {q[0]:.3f} | beta {b:.2f} -> day mean {new.mean():.3f} (change {new.mean() - q.mean():+.3f})")
