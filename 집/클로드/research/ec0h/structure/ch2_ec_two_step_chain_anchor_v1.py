# -*- coding: utf-8 -*-
"""CH2 (2026-10-12 집 클로드; 사용자: 'EC 2단계 사슬 앵커 실험').  사전 고정(이 파일 커밋 후 실행).
아이디어(사용자): 오늘 R(평가) ← 하루 전 P1(평가) ← 이틀 전 P2(학습).  P1 은 P2 의 23시 정답으로 EA2 앵커를 받아 하루 종일 수준을 유지한 예측을 가짐
→ 그 P1 의 23시 예측을 R 의 0시 출발점(앵커 값 A)으로 써서 R 에도 EA2 공식 적용: q_h = SG2_h + (A − SG2 의 0..h 누적평균).
검증: EV(평가 기하, ev_folds_v1) 저장 행 out_v1/v2_ev_ec_rows.csv, BASE 계절, 시드 7/101/2024.  기준 = 제출 사슬 EA2(1단계 앵커만).
2단계 대상(폴드 k 마다): R ∈ val, R 은 EA2 앵커 없음; 짝찾기 v7 거울 CV 선택(cv_all400_choice_v7m) R→P1 p ≥ .8;
  P1 ∈ val(같은 폴드, 숨김 = 평가 기록 역할, 'del' 아님), P1 은 R 과 같은 농장 ID 이고 번호가 R 보다 작음(규정: 같은 ID 앞 번호 평가 입력만);
  P1 은 EA2 에서 앵커를 받은 기록(= P1→P2 p ≥ .8, P2 정답 있음, P2 가 val/del 아님).
두 안(본페로니 2안, α = .0125):
  C2P (사용자 안) A = P1 의 EA2 예측 23시 값(같은 시드).
  C2L (직접)     A = P2 의 23시 정답(= P1 의 anchor_ec23).
채택: 3시드 모두 EV 2차 행 RMSE 개선 AND 전체 val 행 RMSE 개선 AND 기록 단위 부트스트랩(4000, 2차 val 기록) P(악화) < .0125 (3시드 모두)
  AND 2차 이득 상위 1기록 몫 ≤ .5 (시드 평균).
서술: 대상 기록 수, 바뀐 기록별 오차, 대상이 진짜 같은 동 사슬인지(진단용 truth a1 로 R→P1, P1→P2 둘 다 맞나) 비율.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd
H = r"C:\work\farmai\집\클로드\research"
O = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv"))
O = O[(O.arm == "BASE") & (O.seed.astype(str) != "avg")].copy(); O["seed"] = O.seed.astype(str)
F = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv"))
CH = pd.read_csv(os.path.join(H, "local", "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
TL = pd.read_pickle(os.path.join(H, "local", "linker_v3", "truth.pkl"))
sys.path.insert(0, os.path.join(H, "linker_v4"))
from feat4 import FARM, NUM  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; TRUEPREV = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
out = []; targets = []
for k, Ok in O.groupby("fold"):
    f = F[F.fold == k]; val = set(f.rec[f.role == "val"]); dele = set(f.rec[f.role == "del"])
    anc = Ok.groupby("rec").anchored.first()
    for sd, G in Ok.groupby("seed"):
        G = G.sort_values(["rec", "hour"]).copy()
        P23 = G[G.hour == 23].set_index("rec").EA2
        for arm in ("C2P", "C2L"): G[arm] = G.EA2.values
        cm = G.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values
        for r in G.rec.unique():
            if anc.get(r, False) or r not in CH.index: continue
            c = CH.loc[r]; p1 = c.prev
            if not (c.p >= .8 and p1 in val and p1 not in dele and p1[:3] == r[:3] and int(p1[4:7]) < int(r[4:7]) and anc.get(p1, False)): continue
            c1 = CH.loc[p1]
            m = (G.rec == r).values
            for arm, A in (("C2P", float(P23[p1])), ("C2L", float(c1.anchor_ec23))):
                G.loc[m, arm] = G.SG2.values[m] + (A - cm[m])
            if sd == "7":
                targets.append(dict(fold=k, rec=r, p1=p1, p2=c1.prev, p_r=c.p, p_p1=c1.p, true_chain=(TRUEPREV.get(r) == p1) and (TRUEPREV.get(p1) == c1.prev), A_pred=float(P23[p1]), A_lab=float(c1.anchor_ec23)))
        out.append(G)
O2 = pd.concat(out); T = pd.DataFrame(targets)
pd.set_option("display.width", 220)
print(f"2-step targets: {len(T)} (pass-2 {int((T.rec.str[4:7].astype(int) >= 179).sum()) if len(T) else 0}); true same-house chain {T.true_chain.mean() if len(T) else float('nan'):.2f}")
lo, hi = O2.y.min(), O2.y.max()
for arm in ("C2P", "C2L"): O2[arm] = O2[arm].clip(lo, hi)   # EV chain clips to fold-training range; global clip only guards extremes
rng = np.random.default_rng(20261012); rows = []
for sd, g in O2.groupby("seed"):
    g2 = g[g.day >= 179]
    for arm in ("C2P", "C2L"):
        E = pd.DataFrame({"rec": g2.rec, "b": (g2.EA2 - g2.y) ** 2, "s": (g2[arm] - g2.y) ** 2}).groupby("rec").agg(b=("b", "sum"), s=("s", "sum"), n=("b", "size"))
        idx = rng.integers(0, len(E), (4000, len(E))); pw = float((E.s.values[idx].sum(1) >= E.b.values[idx].sum(1)).mean())
        gain = (E.b - E.s); share = float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan
        rows.append(dict(seed=sd, arm=arm, p2_base=rm(g2.EA2 - g2.y), p2_arm=rm(g2[arm] - g2.y), all_base=rm(g.EA2 - g.y), all_arm=rm(g[arm] - g.y), P_worse=pw, top_share=share, changed=int((gain.abs() > 1e-12).sum())))
R = pd.DataFrame(rows); R["p2_%"] = 100 * (R.p2_arm / R.p2_base - 1); R["all_%"] = 100 * (R.all_arm / R.all_base - 1)
print(R.round(4).to_string(index=False))
for arm in ("C2P", "C2L"):
    S = R[R.arm == arm]
    ok = (S.p2_arm < S.p2_base).all() and (S.all_arm < S.all_base).all() and (S.P_worse < .0125).all() and S.top_share.mean() <= .5
    print(f"{arm}: {'ADOPT' if ok else 'FAIL'}")
if len(T):
    g = O2[O2.seed == "7"]; e = g.groupby("rec").apply(lambda d: pd.Series({"y": d.y.mean(), "EA2": d.EA2.mean(), "C2P": d.C2P.mean(), "C2L": d.C2L.mean(),
        "rmse_EA2": rm(d.EA2 - d.y), "rmse_C2P": rm(d.C2P - d.y), "rmse_C2L": rm(d.C2L - d.y)}))
    print(T.set_index("rec").join(e).round(3).to_string())
