# -*- coding: utf-8 -*-
"""INC4 (2026-10-13 집 클로드, hx8/increment; 이어서 하기).  서술(채택 판정 아님; 제출물 아님).
INC3 은 마지막 줄(F13_237 출력, cp949 로 em-dash 인코딩 오류)에서 멈춤 — 나머지 (a)~(g) 는 inc3_v1.log 에 있음.
여기서 하는 것:
 (1) F13_237 미리보기 줄 완성: F13_233 23시 정답(학습), F13_235(같은 ID 앞 번호 평가, 13회차 예측 하루 평균), F13_237 13회차.
 (2) INC3(f) '보통 앵커(<1) -> 모델이 고EC(>=.9)로 본 k=2 대상' 을 F13_237 상황에 더 가깝게:
     사이 날(k=1) 모델 예측도 보통(<.9)인 사례만 (F13_235 13회차 .651 흉내). k=1·2·3 모두 보고(일관성).
     추정량: M(현재 모델 OOF, DIAG10 3시드 평균), P(앵커 23시), H(반반), L(M + lam·(P - M), lam 은 에피소드 밖 사례로 최소제곱 0..1).
     에피소드 부트스트랩 95% 구간, 시드별 방향(3시드), 사례 목록.
 (3) 사례별 위치 비율 r = (y - P)/(M - P) (0 = 앵커가 맞음, 1 = 모델이 맞음) 분포.
규정: 학습 기록(정답·OOF)만 사용. 평가 쪽은 13회차 제출 하루 평균(이미 만든 예측)과 학습 F13_233 정답만 읽어 미리보기(계수 추정에 안 씀).
DM11 미사용(학습 전용 달력 dmt3e_E0 만).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import EC, FARM, NUM, N, TRAIN, S3
import numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
T1 = pd.read_csv(os.path.join(here, "inc1_targets_v1.csv"), index_col=0); y, comp = T1.y, T1.comp
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
Ms = O.groupby(["seed", "rec"]).sg.mean().unstack(0); seeds = list(Ms.columns); Mm = Ms.mean(axis=1)
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
def consec(a, b): return a in CAL.index and b in CAL.index and CAL.seg[a] == CAL.seg[b] and CAL.day[b] == CAL.day[a] + 1
succ = {}
for a, b in zip(TL.a1, TL.b):
    a, b = rid(int(a)), rid(int(b))
    if a in y.index and b in y.index and consec(a, b): succ.setdefault(a, []).append(b)
rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
S = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv")); S["rec"] = S.row_id.str[:7]
p13 = S.groupby("rec").sub_ec.mean()

print("(1) F13_237 preview inputs")
print(f"   F13_233 (train) day mean {np.nanmean(EC[idx['F13_233']]):.3f}, 23h {EC[idx['F13_233'], 23]:.3f} | F13_235 13th day mean {p13['F13_235']:.3f} | F13_237 13th day mean {p13['F13_237']:.3f}")

def build(k, mid_normal):
    rows = []
    for a in y.index:
        if y[a] >= 1.0: continue
        ps = [[a]]
        for _ in range(k): ps = [p + [q] for p in ps for q in succ.get(p[-1], [])]
        for p in ps:
            t = p[-1]
            if t not in Mm.index or Mm[t] < 0.9: continue
            mids = p[1:-1]
            if mid_normal and any((m_ not in Mm.index) or Mm[m_] >= 0.9 for m_ in mids): continue
            rows.append(dict(k=k, a=a, mid=",".join(mids), t=t, anc=EC[idx[a], 23], yt=y[t], m=Mm[t], ep=comp[a],
                             **{f"m{s}": Ms[s].get(t, np.nan) for s in seeds}))
    return pd.DataFrame(rows)

rng = np.random.default_rng(13)
for mid_normal in (False, True):
    print(f"\n(2) normal anchor (<1, 23h) -> model-high target (>=.9){' with intermediate model < .9' if mid_normal else ''}")
    for k in (1, 2, 3):
        G = build(k, mid_normal)
        if len(G) == 0: print(f"   k{k}: n 0"); continue
        G = G.drop_duplicates(["a", "t"])  # same (anchor, target) via different paths counted once
        lam = np.full(len(G), np.nan)
        for j, (_, c) in enumerate(G.iterrows()):
            tr = G[G.ep != c.ep]
            if len(tr) >= 3:
                u = (tr.anc - tr.m).values; v = (tr.yt - tr.m).values; lam[j] = float(np.clip(u @ v / (u @ u + 1e-12), 0, 1))
            else:
                lam[j] = 0.0
        G["L"] = G.m + lam * (G.anc - G.m); G["H"] = (G.m + G.anc) / 2; G["lam"] = lam
        eps = G.ep.unique(); line = []
        for X_ in ["anc", "H", "L"]:
            base = rm(G[X_], G.yt) - rm(G.m, G.yt); bs = []
            for _ in range(2000):
                gg = pd.concat([G[G.ep == e] for e in rng.choice(eps, len(eps))]); bs.append(rm(gg[X_], gg.yt) - rm(gg.m, gg.yt))
            nm = {"anc": "P", "H": "H", "L": "L"}[X_]
            line.append(f"{nm} {base:+.3f} [{np.quantile(bs, .025):+.3f},{np.quantile(bs, .975):+.3f}] win {np.mean(np.abs(G[X_] - G.yt) < np.abs(G.m - G.yt)):.2f}")
        seedline = " ".join(f"s{s}:P{rm(G.anc, G.yt) - rm(G[f'm{s}'], G.yt):+.3f}/H{rm((G[f'm{s}'] + G.anc) / 2, G.yt) - rm(G[f'm{s}'], G.yt):+.3f}" for s in seeds)
        r = (G.yt - G.anc) / (G.m - G.anc)
        print(f"   k{k}: n {len(G)} eps {len(eps)} truly-high {int((G.yt >= 1).sum())} | mean y {G.yt.mean():.3f} M {G.m.mean():.3f} anchor {G.anc.mean():.3f} | RMSE M {rm(G.m, G.yt):.3f} P {rm(G.anc, G.yt):.3f} H {rm(G.H, G.yt):.3f} L {rm(G.L, G.yt):.3f} (lam mean {np.mean(lam):.2f})")
        print(f"        diff vs M: " + " | ".join(line))
        print(f"        per seed (RMSE X - RMSE M_seed): {seedline}")
        print(f"        position r=(y-P)/(M-P): median {np.median(r):+.2f}, quartiles [{np.quantile(r, .25):+.2f}, {np.quantile(r, .75):+.2f}], share r<.5 {np.mean(r < .5):.2f}; gap M-P mean {np.mean(G.m - G.anc):+.3f} max {np.max(G.m - G.anc):+.3f}")
        if k == 2:
            print(G[["a", "mid", "t", "ep", "anc", "yt", "m", "H", "L"]].round(3).to_string(index=False))
            if mid_normal:
                lam_all = float(np.clip(((G.anc - G.m) @ (G.yt - G.m)) / ((G.anc - G.m) @ (G.anc - G.m) + 1e-12), 0, 1))
                a237 = EC[idx["F13_233"], 23]; m237 = p13["F13_237"]
                print(f"   F13_237 preview (k2, anchor F13_233 23h {a237:.3f}, mid F13_235 13th {p13['F13_235']:.3f}): 13th {m237:.3f} | H {(m237 + a237) / 2:.3f} | lam(all cases) {lam_all:.2f} -> L {m237 + lam_all * (a237 - m237):.3f}  [gap 13th-anchor {m237 - a237:+.3f}; training gaps above]")
