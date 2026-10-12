# -*- coding: utf-8 -*-
"""INC5 (2026-10-13 집 클로드, hx8/increment; 이어서 하기).  서술(채택 판정 아님; 제출물 아님).
INC2~4 가 안 본 평가 상황: '뒤쪽 보통 앵커'.
  번호 -2 = 같은 동 전날(6.549) 구조로 F13 홀수 사슬은 217(학습 1.69) -> 219(지움) -> 221 -> 223 -> 225 -> 227 -> 229 -> 231(학습 .621, 0시 .581).
  13회차는 221~225 를 고EC(1.6~1.8), 227·229 를 보통(.77·.74)으로 봄 -> F13_225 는 k=3 뒤에 보통 학습 정답(F13_231 0시)이 있음.
  F47 홀수 사슬 215(학습 .705, 23시 .709) -> 217(지움) -> 219(평가 .662) -> 221(1.401) -> ... -> 229(학습 1.79) : F47_221 은 앞 k=3 보통 앵커도 있음.
질문: 학습 가짜 사례에서 '뒤 k 걸음 보통 앵커(0시, 하루 평균 < 1) -> 모델이 고EC(>=.9)로 본 대상' 이면 앵커가 모델보다 나은가?
  사이 날 모델 예측이 보통(<.9)인 사례(F13_227·229 흉내)도 따로.
추정량: M(DIAG10 3시드 평균 OOF), P(앵커 0시), H(반반), L(M + lam(P-M), lam 은 에피소드 밖 최소제곱 0..1). 에피소드 부트스트랩 95% 구간, 시드별 방향.
끝에 평가 8기록별 '학습 가짜 사례가 지지하는 것' 표(13회차 하루 평균 대비; 미리보기, 계수 추정에 평가 안 씀).
규정: 학습 기록 정답·OOF 만으로 계수. 평가는 13회차 제출 하루 평균과 학습 앵커만 읽음. 사슬은 기록 번호(공식 시간 순서) 구조로만 정함. DM11 미사용.
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
prev = {}
for a, b in zip(TL.a1, TL.b):
    a, b = rid(int(a)), rid(int(b))
    if a in y.index and b in y.index and consec(a, b): prev.setdefault(b, []).append(a)
rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
S = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv")); S["rec"] = S.row_id.str[:7]
p13 = S.groupby("rec").sub_ec.mean()
rng = np.random.default_rng(21)

def build(k, mid_normal):
    rows = []
    for b in y.index:
        if y[b] >= 1.0: continue                      # later-day anchor is a normal day
        ps = [[b]]
        for _ in range(k): ps = [p + [q] for p in ps for q in prev.get(p[-1], [])]
        for p in ps:
            t = p[-1]
            if t not in Mm.index or Mm[t] < 0.9: continue
            mids = p[1:-1]
            if mid_normal and any((m_ not in Mm.index) or Mm[m_] >= 0.9 for m_ in mids): continue
            rows.append(dict(k=k, b=b, mid=",".join(mids), t=t, anc=EC[idx[b], 0], yt=y[t], m=Mm[t], ep=comp[b], **{f"m{s}": Ms[s].get(t, np.nan) for s in seeds}))
    return pd.DataFrame(rows)

for mid_normal in (False, True):
    print(f"\n(A) backward NORMAL anchor (later day mean <1, its 0h) -> model-high target (>=.9){' with intermediates model < .9' if mid_normal else ''}")
    for k in (1, 2, 3, 4):
        G = build(k, mid_normal)
        if len(G) == 0: print(f"   k{k}: n 0"); continue
        G = G.drop_duplicates(["b", "t"])
        lam = np.zeros(len(G))
        for j, (_, c) in enumerate(G.iterrows()):
            tr = G[G.ep != c.ep]
            if len(tr) >= 3:
                u = (tr.anc - tr.m).values; v = (tr.yt - tr.m).values; lam[j] = float(np.clip(u @ v / (u @ u + 1e-12), 0, 1))
        G["L"] = G.m + lam * (G.anc - G.m); G["H"] = (G.m + G.anc) / 2
        eps = G.ep.unique(); line = []
        for X_, nm in (("anc", "P"), ("H", "H"), ("L", "L")):
            base = rm(G[X_], G.yt) - rm(G.m, G.yt); bs = []
            for _ in range(2000):
                gg = pd.concat([G[G.ep == e] for e in rng.choice(eps, len(eps))]); bs.append(rm(gg[X_], gg.yt) - rm(gg.m, gg.yt))
            line.append(f"{nm} {base:+.3f} [{np.quantile(bs, .025):+.3f},{np.quantile(bs, .975):+.3f}] win {np.mean(np.abs(G[X_] - G.yt) < np.abs(G.m - G.yt)):.2f}")
        seedline = " ".join(f"s{s}:P{rm(G.anc, G.yt) - rm(G[f'm{s}'], G.yt):+.3f}/H{rm((G[f'm{s}'] + G.anc) / 2, G.yt) - rm(G[f'm{s}'], G.yt):+.3f}" for s in seeds)
        print(f"   k{k}: n {len(G)} eps {len(eps)} truly-high {int((G.yt >= 1).sum())} | mean y {G.yt.mean():.3f} M {G.m.mean():.3f} anchor {G.anc.mean():.3f} | RMSE M {rm(G.m, G.yt):.3f} P {rm(G.anc, G.yt):.3f} H {rm(G.H, G.yt):.3f} L {rm(G.L, G.yt):.3f} (lam mean {lam.mean():.2f})")
        print(f"        diff vs M: " + " | ".join(line))
        print(f"        per seed: {seedline}")
        if k >= 3:
            print(G[["b", "mid", "t", "ep", "anc", "yt", "m", "H"]].round(3).to_string(index=False))

print("\n(B) eval 8 records — chains by record number (6.549 structure); 13th = submission_22 day mean")
a231 = EC[idx["F13_231"], 0]; a215 = EC[idx["F47_215"], 23]; a229 = EC[idx["F47_229"], 0]; a217 = EC[idx["F13_217"], 23]; a233 = EC[idx["F13_233"], 23]
print(f"   anchors: F13_217 23h {a217:.3f} (high) | F13_231 0h {a231:.3f} (normal) | F13_233 23h {a233:.3f} (normal) | F47_215 23h {a215:.3f} (normal) | F47_229 0h {a229:.3f} (high)")
for r_ in ["F13_221", "F13_223", "F13_225", "F13_227", "F13_229", "F13_235", "F13_237", "F47_219", "F47_221", "F47_223", "F47_225", "F47_227"]:
    print(f"   {r_}: 13th day mean {p13[r_]:.3f}")
