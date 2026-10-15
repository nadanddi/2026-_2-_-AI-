# -*- coding: utf-8 -*-
"""AWX1 (2026-10-15 집 클로드; 사용자 '왜 소수만 이득인지, 그 구조만 가져올 수 없는지').  서술 진단(판정 없음), 실행 전 커밋.  EC.
대상: FL 확신 앵커(p ≥ .8·폴드 밖 정답 전날·|dT| ≤ 1.5)가 걸린 기록.  이득 g = SSE(강도 1) − SSE(강도 .75)  (> 0 = 줄이는 게 이득), 시드 평균, 기록 단위.
발견 판 = DIAG10 만(규칙을 찾는 곳).  P2LOO·EL1 은 같은 표를 '서술만' — 규칙 판정은 다음 사전 고정 시험에서.
설명 변수(진단용 정답):  delta = 오늘 하루 평균 EC − A23 (어제 23시),  |delta|.
예측 시점 단서(0시에 알 수 있음, 현장형):
  p(링크 확률), dT(자정 실내 변화), gap0 = A23 − NH_0, A23 수준, NH_0,
  ytrend = A23 − 어제 하루 평균 EC(어제 정답: 하루 끝에 올라가 있었나), yslope = A23 − 어제 18시 EC,
  dout = 오늘 0시 외기 − 어제 23시 외기, yrad = 어제 일사 합(학습 입력), yin = 어제 실내 평균.
출력: 각 단서와 g 의 Spearman, g > 0 기록과 g ≤ 0 기록의 단서 중앙값 비교, |delta| 와 g 의 관계.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u awx1_shrink_winner_diagnosis_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "cb2_ec_anchor_shaping_v1.py"), encoding="utf-8").read()
exec(src[:src.index("WS = tuple(")])
__file__ = os.path.join(HERE, "awx1_shrink_winner_diagnosis_v1.py")
from scipy.stats import spearmanr
ECL = {r: EC_[i] for r, i in idx_.items()}
_XA = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _XA = _XA[_XA.row_id.str[:3].isin(["F13", "F47"])]
_XA["rec"] = _XA.row_id.str[:7]; _XA["h"] = _XA.row_id.str[8:].astype(int)
OUTT = _XA.pivot(index="rec", columns="h", values="out_temp"); RAD = _XA.pivot(index="rec", columns="h", values="out_rad")
rows = []
for s in SEEDS:
    x0 = (R[f"A1_{s}"].values - .1 * mt) / .9
    for (v, k, r), ii in GRP:
        if r not in FL3 or FL3[r][0] in FOLDREC[(v, k)] or FL3[r][1] < .8 or DT[r] > 1.5: continue
        pv = FL3[r][0]; x = x0[ii]; y = R.y.values[ii]; pm = np.cumsum(x) / np.arange(1, len(x) + 1); A = float(EC_[idx_[pv], 23])
        f1, f75 = x + (A - pm), x + .75 * (A - pm); ye = ECL[pv]
        rows.append(dict(val=v, seed=s, rec=r, p2=bool(R.p2.values[ii][0]), g=float(((f1 - y) ** 2).sum() - ((f75 - y) ** 2).sum()), delta=float(y.mean() - A),
                         p=FL3[r][1], dT=DT[r], gap0=A - x[0], A=A, nh0=x[0], ytrend=A - np.nanmean(ye), yslope=A - ye[18] if np.isfinite(ye[18]) else np.nan,
                         dout=OUTT.at[r, 0] - OUTT.at[pv, 23], yrad=RAD.loc[pv].sum(min_count=1), yin=TIN.loc[pv].mean()))
D = pd.DataFrame(rows).groupby(["val", "rec"]).mean(numeric_only=True).reset_index(); D["absdelta"] = D.delta.abs(); D["absgap0"] = D.gap0.abs()
FE = ["absdelta", "p", "dT", "gap0", "absgap0", "A", "nh0", "ytrend", "yslope", "dout", "yrad", "yin"]
pd.set_option("display.width", 220)
for v in ("DIAG10", "P2LOO", "EL1"):
    g = D[D.val == v]; win = g.g > 0
    print(f"\n== {v}: anchored records {len(g)}, winners {int(win.sum())}, total gain {g.g.sum():+.3f} (winners {g.g[win].sum():+.3f}, losers {g.g[~win].sum():+.3f}){'  [discovery]' if v == 'DIAG10' else '  [describe only]'}")
    out = []
    for f in FE:
        c = g[[f, "g"]].dropna()
        out.append(dict(feature=f, spearman_g=spearmanr(c[f], c.g).correlation if len(c) > 4 else np.nan, med_win=g[f][win].median(), med_lose=g[f][~win].median()))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    print("  top winners:", g.nlargest(5, "g")[["rec", "g", "delta", "ytrend", "gap0"]].round(3).values.tolist())
D.to_csv(os.path.join(HERE, "awx1_rows_v1.csv"), index=False)
