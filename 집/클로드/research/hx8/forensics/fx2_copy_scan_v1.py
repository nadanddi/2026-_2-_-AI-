# -*- coding: utf-8 -*-
# hx8/forensics fx2: 복제·시간 이동 복제 흔적 검사(서술).
# 가설 H1: 주최 측이 평가(순수 창) 기록 입력을 다른 기록 입력에서 통째/구간 복사(시간 이동 포함)해 만들었다.
#   반드시 보일 흔적: 실내 3변수(in_temp 0.1, in_hum, in_co2 정수)가 L시간 연속 '정확히' 같은 창이 다른 기록에 있음(우연 확률 ~0).
#   보이면 안 되는 것: 학습-학습 기준선과 같은 빈도(우연 수준).
#   기존과 차이: 1.11(다른 온실 in_temp 일치율), 3.1(24시간 벡터 중복 0%), 6.153(6시간·상수 차 허용 창, 우연 1.22배)
#   → 여기서는 상수 차 없이 정확 일치, 짧은 창(L=3,4,6), 모든 시각 이동, 51개 온실 학습 + F13/F47 학습, 8기록 집중.
#   그리고 24시간 상관 ≥ .999(시각 이동 −6..+6, 겹침 ≥ 18시간)도 F13/F47 학습 상대 검사.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from collections import defaultdict

T8 = ["F13_221", "F13_223", "F13_225", "F13_237", "F47_221", "F47_223", "F47_225", "F47_227"]
tr = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "in_hum", "in_co2"])
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id", "in_temp", "in_hum", "in_co2"])
def arr(d):
    d = d.copy(); d["rec"] = d.row_id.str[:7]; d["h"] = d.row_id.str[8:].astype(int)
    recs = d.rec.drop_duplicates().tolist(); ix = {r: k for k, r in enumerate(recs)}
    A = np.full((len(recs), 24, 3), np.nan)
    A[d.rec.map(ix).values, d.h.values] = d[["in_temp", "in_hum", "in_co2"]].values
    return recs, A
rtr, Atr = arr(tr); rte, Ate = arr(te)
print("train records", len(rtr), "eval records", len(rte))
# integer keys: temp*10, hum*10, co2*10 (robust to float)
def keyarr(A): return np.round(A * 10).astype("float")
Ktr, Kte = keyarr(Atr), keyarr(Ate)
res = []
for L in (3, 4, 6):
    idx = defaultdict(list)
    for i in range(len(rtr)):
        for s in range(0, 25 - L):
            w = Ktr[i, s:s + L]
            if np.isnan(w).any(): continue
            idx[w.tobytes()].append((i, s))
    def query(K, i, own=None):
        hits = []
        for s in range(0, 25 - L):
            w = K[i, s:s + L]
            if np.isnan(w).any(): continue
            for (j, s2) in idx.get(w.tobytes(), []):
                if own is not None and j == own: continue
                const = bool((np.ptp(w, 0) == 0).all())
                hits.append((s, j, s2, const))
        return hits
    # eval 8 targets
    for r in T8:
        i = rte.index(r); h = query(Kte, i)
        nc = [x for x in h if not x[3]]
        res.append(dict(L=L, rec=r, kind="EVAL8", hits=len(h), nonconst=len(nc),
                        ex=";".join(f"h{s}->{rtr[j]}@h{s2}" for s, j, s2, c in nc[:5])))
    # baseline: all 60 eval and F13/F47 train records vs train (excluding self)
    for kind, recs, K, own in (("EVAL60", rte, Kte, False), ("TRAIN_F13F47", [r for r in rtr if r[:3] in ("F13", "F47")], Ktr, True)):
        tot = ntot = nrec = 0
        for r in recs:
            i = (rte if kind == "EVAL60" else rtr).index(r)
            h = query(K, i, own=i if own else None)
            nc = [x for x in h if not x[3]]
            tot += len(h); ntot += len(nc); nrec += 1 if nc else 0
        res.append(dict(L=L, rec=f"(n={len(recs)})", kind=kind, hits=tot, nonconst=ntot, ex=f"records with any nonconst hit: {nrec}"))
R = pd.DataFrame(res)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(R.to_string(index=False))
# correlation >= .999 with shifts, vs F13/F47 train (in_temp & in_hum & in_co2 all), overlap >= 18 h
fi = [k for k, r in enumerate(rtr) if r[:3] in ("F13", "F47")]
print("\n24h shifted correlation scan vs F13/F47 train (all 3 indoor vars r >= .999, overlap >= 18h)")
best = []
for r in T8 + ["__base__"]:
    targets = [rte.index(r)] if r != "__base__" else None
    if r == "__base__":
        # baseline: F13/F47 train vs F13/F47 train
        cnt = 0; pairs = 0
        for a in fi:
            for b in fi:
                if a >= b: continue
                for sh in range(-6, 7):
                    x = Atr[a, max(0, sh):24 + min(0, sh)]; y = Atr[b, max(0, -sh):24 + min(0, -sh)]
                    ok = ~np.isnan(x).any(1) & ~np.isnan(y).any(1)
                    if ok.sum() < 18: continue
                    pairs += 1
                    c = [np.corrcoef(x[ok, v], y[ok, v])[0, 1] for v in range(3)]
                    if np.nanmin(c) >= .999: cnt += 1
        print(f"  baseline train-train pairs x shifts {pairs}, hits {cnt}")
        continue
    i = targets[0]; top = []
    for b in fi:
        for sh in range(-6, 7):
            x = Ate[i, max(0, sh):24 + min(0, sh)]; y = Atr[b, max(0, -sh):24 + min(0, -sh)]
            ok = ~np.isnan(x).any(1) & ~np.isnan(y).any(1)
            if ok.sum() < 18: continue
            with np.errstate(invalid="ignore"):
                c = [np.corrcoef(x[ok, v], y[ok, v])[0, 1] for v in range(3)]
            top.append((np.nanmin(c), rtr[b], sh))
    top.sort(reverse=True)
    print(f"  {r}: best min-corr {top[0][0]:.4f} ({top[0][1]} shift {top[0][2]}), 2nd {top[1][0]:.4f} ({top[1][1]}), hits>=.999: {sum(t[0] >= .999 for t in top)}")
