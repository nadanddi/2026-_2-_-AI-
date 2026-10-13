# -*- coding: utf-8 -*-
"""CP1 (2026-10-14 집 클로드; 사용자: 'RNN 학습 전 전처리 — 날짜 순서 & 이어지는 순서, 동 구분').  전처리·서술(판정 아님).
목적: 가변 길이 사슬 RNN(같은 동 연속 구간만큼 이어서 학습)의 입력 표 — 기록마다 날짜·사슬(같은 동 연속 구간)·사슬 안 위치·앞뒤 연결·운영 지문 유형.
1. 날짜: 학습 기록 = 학습 전용 달력 DMT3e (구간 seg, 날짜 day).  평가 기록 = 0시 외기 4칸이 같은 학습 기록들의 DMT3e 최빈 (서술용; 실제 예측 땐 규정대로).
2. 학습 기록끼리 연결 = truth.pkl cert(학습 정답 EC·배지온 연속으로 정한 확실한 진짜 전날, 학습 자료만).  한 기록을 둘 이상이 전날로 지목하면 비용(c1)이 가장 낮은 하나만.
3. 연결 점검: date_ok = 전날 날짜 == 대상 날짜 − 1(같은 구간);  fp = 하루 운영 지문 거리(구동기·CO2 8열, 학습 표준화, 평균 |차| 4 로 자름);
   끊기: date_ok 가 거짓으로 확인되거나(둘 다 날짜 있음) fp > 학습 cert 쌍 fp 의 95% 분위 → 연결 버리고 새 사슬.
4. 사슬 = 남은 연결을 따라간 같은 동 연속 구간(동 구분은 '모델 분리' 가 아니라 '사슬 분리').  사슬 번호·위치(0 = 시작), 시작이 끊김인지(앞 기록 있으나 연결 버림/없음).
5. 운영 지문 유형: 학습 기록 하루 지문(24 x 8) 을 KMeans(k = 4, 학습 기록만으로 적합, random_state 0) — 동의 운영 성격 표시(RNN 입력 후보).
6. 평가 기록(서술): 규정상 이을 수 있는 앞 기록 = 학습 기록 또는 같은 ID 앞 번호 평가 기록, 날짜 d−1·같은 구간 후보 중 하루 지문 최근접(확신 2등/1등 >= 1.5) — 예측 땐 시각별로 다시 계산.
출력: cp1_records_v1.csv(rec, train, day, seg, chain, pos, prev, next, link_date_ok, link_fp, chain_len, start_broken, fp_type, high), 요약 출력.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
LAB = TRAIN & np.isfinite(DM)
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
def tdate(b):
    vi = np.where(LAB)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return np.nan, np.nan
    d = pd.Series(DAYT[m]).mode().iloc[0]; return d, pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
DAY, SEG = DAYT.copy(), SEGT.copy()
for i in range(N):
    if not TRAIN[i]: DAY[i], SEG[i] = tdate(i)
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
fpd = lambda a, b: float(np.nanmean(np.minimum(np.abs(FPA[a] - FPA[b]), 4)))
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TLc = TL[TL.cert].sort_values("c1")
prev, used = {}, set()
for a, b in zip(TLc.a1.astype(int), TLc.b.astype(int)):
    if b in prev or a in used or not (LAB[a] and LAB[b]): continue
    prev[b] = a; used.add(a)
fps = np.array([fpd(a, b) for b, a in prev.items()]); FPCUT = float(np.quantile(fps, .95))
date_ok = {b: (np.nan if (np.isnan(DAY[a]) or np.isnan(DAY[b])) else float(DAY[a] == DAY[b] - 1 and SEG[a] == SEG[b])) for b, a in prev.items()}
cut = {b for b, a in prev.items() if date_ok[b] == 0.0 or fpd(a, b) > FPCUT}
link = {b: a for b, a in prev.items() if b not in cut}; nxt = {a: b for b, a in link.items()}
print(f"cert links usable {len(prev)} | cut: date mismatch {sum(date_ok[b] == 0.0 for b in prev)}, fingerprint > {FPCUT:.3f}: {sum(fpd(a, b) > FPCUT for b, a in prev.items())} | kept {len(link)}", flush=True)
chain, pos = {}, {}; cid = 0
for i in np.where(LAB)[0]:
    if i in link: continue
    cur, p = i, 0
    while True:
        chain[cur], pos[cur] = cid, p
        if cur not in nxt: break
        cur, p = nxt[cur], p + 1
    cid += 1
km = KMeans(4, random_state=0, n_init=10).fit(np.nan_to_num(FPA[LAB].reshape(LAB.sum(), -1)))
fpt = np.full(N, -1); fpt[LAB] = km.labels_
rest = np.where(~LAB)[0]; fpt[rest] = km.predict(np.nan_to_num(FPA[rest].reshape(len(rest), -1)))
rows = []
for i in range(N):
    r = rid(i)
    rows.append(dict(rec=r, train=bool(LAB[i]), day=DAY[i], seg=SEG[i], chain=chain.get(i, -1), pos=pos.get(i, -1), prev=rid(link[i]) if i in link else None, next=rid(nxt[i]) if i in nxt else None,
                     link_date_ok=date_ok.get(i, np.nan) if i in link else np.nan, link_fp=fpd(link[i], i) if i in link else np.nan, start_broken=bool(i in prev and i not in link),
                     fp_type=int(fpt[i]), high=bool(LAB[i] and DM[i] >= 1.0), dm=float(DM[i]) if LAB[i] else np.nan))
R = pd.DataFrame(rows); R["chain_len"] = R.groupby("chain").rec.transform("size").where(R.chain >= 0, 0)
# eval attachment (descriptive)
ev = []
for i in rest:
    d, s = DAY[i], SEG[i]; f = FARM[i]
    if np.isnan(d): ev.append(dict(rec=rid(i), date=False)); continue
    cand = [a for a in range(N) if a != i and DAY[a] == d - 1 and SEG[a] == s and (LAB[a] or (not TRAIN[a] and FARM[a] == f and NUM[a] < NUM[i]))]
    if not cand: ev.append(dict(rec=rid(i), date=True, ncand=0)); continue
    dd = sorted((fpd(a, i), a) for a in cand); conf = len(dd) == 1 or dd[1][0] >= 1.5 * max(dd[0][0], 1e-9)
    ev.append(dict(rec=rid(i), date=True, ncand=len(cand), pick=rid(dd[0][1]), pick_labelled=bool(LAB[dd[0][1]]), conf=conf, fp=dd[0][0]))
E = pd.DataFrame(ev)
R.to_csv(os.path.join(HERE, "cp1_records_v1.csv"), index=False); E.to_csv(os.path.join(HERE, "cp1_eval_attach_v1.csv"), index=False)
T = R[R.train]; L = T.drop_duplicates("chain").chain_len
print(f"\ntraining records {len(T)} -> chains {T.chain.nunique()} | length distribution: {L.value_counts().sort_index().to_dict()}")
print(f"records in chains of length >= 2: {int((T.chain_len >= 2).sum())} ({(T.chain_len >= 2).mean():.2f}) | >= 3: {int((T.chain_len >= 3).sum())} | >= 5: {int((T.chain_len >= 5).sum())}")
print("links crossing ID (F13<->F47):", int(sum(FARM[a] != FARM[b] for b, a in link.items())), "of", len(link))
H = T[T.high]; print(f"high-EC records {len(H)}: in chains len>=2 {int((H.chain_len >= 2).sum())}; high episodes (consecutive high within a chain): {int(((T.sort_values(['chain', 'pos']).groupby('chain').high.apply(lambda s: ((s) & (~s.shift(fill_value=False))).sum()))).sum())}")
print("pass-2 training records in chains len>=2:", int(((T.rec.str[4:].astype(int) >= 179) & (T.chain_len >= 2)).sum()), "of", int((T.rec.str[4:].astype(int) >= 179).sum()))
print("fingerprint types (train counts):", T.fp_type.value_counts().sort_index().to_dict(), "| high-EC share by type:", T.groupby("fp_type").high.mean().round(2).to_dict())
print(f"\neval records {len(E)}: date found {int(E.date.sum())} | candidate exists {int((E.ncand > 0).sum()) if 'ncand' in E else 0} | confident pick {int(E.conf.fillna(False).sum()) if 'conf' in E else 0} (labelled {int((E.conf.fillna(False) & E.pick_labelled.fillna(False)).sum()) if 'conf' in E else 0})")
