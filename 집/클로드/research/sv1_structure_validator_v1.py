# -*- coding: utf-8 -*-
"""SV1 (2026-10-14 집 클로드; 사용자: '평가 구조 일치 검증기 + 상황별 판정 둘 다 묶어서 만들어봐').  검증 기반(판정 아님) — 1단계: 배치 생성 + 상황 표시 + 평가와 닮음 확인.
평가 구조(test_X 번호, 6.850): 농장마다 2차 번호 안 덩어리 4개(길이 5·10·10·5), 덩어리 바로 앞뒤 번호 1개씩 없음(지워짐), 그 너머 학습 기록.
가짜 평가 배치(draw):
  2차: 학습 기록 연속 구간(길이 L >= 3)마다 양끝 1개를 '지움(del)', 가운데 L−2 개를 '평가(val)' — 구조상 한 가지뿐(고정).
  1차: 농장마다 길이 5·10·10·5 덩어리를 겹치지 않게 무작위 위치(번호 8..178, 정답 있는 학습 기록이 연속인 곳)에 두고 앞뒤 1개씩 del.
  쌍둥이: val 기록마다 학습 달력(DMT3e) 같은 날짜·다른 ID 학습 기록을 확률 .28 로 'tw'(평가 역할 — 정답·입력 모두 못 씀)로 숨김(평가의 같은 날 다른 ID 평가 비율 6.578).
  보이는 정답 = 학습 & val·del·tw 아님.  입력 쓸 수 있는 기록(평가 규칙): 보이는 정답 기록 + 같은 ID 앞 번호 val 기록.
상황(숨긴 val 기록마다, 그 배치의 보이는 정보만으로 다시 계산):
  date_found = 0시 외기 일치 보이는 정답 기록으로 학습 달력 날짜를 찾음;  LKH(h) = 날짜 d−1·같은 구간 보이는 정답 후보 중 0..h 운영 지문 최근접이 확신(2등/1등 >= 1.5, 거리 <= tau_h);
  S1 짝 찾음 = LKH(9) 확신;  S2 짝 있음·못 찾음 = cert 전날이 보이는 정답인데 S1 아님;  S3 짝 없음 = 둘 다 아님;  twin = 같은 날짜 다른 ID 보이는 정답 기록 있음.
닮음 확인: 평가 60기록(보이는 정답 = 학습 전체, tw 없음 — 실제 평가)의 date_found·S1·twin 비율 vs 가짜 배치 2차 val·1차 val 비율(배치 20개 평균).
저장: sv1_draws_v1.csv (draw, rec, role), sv1_situations_v1.csv.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
LAB = TRAIN & np.isfinite(DM)
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {int(b): int(a) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
def dist(a, b, h): return float(np.nanmean(np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4)))
def tdate(b, vis):
    vi = np.where(vis)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; return d, pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
td_all = []
for b, a in CERT.items():
    v = LAB.copy(); v[b] = False; t = tdate(b, v)
    if t is not None and DAYT[a] == t[0] - 1 and SEGT[a] == t[1]: td_all.append([dist(a, b, h) for h in range(24)])
TAU = np.nanquantile(np.array(td_all), .9, axis=0)
def situ(b, vis):
    t = tdate(b, vis); s1 = False
    if t is not None:
        cand = [a for a in np.where(vis)[0] if a != b and DAYT[a] == t[0] - 1 and SEGT[a] == t[1]]
        if cand:
            d = sorted((dist(a, b, 9), a) for a in cand); s1 = (len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)) and d[0][0] <= TAU[9]
    tp = b in CERT and vis[CERT[b]]
    twin = t is not None and any(vis[j] and FARM[j] != FARM[b] and DAYT[j] == t[0] and SEGT[j] == t[1] for j in range(N))
    return dict(date_found=t is not None, S1=bool(s1), S2=bool(tp and not s1), S3=bool(not s1 and not tp), twin=bool(twin))
byfarm = {f: {int(NUM[i]): i for i in range(N) if FARM[i] == f} for f in ("F13", "F47")}
def runs(nums):
    out, cur = [], []
    for n in sorted(nums):
        if cur and n == cur[-1] + 1: cur.append(n)
        else:
            if cur: out.append(cur)
            cur = [n]
    if cur: out.append(cur)
    return out
def make_draw(k):
    rng = np.random.default_rng(1000 + k); role = {}
    for f, m in byfarm.items():
        p2 = [n for n, i in m.items() if n >= 179 and LAB[i]]
        for r in runs(p2):
            if len(r) >= 3:
                role[m[r[0]]] = "del"; role[m[r[-1]]] = "del"
                for n in r[1:-1]: role[m[n]] = "val"
        p1ok = {n for n, i in m.items() if 8 <= n <= 178 and LAB[i]}
        for L in (5, 10, 10, 5):
            for _ in range(500):
                s = int(rng.integers(8, 178 - L - 1)); span = list(range(s - 1, s + L + 1))
                if all(n in p1ok and m[n] not in role for n in span):
                    role[m[s - 1]] = "del"; role[m[s + L]] = "del"
                    for n in range(s, s + L): role[m[n]] = "val"
                    break
    for i, r in list(role.items()):
        if r != "val" or not np.isfinite(DAYT[i]): continue
        for j in range(N):
            if j not in role and LAB[j] and FARM[j] != FARM[i] and DAYT[j] == DAYT[i] and SEGT[j] == SEGT[i] and rng.random() < .28: role[j] = "tw"
    return role
D, S = [], []
for k in range(20):
    role = make_draw(k); vis = LAB & ~np.array([i in role for i in range(N)])
    for i, r in role.items():
        D.append(dict(draw=k, rec=rid(i), role=r))
        if r == "val": S.append(dict(draw=k, rec=rid(i), pass2=NUM[i] >= 179, hi=DM[i] >= 1.0, **situ(i, vis)))
    print("draw", k, "val", sum(r == "val" for r in role.values()), "del", sum(r == "del" for r in role.values()), "tw", sum(r == "tw" for r in role.values()), flush=True)
D = pd.DataFrame(D); S = pd.DataFrame(S); D.to_csv(os.path.join(HERE, "sv1_draws_v1.csv"), index=False); S.to_csv(os.path.join(HERE, "sv1_situations_v1.csv"), index=False)
E = pd.DataFrame([dict(rec=rid(i), **situ(i, LAB)) for i in range(N) if not TRAIN[i]])
pd.set_option("display.width", 200)
cols = ["date_found", "S1", "S2", "S3", "twin"]
T = pd.concat([E[cols].mean().rename("EVAL (60)"), S[S.pass2][cols].mean().rename(f"pseudo pass-2 val (n/draw {S[S.pass2].groupby('draw').size().mean():.0f})"),
               S[~S.pass2][cols].mean().rename(f"pseudo pass-1 val (n/draw {S[~S.pass2].groupby('draw').size().mean():.0f})")], axis=1)
print("\nsituation shares (S2 for EVAL unknown — no truth):"); print(T.round(3).to_string())
print("\npseudo val high-EC share: pass-2", round(float(S[S.pass2].hi.mean()), 3), "| pass-1", round(float(S[~S.pass2].hi.mean()), 3))
print("pseudo pass-2 val by situation x high:"); print(pd.crosstab([S[S.pass2].S1, S[S.pass2].S2], S[S.pass2].hi, normalize=False).to_string())
