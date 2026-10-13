# -*- coding: utf-8 -*-
"""R2C2 (2026-10-14 집 클로드; 사용자: '틀렸다고 그냥 끝내는 거야?').  서술.  R2C1 의 '끊긴 줄'(덩어리 앞 지워진 번호에서 멈춤) 을, 지워진 번호를 건너뛰어(r−2 가 없으면 r−4 …) 계속 따라감.
같은 ID 앞 번호만(학습 기록 또는 평가 기록 — 규정상 허용) — 처음 만나는 정답 학습 기록 = 조상.  날짜 차는 학습 달력(평가 대상은 0시 외기 일치 최빈), 하루 운영 지문 거리, 조상 EC(하루 평균·23시).
비교: 지문 거리의 기준 — 학습 cert 1일 쌍 거리 분포(중앙·95%).  14회차 예측과 나란히."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "r2c1_minus2_chain_check_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "r2c1_minus2_chain_check_v1.py")
exec(src[:src.index("P14 = pd.read_csv")])
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; pm = P14.groupby("rec").sub_ec.mean()
cpd = [fpd(b, a) for b, a in CERT.items() if b in T.index and a in T.index]; print(f"cert 1-day pair fingerprint distance: median {np.median(cpd):.3f}, 95% {np.quantile(cpd, .95):.3f}")
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
out = []
for r in sorted(T[~T.train & ~T.lock].index):
    if r not in pm.index: continue
    n, path, anc, skipped = int(r[4:]), [], None, 0
    while n > 2:
        n -= 2; nx = key(r[:3], n)
        if nx not in T.index: skipped += 1; path.append(f"({nx} deleted)"); continue
        path.append(nx)
        if T.at[nx, "train"] and not T.at[nx, "lock"]: anc = nx; break
        if len(path) > 8: break
    dday = (T.at[r, "day"] - T.at[anc, "day"]) if (anc and np.isfinite(T.at[r, "day"]) and np.isfinite(T.at[anc, "day"])) else np.nan
    out.append(dict(rec=r, pure=inpure(r), pred14=round(float(pm[r]), 3), ancestor=anc, hops=len(path), skipped=skipped, anc_daymean=round(float(DM[idx[anc]]), 3) if anc else np.nan,
                    anc_EC23=round(float(EC[idx[anc], 23]), 3) if anc else np.nan, cal_gap=dday, fp_dist=round(fpd(r, anc), 3) if anc else np.nan, path=">".join(path)))
E = pd.DataFrame(out); pd.set_option("display.width", 240)
print(E[E.skipped > 0].to_string(index=False))
print("\nskip-chain reaches ancestor:", int(E[E.skipped > 0].ancestor.notna().sum()), "of", int((E.skipped > 0).sum()), "records whose -2 chain hit a deleted number")
E.to_csv(os.path.join(HERE, "r2c2_skip_chains_v1.csv"), index=False)
