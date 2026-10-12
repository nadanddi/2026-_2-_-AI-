# -*- coding: utf-8 -*-
"""CMP1 고EC 압축 원인 분해 (2026-10-13 집 클로드, hx8/compress, 서술·진단 — 채택 판정 없음).
A. 저장 OOF(CUR, DIAG10/P2LOO/EL1, 3시드 평균): 고EC(학습 하루 평균 >= 1.0) 압축 크기, 하루 수준 vs 시각 몫,
   '고EC 임을 안다면' 오라클 분해(폴드 밖 계수): M1 고EC 상수, M2 모델+고EC 평균 잔차, M3 고EC 안 선형 재보정.
B. 입력 이웃 섞임: 하루 요약(S1, 42차원) / 시간별 벡터(S2, 336차원) 표준화 공간에서 다른 폴드 학습 기록 최근접 K
   → 고EC 비율, 이웃 정답 평균; 모델 예측이 '이웃 평균'처럼 행동하는지, 고EC 이웃끼리의 평균이 크기를 아는지.
C. 대리 ExtraTrees(시간 행, 현재값·0..h 누적평균·0시값·시각·학습 달력일, DIAG10 같은 폴드) 잎 구성:
   고EC 행이 떨어지는 잎 안 고EC 행 비율, 잎 안 고EC 행 평균 vs 실제 크기 (min_samples_leaf 1/5/20).
학습 자료만 사용(평가 입력 없음).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm_common import *
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

THR = 1.0
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# ---------------- A ----------------
P("=== A. 저장 OOF 압축 ===")
for V in ["DIAG10", "P2LOO", "EL1"]:
    sg, fold = oof(V, "sg")
    rids = [k for k in sg if TRAIN[rid2i[k]]]
    idx = np.array([rid2i[k] for k in rids])
    y = ydm[idx]; pm = np.array([sg[k].mean() for k in rids])
    H = y >= THR
    P(f"[{V}] 기록 {len(rids)}, 고EC {H.sum()}: 정답 평균 {y[H].mean():.3f}, 예측 {pm[H].mean():.3f}, 비 {pm[H].mean()/y[H].mean():.3f}")
    if H.sum() >= 4:
        b = np.polyfit(y[H], pm[H], 1)[0]; rho = spearmanr(y[H], pm[H])[0]
        P(f"   고EC 안 예측~정답 기울기 {b:.3f}, 순위상관 {rho:.3f}; 전체 기울기 {np.polyfit(y, pm, 1)[0]:.3f}")
        PH = pm >= 0.9
        if PH.sum() >= 4:
            P(f"   예측>=.9 {PH.sum()}기록: 정답~예측 기울기 {np.polyfit(pm[PH], y[PH], 1)[0]:.3f}, 실제 고EC {int((H&PH).sum())}")
    # 하루 수준 vs 시각 몫 (고EC)
    Yh = np.array([Ytr[rid2i[k]] for k in rids]); Ph = np.array([sg[k] for k in rids])
    e = Yh - Ph; dl = e.mean(1, keepdims=True)
    for nm_, m in [("고EC", H), ("보통", ~H)]:
        sse = (e[m] ** 2).sum(); sdl = 24 * (dl[m] ** 2).sum()
        P(f"   {nm_}: 시간 SSE {sse:.2f}, 하루 수준 몫 {sdl/sse:.3f}, 전체 SSE 중 {nm_} 몫 {sse/(e**2).sum():.3f}")
    if V == "DIAG10":
        # 오라클(고EC 를 안다고 가정), 폴드 밖 계수
        fr = np.array([fold[k] for k in rids])
        m1 = np.zeros(H.sum()); m2 = np.zeros(H.sum()); m3 = np.zeros(H.sum())
        hy, hp, hf = y[H], pm[H], fr[H]
        for j in range(H.sum()):
            o = hf != hf[j]
            m1[j] = hy[o].mean(); m2[j] = hp[j] + (hy[o] - hp[o]).mean()
            bb = np.polyfit(hp[o], hy[o], 1); m3[j] = np.polyval(bb, hp[j])
        P(f"   오라클(고EC {H.sum()}, 하루 수준 RMSE): 모델 {rmse(hy-hp):.3f} | M1 고EC 상수 {rmse(hy-m1):.3f} | "
          f"M2 모델+평균잔차 {rmse(hy-m2):.3f} | M3 고EC 안 선형 {rmse(hy-m3):.3f}")
        P(f"   고EC 정답 분산(표준편차) {hy.std():.3f}, 모델 잔차 평균 {np.mean(hy-hp):+.3f}, 잔차 표준편차 {np.std(hy-hp):.3f}")
        # 정답 크기 구간별
        for lo, hi_ in [(1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 9)]:
            m = (hy >= lo) & (hy < hi_)
            if m.any():
                P(f"     정답 {lo}~{hi_}: n {m.sum()}, 정답 {hy[m].mean():.2f}, 예측 {hp[m].mean():.2f}, 비 {hp[m].mean()/hy[m].mean():.2f}")
        A_rids, A_y, A_pm, A_H, A_fold = rids, y, pm, H, fr

# ---------------- B ----------------
P("\n=== B. 입력 이웃 섞임 (DIAG10 다른 폴드 학습 기록 중 최근접) ===")
rids, y, pm, H, fr = A_rids, A_y, A_pm, A_H, A_fold
idx = np.array([rid2i[k] for k in rids])
trn = np.where(TRAIN)[0]
mu = np.nanmean(A[trn].reshape(-1, 14), 0); sd = np.nanstd(A[trn].reshape(-1, 14), 0) + 1e-9
Z = (A - mu) / sd
S1 = np.concatenate([np.nanmean(Z, 1), np.nanmean(Z[:, 0:6], 1), np.nanmean(Z[:, 9:17], 1)], 1)
S2 = Z.reshape(N, -1)
for nm_, S in [("S1 하루요약", S1), ("S2 시간벡터", S2)]:
    S = np.nan_to_num(S)
    Si = S[idx]
    Dm = ((Si[:, None, :] - Si[None, :, :]) ** 2).sum(-1)
    same = fr[:, None] == fr[None, :]
    Dm[same] = np.inf
    for K in [5, 10, 20]:
        o = np.argsort(Dm, 1)[:, :K]
        shH = H[o].mean(1); nbm = y[o].mean(1)
        rho_all = spearmanr(pm, nbm)[0]
        P(f"[{nm_} K{K}] 고EC 기록: 이웃 고EC 비율 중앙 {np.median(shH[H]):.2f}(평균 {shH[H].mean():.2f}), 이웃 정답 평균 {nbm[H].mean():.3f} "
          f"(모델 {pm[H].mean():.3f}, 정답 {y[H].mean():.3f}); 보통 기록 이웃 고EC 비율 {shH[~H].mean():.3f}; "
          f"전체 모델~이웃평균 순위상관 {rho_all:.2f}; 고EC 안 (예측/정답)~이웃고EC비율 {spearmanr(pm[H]/y[H], shH[H])[0]:.2f}")
    # 고EC 이웃끼리의 평균이 크기를 아는가 (2단계 아이디어의 상한 진단)
    DmH = Dm[np.ix_(H, H)]
    for K in [3, 5]:
        o = np.argsort(DmH, 1)[:, :K]
        hy = y[H]; nb = hy[o].mean(1)
        P(f"[{nm_}] 고EC끼리 최근접 K{K}: 정답~이웃평균 순위상관 {spearmanr(hy, nb)[0]:.2f}, RMSE 이웃평균 {rmse(hy-nb):.3f} vs 모델 {rmse(hy-pm[H]):.3f}"
          f" vs 고EC 상수 {rmse(hy-hy.mean()):.3f}")

# ---------------- C ----------------
P("\n=== C. 대리 ExtraTrees 잎 구성 (DIAG10 같은 폴드) ===")
cday = np.array([cal.get(k, np.nan) for k in r.rid])
def rowfeat(i):
    a = A[i]; cm = np.cumsum(np.nan_to_num(a), 0) / np.arange(1, 25)[:, None]
    f = np.concatenate([a, cm, np.repeat(a[:1], 24, 0), np.arange(24)[:, None], np.full((24, 1), cday[i])], 1)
    return f
F = np.stack([rowfeat(i) for i in range(N)])  # N x 24 x 44
med = np.nanmedian(F[trn].reshape(-1, F.shape[2]), 0)
F = np.where(np.isnan(F), med, F)
Hrec = np.array([TRAIN[i] and ydm[i] >= THR for i in range(N)])
for msl in [1, 5, 20]:
    predd = {}; comp = []
    for f in sorted(set(fr)):
        te = idx[fr == f]; tr_ = idx[fr != f]
        Xtr_ = F[tr_].reshape(-1, F.shape[2]); ytr_ = Ytr[tr_].reshape(-1)
        hrow = np.repeat(Hrec[tr_], 24)
        et = ExtraTreesRegressor(200, max_features=0.5, min_samples_leaf=msl, n_jobs=4, random_state=7).fit(Xtr_, ytr_)
        for i in te:
            predd[i] = et.predict(F[i])
        teH = [i for i in te if Hrec[i]]
        if teH:
            Lt = et.apply(Xtr_)  # rows x trees
            Xq = np.concatenate([F[i] for i in teH]); Lq = et.apply(Xq)
            yq = np.concatenate([Ytr[i] for i in teH]); rq = np.repeat(teH, 24)
            S = np.zeros(len(Xq)); MH = np.zeros(len(Xq)); MN = np.zeros(len(Xq)); NOH = np.zeros(len(Xq)); PR = np.zeros(len(Xq))
            for t in range(Lt.shape[1]):
                ml = max(Lt[:, t].max(), Lq[:, t].max()) + 1
                cnt = np.bincount(Lt[:, t], minlength=ml); ch = np.bincount(Lt[:, t], weights=hrow, minlength=ml)
                sy = np.bincount(Lt[:, t], weights=ytr_, minlength=ml); syh = np.bincount(Lt[:, t], weights=ytr_ * hrow, minlength=ml)
                l = Lq[:, t]
                s = ch[l] / cnt[l]; S += s
                mh = np.where(ch[l] > 0, syh[l] / np.maximum(ch[l], 1), np.nan)
                mn = np.where(cnt[l] - ch[l] > 0, (sy[l] - syh[l]) / np.maximum(cnt[l] - ch[l], 1), np.nan)
                MH += np.nan_to_num(mh); NOH += (ch[l] == 0); MN += np.nan_to_num(mn)
                PR += sy[l] / cnt[l]
            T_ = Lt.shape[1]
            comp.append(pd.DataFrame({"rec": rq, "y": yq, "share": S / T_, "noh": NOH / T_,
                                      "mh": MH / np.maximum(T_ - NOH, 1), "pred": PR / T_}))
    pmE = np.array([predd[i].mean() for i in idx])
    C = pd.concat(comp).groupby("rec").mean()
    P(f"[ET 잎{msl}] 전체 하루수준 RMSE {rmse(y-pmE):.3f}(저장 CUR {rmse(y-pm):.3f}); 고EC {H.sum()}: 정답 {y[H].mean():.3f} 예측 {pmE[H].mean():.3f} "
      f"기울기 {np.polyfit(y[H], pmE[H], 1)[0]:.2f}(CUR {np.polyfit(y[H], pm[H], 1)[0]:.2f})")
    P(f"   고EC 행이 떨어진 잎: 고EC 행 비율 평균 {C.share.mean():.3f}(중앙 {C.share.median():.3f}), 고EC 행 0인 잎 비율 {C.noh.mean():.3f}, "
      f"잎 안 고EC 행 평균 {C.mh.mean():.3f}(정답 {C.y.mean():.3f}); 정답~잎고EC평균 순위상관 {spearmanr(C.y, C.mh)[0]:.2f}")
    # 섞임 분해: 예측 부족분(정답-예측) = [정답 - 잎고EC평균] + [잎고EC평균 - 예측]
    P(f"   부족분 {np.mean(C.y-C.pred):+.3f} = 고EC 안 흩어짐 {np.mean(C.y-C.mh):+.3f} + 보통 행 섞임 {np.mean(C.mh-C.pred):+.3f}")
    for lo, hi_ in [(1.0, 1.5), (1.5, 2.0), (2.0, 9)]:
        m = (C.y >= lo) & (C.y < hi_)
        if m.any():
            P(f"     정답 {lo}~{hi_}: n {m.sum()}, 정답 {C.y[m].mean():.2f}, 잎고EC평균 {C.mh[m].mean():.2f}, 예측 {C.pred[m].mean():.2f}, 고EC 비율 {C.share[m].mean():.2f}")

open(os.path.join(HERE, "cmp1_decompose_v1.log"), "w", encoding="utf-8").write("\n".join(out))
