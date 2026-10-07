# -*- coding: utf-8 -*-
"""MP0b — 다단계 전파 상한 보강: (1) 양쪽 고정 보간, (2) 단계별 가중 혼합 (진단, 출처 완벽 가정) — 2026-10-07 연구실 클로드
MP0 사슬·전파 정의 재사용. 길이 L(=3,5,7) 연속 구간을 사슬에서 가린다고 보고, 구간 앞 정답 노드의 EC23(앞쪽)·구간 뒤 정답 노드의 EC0(뒤쪽)에서
 앞쪽 전파 f_k(모델 하루 안 변화 누적), 뒤쪽 전파 b_k(거꾸로: 뒤 노드 EC0 에서 모델 (m_0 − m_23) 누적) 를 만든다.
 (1) 양쪽: 위치 i(1..L) 에서 가중 (L+1−i)/(L+1)·f + i/(L+1)·b
 (2) 혼합: 앞쪽만 w_k·f + (1−w_k)·모델, w_k = 학습 쌍에서 k 별 MSE 로 정한 역분산 가중(전 사슬 공통, 폴드 구분 없음 → 낙관적 표시)
 (3) 양쪽+혼합: (1) 과 모델을 거리 min(i, L+1−i) 별 같은 방식으로 혼합
비교: 모델 하루 평균. 2차 노드 따로.
"""
import runpy, os, sys
g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ec_mp0_propagation_premise_v1.py"), run_name="mp0")
np, pd, chains, EC, M, ym, O = g["np"], g["pd"], g["chains"], g["EC"], g["M"], g["ym"], g["O"]
rm = lambda x: np.sqrt(np.mean(np.square(x)))
# k 별 역분산 가중(앞쪽 전파 vs 모델)
wk = {}
for k in range(1, 9):
    q = O[O.k == k]; vf, vm = np.mean(q.err ** 2), np.mean(q.merr ** 2); cov = np.mean(q.err * q.merr)
    wk[k] = max(0, min(1, (vm - cov) / (vf + vm - 2 * cov)))
print("단계별 혼합 가중(전파 몫):", {k: round(v, 2) for k, v in wk.items()})
rows = []
for ch in chains:
    for L in (3, 5, 7):
        for s in range(len(ch) - L - 1):
            a, b = ch[s], ch[s + L + 1]; mid = ch[s + 1:s + L + 1]
            if any(x not in M.index for x in mid):
                continue
            f, cur = [], EC.loc[a, 23]
            for nd in mid:
                m = M.loc[nd].values; f.append(cur + (np.nanmean(m) - m[0])); cur = cur + (m[23] - m[0])
            bk, cur = [], EC.loc[b, 0]
            for nd in reversed(mid):
                m = M.loc[nd].values; bk.append(cur + (np.nanmean(m) - m[23])); cur = cur + (m[0] - m[23])
            bk = bk[::-1]
            for i, nd in enumerate(mid, 1):
                two = ((L + 1 - i) * f[i - 1] + i * bk[i - 1]) / (L + 1)
                mod = np.nanmean(M.loc[nd].values); d = min(i, L + 1 - i)
                rows.append(dict(L=L, i=i, d=d, late=nd[1] >= 179, y=ym[nd], f=f[i - 1], two=two, mod=mod,
                                 fbl=wk[min(i, 8)] * f[i - 1] + (1 - wk[min(i, 8)]) * mod, tbl=wk[min(d, 8)] * two + (1 - wk[min(d, 8)]) * mod))
Q = pd.DataFrame(rows)
for L in (3, 5, 7):
    for nm, m in (("전체", Q.L == L), ("2차", (Q.L == L) & Q.late), ("일반", (Q.L == L) & (Q.y < 1)), ("고EC", (Q.L == L) & (Q.y >= 1))):
        q = Q[m]
        print("L=%d %-4s n=%4d | 모델 %.3f | 앞쪽만 %.3f | 양쪽 %.3f | 앞쪽+혼합 %.3f | 양쪽+혼합 %.3f" % (
            L, nm, len(q), rm(q["mod"] - q.y), rm(q.f - q.y), rm(q.two - q.y), rm(q.fbl - q.y), rm(q.tbl - q.y)))
