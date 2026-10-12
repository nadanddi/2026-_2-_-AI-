# -*- coding: utf-8 -*-
"""V15c (2026-10-13 집 클로드; 사용자: '확신만큼만 따라가기 — 뭐라도 해봐'). v15b(PI .15) 를 처음 쓰는 거울 기하 G1 에서만: 10기록 구간 시작을 +5 옮기고(lo+5 기준), 다른 ID 거울 이동 ±4(F13 −4, F47 +4). EV·평가 없음. 출력 local/linker_v10/v15g1_mirror.csv.
(원) V15b (2026-10-13 집 클로드; 사용자: '없음 비율 보정해서 다시'). v15 와 같되 '없음' 학습 사례에 가중 PI/(1−PI)(연결 사례 1) → 학습상 없음 비율 PI.
인자 PI ∈ {0.15, 0.25, 0.35}. 출력 local/linker_v10/v15pi{PI*100}_{mirror,ev,eval}.csv. (v15 는 PI .5 에 해당)
(원) V15 (2026-10-13 집 클로드; 사용자: '전날 없음 선택지 넣어봐').  = v14(날짜 투표, DMT3e E0 달력, L 학습) + '없음' 후보.
문제(6.760~6.764): 링커 확신 p 는 후보끼리 상대값 → 진짜 전날이 안 보여도 누군가에 높은 p(F47_146 .97).
'없음' 후보: 실제 후보 특징은 0, 아래 '없음 전용 특징'만 가짐(실제 후보는 이 특징 0):
  nl_bias(1), nl_dm1_0(투표 날짜−1·같은 구간 보이는 정답 기록 0개), nl_dm1(그 수, 4 에서 자름), nl_date_unk(투표 날짜 모름),
  nl_min_oj(후보 중 외기 자정 비용 최소), nl_min_fp(후보 중 0시 지문 거리 최소), nl_n_lab(보이는 정답 후보 수/10)
학습: 학습 확실 연결(TL7) 마다 두 사례 — (가) 진짜 전날 보임 → 정답 = 그 기록, (나) 진짜 전날을 숨김(후보에서 제거·보이는 정답에서 제외) → 정답 = 없음.
출력(local/linker_v10/v15_{mirror,ev,eval}.csv): rec, fold, date_src, p_null, 실제 후보 상위 6(prev_k, p_k, lab_k; p 는 없음 포함 전체에서의 확률).
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); PI = 0.15; TAGP = "v15g1"; sys.argv = [sys.argv[0]]
src = open(os.path.join(HERE, "v14_ec_vote_v1.py"), encoding="utf-8").read()
exec(src[:src.index("TRG = []")])  # features10, feats14, vote_date, fit_clogit, idx, DAYT, SEGT, ojf, F12, OUT, CUR_GEO, H_
NL = ["nl_bias", "nl_dm1_0", "nl_dm1", "nl_date_unk", "nl_min_oj", "nl_min_fp", "nl_n_lab"]
FALL = F12 + NL
def build(b, lab, removed):
    D, src_ = feats14(b, lab, removed)
    if len(D) == 0: return None, None, src_
    a = D.a.values.astype(int); Z = np.zeros((len(D) + 1, len(FALL))); Z[:-1, :len(F12)] = D[F12].values.astype(float)
    labb = lab & (np.arange(N) != b); tb, _ = vote_date(b, labb)
    ndm1 = int(np.sum(labb & (DAYT == tb[0] - 1) & (SEGT == tb[1]))) if tb else 0
    Z[-1, len(F12):] = [1.0, float(tb is not None and ndm1 == 0), min(ndm1, 4), float(tb is None), float(D.oj.min()), float(D.fp.min()), lab[a].sum() / 10]
    return Z, a, src_
TRG = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    Z, ca, _ = build(b, lab, TEST_IDX)
    if Z is None: continue
    y = np.where(ca == a)[0]
    if not len(y): continue
    TRG.append((b, a, Z, y[0], 1.0))
    lab2 = lab.copy(); lab2[a] = False
    Z2, ca2, _ = build(b, lab2, set(TEST_IDX) | {a})
    if Z2 is not None: TRG.append((b, a, Z2, len(ca2), PI / (1 - PI)))
print("training groups", len(TRG), flush=True)
def fit_wt(groups, l2=.5, iters=300, lr=.1):
    k = groups[0][0].shape[1]; w = np.zeros(k); m = np.zeros(k); v = np.zeros(k); sw = sum(g[2] for g in groups)
    for t in range(1, iters + 1):
        g = l2 * w.copy()
        for Xc, y, wt in groups:
            s_ = Xc @ w; s_ -= s_.max(); p = np.exp(s_); p /= p.sum(); g -= wt * (Xc[y] - p @ Xc) * len(groups) / sw
        g /= len(groups); m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
        w -= lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    return w
fitw = lambda touch: fit_wt([(t[2], t[3], t[4]) for t in TRG if not touch[t[0]] and not touch[t[1]]])
def pick(b, lab, removed, w):
    Z, a, src_ = build(b, lab, removed)
    if Z is None: return None
    s = Z @ w; p = np.exp(s - s.max()); p /= p.sum(); r = dict(rec=rid(b), date_src=src_, p_null=float(p[-1]))
    o = np.argsort(-p[:-1])[:6]
    for k, j in enumerate(o): r[f"prev{k}"] = rid(a[j]); r[f"p{k}"] = float(p[j]); r[f"lab{k}"] = bool(lab[a[j]])
    return r
CUR_GEO[0] = "mirror"; rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -4 if f == "F13" else 4
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo - 5, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist())
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w = fitw(touch); lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if TRAIN[b]:
                r = pick(b, lab, rem | oth, w)
                if r: rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{TAGP}_mirror.csv"), index=False); print("mirror", len(rows), flush=True)
