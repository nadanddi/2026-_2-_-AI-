# linker v10 (2026-10-12 집 클로드; 사용자: '짝찾기 고쳐서 다시'): v7 + 10-12 규정 반영 — 정답이 보이는 학습 기록은 번호·ID 무관하게 후보.
# 평가 역할 기록(정답 안 보임)은 기존대로 같은 ID 앞 번호만(입력만), 다른 ID 평가 기록 불가.  RC1(6.714): 1차 대상 53개의 진짜 전날이 '같은 ID 뒤 번호 학습 기록'이라 v7 에서 빠졌음.
# 추가 특징 lat = 같은 ID 이면서 대상보다 번호가 뒤인 후보(그 경우 번호 차 특징 d1/d3/ldn/d2* 는 0).
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v7"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import feat7 as F7
from feat7 import *
FEATS10 = FEATS7 + ["lat"]
def _vis10(labeled):
    def vis(b, removed):
        f = FARM[b]; v = labeled.copy() | ((FARM == f) & (NUM < NUM[b]))
        if removed: v[list(removed)] = False
        v[b] = False
        return v
    return vis
def features10(b, h, labeled, removed=set()):
    old = F7.visible; F7.visible = _vis10(labeled if labeled is not None else TRAIN)
    try:
        D, info = F7.features7(b, h, labeled, removed)
    finally:
        F7.visible = old
    if len(D) == 0: return D, info
    a = D.a.values.astype(int); lat = (FARM[a] == FARM[b]) & (NUM[a] > NUM[b])
    D["lat"] = lat.astype(float)
    for c in ("d1", "d3", "ldn", "d2w", "d2c"): D.loc[lat, c] = 0.0
    return D, info
def train_groups10():
    TRG = []
    for b, a in zip(TL7.b, TL7.a):
        lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        df, _ = features10(b, 0, lab, removed=TEST_IDX)
        if len(df) == 0: continue
        y = np.where(df.a.values == a)[0]
        if len(y): TRG.append((b, a, df[FEATS10].values.astype(float), y[0]))
    return TRG
def pick10(b, h, lab, removed, w):
    df, _ = features10(b, h, lab, removed=removed)
    if len(df) == 0: return None, 0.0
    s = df[FEATS10].values.astype(float) @ w; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p)); return int(df.a.values[k]), float(p[k])
