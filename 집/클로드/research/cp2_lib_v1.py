# -*- coding: utf-8 -*-
"""CP2 라이브러리 (2026-10-14 집 클로드).  가변 길이 사슬 RNN 학습·검증에서 쓰는 함수 — cp2_preprocess_v1.py 출력 표를 읽어 씀.
- cut_tracks(T, hidden): 검증에서 숨긴 기록(hidden)이 들어 있는 동 트랙을 그 기록과 앞뒤 1개에서 잘라, 학습 시퀀스가 숨긴 기록의 정답·이웃을 건너 잇지 않게 함.
- label_mask(rng, n, p): 학습 때 앞날 EC 정답 입력을 가릴지(1 = 가림) — 평가 순수 창 흉내(p 는 미리 정함).
- eval_history(E, rec): 평가 기록이 규정상 이을 수 있는 앞 기록 후보(학습 기록 또는 같은 ID 앞 번호 평가 기록) — 시각별 선택은 예측 때.
"""
import numpy as np, pandas as pd


def cut_tracks(T, hidden):
    """T: cp2_tracks 표(rec, track, track_pos).  hidden: 숨길 rec 집합.  반환: rec -> (새 트랙 번호, 위치), 숨긴 기록과 그 앞뒤 1개는 학습 시퀀스에서 제외(None)."""
    T = T.sort_values(["track", "track_pos"]); drop = set(hidden)
    for tr, g in T.groupby("track"):
        recs = list(g.rec)
        for j, r in enumerate(recs):
            if r in hidden:
                if j > 0: drop.add(recs[j - 1])
                if j + 1 < len(recs): drop.add(recs[j + 1])
    out, new = {}, 0
    for tr, g in T.groupby("track"):
        pos = 0; started = False
        for r in g.rec:
            if r in drop:
                if started: new += 1; started = False; pos = 0
                out[r] = None; continue
            out[r] = (new, pos); pos += 1; started = True
        if started: new += 1
    return out


def label_mask(rng, n, p):
    return (rng.random(n) < p).astype(np.float32)


def eval_history(E, rec):
    g = E[E.rec == rec]
    return g.sort_values("fp").reset_index(drop=True)
