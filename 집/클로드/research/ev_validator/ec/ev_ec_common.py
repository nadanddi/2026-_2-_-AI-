# -*- coding: utf-8 -*-
"""EV-EC (2026-10-12 집 클로드 하위 에이전트): shared data / fold / season code for scoring the submitted EC chain
(R3 + TabPFN -> SG2 -> EA2) on the evaluation-geometry validator EV (ev_validator/ev_folds_v1.csv), BASE season vs SE1.
QUANTITIES AND RULES FIXED BEFORE RUNNING (see ev_ec_post.py header for the scoring plan).
Per EV fold k:
  training records = labelled F13/F47 records that are neither 'val' nor 'del' in fold k.
  season: season.mapping refit on fold-k training records only (PAV anchors from training pass-2 records' exact twins);
    BASE = mapping's interp value for every val record; SE1 (pass-2 val rows only) = causal twin season exactly as
    se1_season_twin/se1_cv.py (own outdoor hours 0..h, h >= 5, z-RMSE <= .05 against all fold-training records of both
    farms with their training season values; mu/sd = mapping's training pass-1 weather) else BASE.
  val-row model features = model.features (record's own hours 0..h only) -> no other-ID val or 'del' inputs.
"""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import env_extra  # noqa: F401
import os
from pathlib import Path
import numpy as np, pandas as pd

PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
sys.path.insert(0, PK)
import model as M  # noqa: E402
from season import mapping, vectors, W  # noqa: E402

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = HERE / "out_v1"; OUT.mkdir(exist_ok=True)
DATA = Path(PK) / "data"
FOLDS = pd.read_csv(HERE.parent / "ev_folds_v1.csv")
THR, HMIN = .05, 5


def load_train():
    cache = OUT / "tr_features.pkl"
    x = pd.read_csv(DATA / "train_X.csv", usecols=["row_id"] + W + M.RAW)
    if cache.exists():
        tr = pd.read_pickle(cache)
    else:
        y = pd.read_csv(DATA / "train_y.csv", usecols=["row_id", "sub_ec"])
        tr = M.features(x).merge(y, on="row_id", validate="one_to_one")
        tr["day"] = tr.row_id.str[4:7].astype(int); tr["rec"] = tr.row_id.str[:7]
        tr.to_pickle(cache)
    return x, tr


def twin_seasons(vrecs, refs, rseason, mu, sd, VEC):
    R = np.stack([(VEC[k] - mu[:, None]) / sd[:, None] for k in refs]); rs = np.asarray(rseason, float); out = {}
    for k in vrecs:
        qz = (VEC[k] - mu[:, None]) / sd[:, None]; s = np.full(24, np.nan)
        for h in range(HMIN, 24):
            dist = np.sqrt(np.nanmean((R[:, :, :h + 1] - qz[None, :, :h + 1]) ** 2, axis=(1, 2))); sel = dist <= THR
            if sel.any(): s[h] = rs[sel].mean()
        out[k] = s
    return out


def fold_data(k, x, tr, VEC):
    """returns trn (with season), va_base, va_se1 (same rows/order), val set, del set."""
    f = FOLDS[FOLDS.fold == k]; val = set(f.rec[f.role == "val"]); dele = set(f.rec[f.role == "del"])
    trn = tr[~tr.rec.isin(val | dele)].copy(); va = tr[tr.rec.isin(val)].copy().sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    assert len(val) == va.rec.nunique()
    tdays = trn[["farm", "day"]].drop_duplicates()
    vk = sorted(set(zip(va.farm, va.day.astype(int)))); qd = pd.DataFrame(vk, columns=["farm", "day"])
    ts, qs, notes = mapping(tdays, qd, VEC)
    mu, sd = np.array(notes["standardization"]["mu"]), np.array(notes["standardization"]["sd"])
    trn["season"] = [ts[(a, int(d))] for a, d in zip(trn.farm, trn.day)]
    base = dict(zip(vk, qs)); b = np.array([base[(a, int(d))] for a, d in zip(va.farm, va.day)])
    vp2 = [kk for kk in vk if kk[1] >= 179]
    refs = sorted(ts); tw = twin_seasons(vp2, refs, [ts[r] for r in refs], mu, sd, VEC)
    s = np.array([tw[(a, int(d))][h] if (a, int(d)) in tw else np.nan for a, d, h in zip(va.farm, va.day, va.hour)])
    vb = va.assign(season=b); vs = va.assign(season=np.where(np.isfinite(s), s, b))
    return trn, vb, vs, val, dele
