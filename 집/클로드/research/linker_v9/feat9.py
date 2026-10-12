# linker v9 (2026-10-12 집 클로드; 사용자: '짝찾기에 운영 지문 넣는 방식'): v7 + raw same-hour OPERATION SCHEDULE distance.
# fp(b, a; h) = mean over hours 0..h and 8 variables (7 actuators + in_co2, raw train+test standardisation) of min(|b_t - a_t| / sd, 4).
# Differs from v6 (6.576): base v7 (other-ID training candidates), raw schedule values at the SAME hour (not outdoor-residual),
# motivated by LF1 (6.695): same-house consecutive days distance .195 vs other houses same date .579; v7 wrong choices -> true closer 77 % at h0.
# Rules unchanged: b uses only its own hours 0..h; a is a training record or same-ID earlier record (visible()).
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v7"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
from feat7 import *
import env
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
_recs = [rid(i) for i in range(N)]
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=_recs, columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
FEATS9 = FEATS7 + ["fp"]
def features9(b, h, labeled, removed=set()):
    D, info = features7(b, h, labeled, removed)
    if len(D) == 0: return D, info
    a = D.a.values.astype(int)
    d = np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1][None]), 4)
    D["fp"] = np.nan_to_num(np.nanmean(d.reshape(len(a), -1), 1), nan=1.0)
    return D, info
def train_groups9(h=0):
    TRG = []
    for b, a in zip(TL7.b, TL7.a):
        lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        df, _ = features9(b, h, lab, removed=TEST_IDX)
        if len(df) == 0: continue
        y = np.where(df.a.values == a)[0]
        if len(y): TRG.append((b, a, df[FEATS9].values.astype(float), y[0]))
    return TRG
def pick9(b, h, lab, removed, w):
    df, _ = features9(b, h, lab, removed=removed)
    if len(df) == 0: return None, 0.0
    s = df[FEATS9].values.astype(float) @ w; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p)); return int(df.a.values[k]), float(p[k])
