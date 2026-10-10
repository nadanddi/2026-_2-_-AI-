# linker 2-step (2026-10-12 집 클로드): DIRECT search of the same house's record TWO dates earlier, among TRAINING records,
# using only target record b's own hours 0..h + same-greenhouse earlier inputs + both greenhouses' TRAINING inputs/labels.
# No use of any decision/prediction for an intermediate (evaluation) record.
# Fixed before running (no tuning):
#   candidates : labelled TRAINING records (both IDs), visible(b), joint-calendar rank distance ds in {1,2,3} from b's date
#                (feat4 calendar, seg-1 based gpos); b or candidate without calendar position -> no candidates.
#                HARD exclusion (labels only): candidate a whose next date is already a confirmed labelled record
#                (feat4.label_successor on the labelled set) -> dropped.
#   features   : fp (feat6 operation-fingerprint residual distance, hours 0..h; regressions/sd on TRAINING only), fpna,
#                lev (indoor raw level |b - a| same hours 0..h / training sd, mean of 3 vars, cap 6),
#                dsdev = |ds - 2|, xf (other ID).
#   model      : feat4.fit_clogit l2 .5, iters 300, MASK world (TEST_IDX removed). Truth = TL7 twice (b->a1->a2, both cert);
#                in the training groups a1 and b are marked unlabelled (mimics evaluation: middle record unlabelled).
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v7")
sys.path.insert(1, r"C:\work\farmai\집\클로드\research\linker_v6")
from feat7 import *
import feat4 as F4
from feat6 import fp_dist
S2S = r"C:\work\farmai\집\클로드\research\local\linker_2step"
sdI = np.nanstd(IN_[TRAIN].reshape(-1, IN_.shape[2]), 0) + 1e-6
P1 = dict(zip(TL7.b, TL7.a))
TRUTH2 = {b: (P1[a1], a1) for b, a1 in P1.items() if a1 in P1}     # b -> (a2, a1)
FEATS2 = ["fp", "fpna", "lev", "dsdev", "xf"]


def _cal(vis):
    key = vis.tobytes()
    if key not in F4._cal_cache:
        comp = components(vis); gpos, mp = calendar(vis, comp); F4._cal_cache[key] = (comp, gpos, mp)
    return F4._cal_cache[key]


def features2(b, h, labeled, removed=set()):
    f = FARM[b]; nb = NUM[b]
    vis = visible(b, removed); comp, gpos, mp = _cal(vis)
    vi = np.where(vis)[0]
    eq = (np.abs(X[vi, :h + 1] - X[b, :h + 1]) < 1e-6).reshape(len(vi), -1).mean(1)
    thr = 1.0 if h < 6 else 0.9
    mC = {comp[i] for i in vi[eq >= thr]}
    ps = [gpos[c] for c in mC if c in gpos]
    if nb < 179: ps = ps + [mp(b)]
    bpos = np.mean(ps) if ps else np.nan
    if not np.isfinite(bpos): return pd.DataFrame(), dict(bpos=bpos, why="b no calendar")
    allp = sorted(set(list(gpos.values()) + [bpos])); rk = {p: k for k, p in enumerate(allp)}
    succ = succ_for(labeled)
    rows = []
    for a in vi:
        if not labeled[a] or not TRAIN[a] or comp[a] in mC: continue
        pa = gpos.get(comp[a], np.nan)
        if not np.isfinite(pa): continue
        ds = rk[bpos] - rk[pa]
        if ds not in (1, 2, 3): continue
        if a in succ and succ[a] != b: continue          # next date already a confirmed labelled record
        fp = fp_dist(b, a, h)
        lev = np.nanmean(np.abs(IN_[b, :h + 1] - IN_[a, :h + 1]) / sdI)
        rows.append(dict(a=a, fp=0.0 if np.isnan(fp) else min(fp, 6), fpna=float(np.isnan(fp)),
                         lev=2.0 if np.isnan(lev) else min(lev, 6), dsdev=float(abs(ds - 2)), xf=float(FARM[a] != f), ds=ds))
    D = pd.DataFrame(rows)
    return D, dict(bpos=bpos, why="" if len(D) else "no candidates")


def train_groups2(h=0, skip=None):
    """MASK world groups from 2-step truth. skip(b,a1,a2) -> True to drop (CV)."""
    G = []
    for b, (a2, a1) in TRUTH2.items():
        if skip is not None and skip(b, a1, a2): continue
        lab = TRAIN.copy(); lab[a1] = False; lab[b] = False
        df, _ = features2(b, h, lab, removed=TEST_IDX)
        if len(df) < 2: continue
        y = np.where(df.a.values == a2)[0]
        if len(y): G.append((b, a1, a2, df[FEATS2].values.astype(float), y[0]))
    return G


def pick2(b, h, lab, removed, w):
    df, info = features2(b, h, lab, removed=removed)
    if len(df) == 0: return None, 0.0, 0
    s = df[FEATS2].values.astype(float) @ w; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p))
    return int(df.a.values[k]), float(p[k]), len(df)
