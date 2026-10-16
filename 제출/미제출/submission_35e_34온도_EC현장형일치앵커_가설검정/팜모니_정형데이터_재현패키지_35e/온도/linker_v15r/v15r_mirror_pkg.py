# -*- coding: utf-8 -*-
"""submission_25 linker (package, 2026-10-15 집 클로드): v15R 'null option' linker picks for the 400 TRAINING records in the MIRROR geometry
(each training record linked as if its 10-number block were hidden, like the evaluation structure) -> used ONLY as the 'same house yesterday'
table for the TRAINING sequences of the V3L (temperature) and EV3L (EC) GRU members.  Same research chain and the same rule patches as
v15r_eval_pkg.py (operation-fingerprint standardisation FPA from TRAINING records only, not test_X statistics); v13 -> v14 -> v15 mirror sections.
Run (from the package 온도 folder):  LINKER_WORK=<work dir> LINKER_DATA=<folder with train_X.csv,test_X.csv> python linker_v15r/v15r_mirror_pkg.py <out_csv>
"""
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__)); PKG_T = os.path.dirname(HERE)
OUTF = sys.argv[1]; sys.argv = [sys.argv[0], "0.15"]
sys.path.insert(0, os.path.join(PKG_T, "linker_v7"))           # package feat4 / feat7 (paths via LINKER_WORK)
sys.modules["env"] = types.SimpleNamespace(DATA=os.environ["LINKER_DATA"], LOCAL=os.path.join(HERE, "data"), ROOT=PKG_T)
NS = {"__name__": "v15r_pkg", "__file__": os.path.join(HERE, "v15_src.py")}
FEAT10 = '''
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
'''
FPA_OLD = "FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)"
FPA_NEW = "FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)"
def strip_imports(src):
    out = []
    for ln in src.split("\n"):
        if ln.startswith("sys.path.insert(0, r\"C:") or ln.strip() == "from feat10 import *": continue
        out.append(ln)
    return "\n".join(out)
exec(FEAT10, NS)
# ---- v13: mirror picks ----
s13 = strip_imports(open(os.path.join(HERE, "v13_src.py"), encoding="utf-8").read()); assert FPA_OLD in s13; s13 = s13.replace(FPA_OLD, FPA_NEW)
s13 = s13.replace('TAG = sys.argv[1] if len(sys.argv) > 1 else "E0"', 'TAG = "E0"')
exec(s13[:s13.index("# mirror o=0")], NS)
exec('''
_rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w = fitw(touch); lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if TRAIN[b]:
                r = pick(b, lab, rem2, w)
                if r: _rows.append(r)
V13_MIRROR = pd.DataFrame(_rows)
''', NS)
print("v13 mirror picks", len(NS["V13_MIRROR"]), flush=True)
# ---- v14 functions (date voting) + v15 null-option linker, mirror geometry ----
s14 = strip_imports(open(os.path.join(HERE, "v14_src.py"), encoding="utf-8").read()); assert FPA_OLD in s14; s14 = s14.replace(FPA_OLD, FPA_NEW)
V13LINE = 'V13 = {k: pd.read_csv(os.path.join(OUT, f"v13E0_picks_{k}.csv")) for k in ("mirror", "ev", "eval")}'; assert V13LINE in s14
s14 = s14.replace(V13LINE, 'V13 = {"mirror": V13_MIRROR}')
s15 = open(os.path.join(HERE, "v15_src.py"), encoding="utf-8").read()
READ14 = 'src = open(os.path.join(HERE, "v14_ec_vote_v1.py"), encoding="utf-8").read()'; assert READ14 in s15
s15 = s15.replace(READ14, "src = S14_PATCHED")
NS["S14_PATCHED"] = s14
cut = s15.index('CUR_GEO[0] = "mirror"; rows = []')
exec(s15[:cut], NS)
exec('''
CUR_GEO[0] = "mirror"; rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist())
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w = fitw(touch); lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if TRAIN[b]:
                r = pick(b, lab, rem | oth, w)
                if r: rows.append(r)
E = pd.DataFrame(rows)
''', NS)
E = NS["E"]; assert not os.path.exists(OUTF), "refusing to overwrite " + OUTF
E.to_csv(OUTF, index=False); print("wrote", OUTF, len(E))
