# -*- coding: utf-8 -*-
"""submission_23 temperature linker (package copy, 2026-10-13 집 클로드): v15R 'null option' linker picks for the 60 EVALUATION records only.
Chain (research originals copied unchanged as v13_src.py / v14_src.py / v15_src.py and patched at run time):
  v13 (DMT3e training-only calendar) eval picks -> used by v14 date voting -> v15 null-option conditional logit (PI .15) -> eval picks.
Rule patches applied here (vs the research files): the operation-fingerprint standardisation (FPA) in v13 AND v14 uses TRAINING records only
  (research v13/v14 used all records incl. test_X = test-set statistic).  Mirror/EV validation sections are not run (eval only).
Rules: an evaluation record's features use only its own hour-0 inputs, training records (any ID, labels allowed) and inputs of same-ID
  EARLIER-numbered records (features10 visibility); other-ID evaluation inputs and later inputs are never visible.
Run (from the package 온도 folder):  LINKER_WORK=<work dir with base.pkl/truth.pkl from 온도/linker> LINKER_DATA=<folder with train_X.csv,test_X.csv>
       python linker_v15r/v15r_eval_pkg.py <out_csv>
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
# ---- v13: eval picks only ----
s13 = strip_imports(open(os.path.join(HERE, "v13_src.py"), encoding="utf-8").read()); assert FPA_OLD in s13; s13 = s13.replace(FPA_OLD, FPA_NEW)
s13 = s13.replace('TAG = sys.argv[1] if len(sys.argv) > 1 else "E0"', 'TAG = "E0"')
exec(s13[:s13.index("# mirror o=0")], NS)
exec('''
w_all = fitw(np.zeros(N, bool)); _rows = []
for b in [i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13", "F47")]:
    r = pick(b, TRAIN.copy(), set(), w_all)
    if r: _rows.append(r)
V13_EVAL = pd.DataFrame(_rows)
''', NS)
print("v13 eval picks", len(NS["V13_EVAL"]), flush=True)
# ---- v14 functions (date voting) + v15 null-option linker, eval only ----
s14 = strip_imports(open(os.path.join(HERE, "v14_src.py"), encoding="utf-8").read()); assert FPA_OLD in s14; s14 = s14.replace(FPA_OLD, FPA_NEW)
V13LINE = 'V13 = {k: pd.read_csv(os.path.join(OUT, f"v13E0_picks_{k}.csv")) for k in ("mirror", "ev", "eval")}'; assert V13LINE in s14
s14 = s14.replace(V13LINE, 'V13 = {"eval": V13_EVAL}')
s15 = open(os.path.join(HERE, "v15_src.py"), encoding="utf-8").read()
READ14 = 'src = open(os.path.join(HERE, "v14_ec_vote_v1.py"), encoding="utf-8").read()'; assert READ14 in s15
s15 = s15.replace(READ14, "src = S14_PATCHED")
NS["S14_PATCHED"] = s14
cut = s15.index('CUR_GEO[0] = "mirror"; rows = []')
exec(s15[:cut], NS)
exec('''
CUR_GEO[0] = "eval"; w = fitw(np.zeros(N, bool)); rows = []
for b in [i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13", "F47")]:
    r = pick(b, TRAIN.copy(), set(), w)
    if r: rows.append(r)
E = pd.DataFrame(rows)
''', NS)
E = NS["E"]; assert not os.path.exists(OUTF), "refusing to overwrite " + OUTF
E.to_csv(OUTF, index=False); print("wrote", OUTF, len(E))
