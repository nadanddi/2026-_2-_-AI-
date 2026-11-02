import csv, sys, os, re
sys.stdout.reconfigure(encoding="utf-8")
M = r"C:\work\farmai\제출\최종검증\팜모니_이미지\data\manifests"


def load(s):
    for base in ("server", "local_pc"):
        p = os.path.join(M, base, s, "manifest.csv")
        if os.path.exists(p):
            return list(csv.DictReader(open(p, encoding="utf-8")))
    return []


def bn(x):
    return re.split(r"[\\/]", x)[-1].rsplit(".", 1)[0] if x else ""


TR = ["im_set_v4", "im_set_sd2", "im_set_v5", "im_set_gpt1", "im_set_gpt2", "im_set_newreal", "im_set_v7tr",
      "im_set_qwentr", "im_set_h3", "im_set_h3j", "im_set_gptr", "im_set_gptr_src", "im_set_gpt3"]
EV = ["im_set_r2", "im_set_r3", "im_set_r4", "im_set_r5"]
ev = {e: load(e) for e in EV}
TS, TG = set(), set()
for s in TR:
    T = load(s)
    S = {bn(r["source"]) for r in T} - {""}
    O = {bn(r.get("other", "")) for r in T} - {""}
    G = {r["group"] for r in T} - {""}
    TS |= S | O; TG |= G
    print(s, len(T), " ".join(f"{e[7:]}:src{len({bn(r['source']) for r in ev[e]} & (S | O))}" for e in EV))
for e in EV:
    R = ev[e]
    rs = [r for r in R if bn(r["source"]) in TS]
    rg = [r for r in R if r["group"] in TG]
    print(e, len(R), "src-in-train rows", len(rs), "(real", sum(r["label"] == "0" for r in rs), ")",
          "group-in-train rows", len(rg), "groups", len({r['group'] for r in R}), "shared", len({r['group'] for r in R} & TG))
from collections import Counter
R2S = {bn(r["source"]) for r in ev["im_set_r2"]}
for s in ["im_set_v5", "im_set_gpt1", "im_set_gpt2", "im_set_gpt3"]:
    T = [r for r in load(s) if bn(r["source"]) in R2S]
    print(s, Counter((r["label"], r["kind"]) for r in T).most_common(8))
r2k = Counter((r["label"], r["kind"]) for r in ev["im_set_r2"] if bn(r["source"]) in {bn(x["source"]) for x in load("im_set_v5")})
print("r2 rows sharing with v5", r2k.most_common(10))
print("---- GPT inputs")
allsrc = set(); allpair = set()
for s in ["im_set_gpt1", "im_set_gpt2", "im_set_gpt3", "im_set_gptr"]:
    T = [r for r in load(s) if r["label"] == "1"]
    S = {bn(r["source"]) for r in T} - {""}
    O = {bn(r.get("other", "")) for r in T} - {""}
    allsrc |= S; allpair |= O
    print(s, "fake rows", len(T), "unique src", len(S), "other", len(O))
print("union src", len(allsrc), "union src+other", len(allsrc | allpair))
v5 = load("im_set_v5"); print("v5 labels", Counter(r["label"] for r in v5))
