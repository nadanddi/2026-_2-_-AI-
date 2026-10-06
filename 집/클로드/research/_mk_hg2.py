s = open("ec3_HG1_high_gate_anchor_v1.py", encoding="utf-8").read()
a = s.index('"""'); b = s.index('"""', a + 3) + 3
doc = '''"""EC stage-3 HG2: HG1's high-EC gate (6.313: 9/9 better, P .040, normal days +5..+30 %) with a
DOMAIN GUARD from the user's material, judged ONCE by the pass-2-only protocol (6.279) plus a
normal-day protection condition.  Fixed before running; 2026-10-05 집 클로드.
Evidence: FZ1 (6.321): the domain score S_low = z(in_hum) + z(shade) + z(thermal) + z(fog)
- z(in_temp) - z(in_co2) (signs from 코멘트_변수별_영향.txt, no fitted weights) is higher on
FALSE gated days (AUC .77, one-sided p .022, both farms same direction); it failed the pre-set
.80 clue, and the zero threshold below was chosen after seeing FZ1 (natural sign split of a
z-sum, but post-hoc) -> this run is the honest test.
Candidate: SG2 everywhere; on rows with pm_h >= .9 and the best two anchors >= 1.0 AND the
hour-causal S_low <= 0: pred = p + 0.5 (mean(a1, a2) - pm_h).  S_low at hour h = the six
variables' means over the record's hours 0..h, z per farm per h from training REFERENCE
records' 0..h means (no test / validation statistics).  Pass-2 rows only.
Baseline: R3S + SG2 with the same new seeds 37 / 1212 / 4242.
Sets (pass-2 rows): DIAG10, DIAG10q fresh layout (9-record chunks: chunk = day // 9, fold =
chunk % 10, +-1 purge, lock excluded; never used before), EL1.
PASS iff (a) every seed x all three sets improve, (b) seed-mean block bootstrap P(worse) on
DIAG10q pass-2 rows < .025, AND (c) pass-2 NORMAL-day RMSE (label day mean < 1) not worse than
the baseline by more than 2 % in any seed x set cell.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_HG2_domain_guard_gate_v1.py
"""'''
s = s[:a] + doc + s[b:]
s = s.replace("SEEDS = (29, 909, 1111)", "SEEDS = (37, 1212, 4242)").replace('"hg1_ckpt"', '"hg2_ckpt"')
s = s.replace("        SIG[h] = S\n    return R, WV, hrs, SIG", '''        SIG[h] = S
    DV = ["in_hum", "act_shade", "act_thermal", "act_fog", "in_temp", "in_co2"]
    XX = X.sort_values(["farm", "day", "hour"]).copy(); gg = XX.groupby(["farm", "day"])
    for v in DV:
        XX["cm_" + v] = gg[v].transform(lambda z: z.expanding().mean())
    DM = XX.set_index(["farm", "day", "hour"])[["cm_" + v for v in DV]]
    return R, WV, hrs, SIG, DM''')
s = s.replace("def correction(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal):", '''SGN = [1, 1, 1, 1, -1, -1]


def s_low_table(DM, ref):
    """dict (farm, h) -> (mu, sd) from reference records' 0..h means."""
    T = {}
    for f in ("F13", "F47"):
        for h in range(24):
            M = DM.xs(h, level="hour").loc[f]
            M = M[[ (f, d) in ref for d in M.index]]
            T[(f, h)] = (M.mean().values, M.std().replace(0, 1).values)
    return T


def correction(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal, DM=None, ST=None):''')
s = s.replace("            if pm >= .9 and a1 >= 1.0 and np.isfinite(a2) and a2 >= 1.0:", '''            mu_, sd_ = ST[(f, h)]
            slow = float(np.nansum(np.array(SGN) * (DM.loc[(f, d, h)].values - mu_) / sd_))
            if pm >= .9 and a1 >= 1.0 and np.isfinite(a2) and a2 >= 1.0 and slow <= 0:''')
s = s.replace("    R, WV, hrs, SIG = prepare_structure()", "    R, WV, hrs, SIG, DM = prepare_structure()")
s = s.replace('''            frame["sgb_%d" % s], frame["sg_%d" % s] = correction(frame, "base_%d" % s, vd, lockd, ec, R, WV, hrs, SIG, ref, cal)''',
              '''            frame["sgb_%d" % s], frame["sg_%d" % s] = correction(frame, "base_%d" % s, vd, lockd, ec, R, WV, hrs, SIG, ref, cal, DM, ST)''')
s = s.replace("        cal = ref_calendar(R, WV, ref)\n", "        cal = ref_calendar(R, WV, ref)\n        ST = s_low_table(DM, ref)\n", 1)
s = s.replace('''        folds.append(("DIAG10z", k, {(f, int(d)) for f, d in zip(days.farm, days.day) if (d // 8) % 10 == k and (f, int(d)) not in lock}))''',
              '''        folds.append(("DIAG10q", k, {(f, int(d)) for f, d in zip(days.farm, days.day) if (d // 9) % 10 == k and (f, int(d)) not in lock}))''')
s = s.replace('"DIAG10z"', '"DIAG10q"').replace("DIAG10z pass-2", "DIAG10q pass-2")
s = s.replace("ec3_HG1_all.csv", "ec3_HG2_all.csv").replace("R3S+SG2 -> HG1 (new seeds)", "R3S+SG2 -> HG2 (new seeds)")
old = '''    print("\nHG1 decision:", "PASS" if ok and p < 0.025 else "FAIL", "(all seeds x sets better %s, P %.4f)" % (ok, p))'''
assert old in s, "decision line"
s = s.replace(old, '''    nok = True
    for v in ("DIAG10", "DIAG10q", "EL1"):
        A = O[O.validator == v].copy(); A = A[A.groupby(["farm", "day"]).sub_ec.transform("mean") < 1]
        for s in SEEDS:
            nok &= r(A["sg_%d" % s] - A.sub_ec) <= 1.02 * r(A["sgb_%d" % s] - A.sub_ec)
    print("  normal-day protection (<= +2 %% in every cell): %s" % nok)
    print("\nHG2 decision:", "PASS" if ok and p < 0.025 and nok else "FAIL", "(all better %s, P %.4f, normal protected %s)" % (ok, p, nok))''')
open("ec3_HG2_domain_guard_gate_v1.py", "w", encoding="utf-8").write(s)
print("ok")
