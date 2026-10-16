# -*- coding: utf-8 -*-
"""submission_28 temperature anchor list (2026-10-15 집 클로드; CB1-T catalog 6.953 'UNI').
Anchor of an evaluation record = the FIELD linker FL's confident previous day (fl_eval_anchor.csv: p >= .8, labelled training record, |dT_air| <= 1.5);
if FL has none, the FLN linker's pick (ap_fln.py eval: 'no previous day' option) when p0 >= .5, the previous day is a LABELLED training record and |dT_air| <= 1.5.
Writes a csv with the columns make_temp_v22_anchor.py reads (row, anchor_st23, dT_air) + source.
Run:  python make_temp_v28_uni_anchor.py <fl_eval_anchor.csv> <fln_eval_links.csv> <out.csv>
"""
import os, sys
import pandas as pd


def main():
    fa, fn, out = sys.argv[1:4]
    assert not os.path.exists(out), "refusing to overwrite " + out
    A = pd.read_csv(fa, encoding="utf-8-sig")[["row", "prev_h0", "anchor_st23", "dT_air"]].assign(source="FL")
    N = pd.read_csv(fn)
    N = N[(N.p0 >= .5) & N.prev_is_labelled.astype(bool) & N.anchor_st23.notna() & (N.dT_air.abs() <= 1.5) & ~N.rec.isin(A.row)]
    N = N.rename(columns={"rec": "row", "prev0": "prev_h0"})[["row", "prev_h0", "anchor_st23", "dT_air"]].assign(source="FLN")
    U = pd.concat([A, N], ignore_index=True); U.to_csv(out, index=False, encoding="utf-8-sig")
    print("wrote %s | FL anchors %d + FLN-only %d = %d %s" % (out, len(A), len(N), len(U), sorted(N.row)))


if __name__ == "__main__":
    main()
