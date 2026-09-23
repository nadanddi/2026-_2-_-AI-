# -*- coding: utf-8 -*-
import sys
import pandas as pd
a = pd.read_csv(sys.argv[1]); b = pd.read_csv(sys.argv[2])
for c in ["sub_temp", "sub_ec"]:
    d = (a[c] - b[c]).abs()
    print("%-9s max|d| %.3e  n_diff %d / %d" % (c, d.max(), int((d > 0).sum()), len(d)))
