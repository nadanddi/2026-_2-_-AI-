# -*- coding: utf-8 -*-
"""Parallel fold runner (speed only; results identical to a single run).
Usage: PYTHONPATH="" py -3.12 -u run_parallel_folds_v1.py <script.py> <worker k> <workers N>
A checkpoint whose fold file name hashes to another worker is treated as already done, so N workers
started together compute disjoint folds.  After all workers finish, run the original script once
(all checkpoints exist -> it only aggregates and prints the judgement)."""
import env  # noqa: F401
import importlib.util, os, sys, zlib
script, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
spec = importlib.util.spec_from_file_location("target", os.path.join(os.path.dirname(os.path.abspath(__file__)), script))
sys.argv = ["x"]
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
_exists = os.path.exists


def exists(p):
    if p.endswith(".csv") and os.path.basename(os.path.dirname(p)) == os.path.basename(mod.CK):
        if zlib.crc32(os.path.basename(p).encode()) % n != k:
            return True
    return _exists(p)


mod.os.path.exists = exists
try:
    mod.main()
except Exception as e:  # aggregation with missing checkpoints is expected to fail in workers
    print("worker %d/%d finished folds; aggregation skipped (%s)" % (k, n, type(e).__name__), flush=True)
