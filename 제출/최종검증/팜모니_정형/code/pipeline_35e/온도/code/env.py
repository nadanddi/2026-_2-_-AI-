# -*- coding: utf-8 -*-
"""Bootstrap for the reproduction package.

Resolves the competition CSVs and the output directory relative to this
package, so the reviewer only has to drop the three files into data/.

  <package>/data/train_X.csv, train_y.csv, test_X.csv   (put them here)
  <package>/output/                                      (written by the run)

Set AGRI_DATA to point somewhere else if you prefer.

The optional block at the bottom registers DLL directories bundled inside
numpy/scipy/sklearn.  It is a no-op on a normal install and only matters on
machines without the Visual C++ / OpenMP runtimes, where lightgbm would
otherwise fail to load.

Usage:  import env   # must be the first project import
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DATA = os.environ.get("AGRI_DATA") or os.path.join(PKG, "data")
OUTDIR = os.path.join(PKG, "output")


def _register_dll_dirs():
    if not hasattr(os, "add_dll_directory"):
        return
    for mod in ("numpy", "scipy", "sklearn"):
        try:
            base = os.path.dirname(__import__(mod).__file__)
        except Exception:
            continue
        for d in glob.glob(os.path.join(base, ".libs")) + \
                 glob.glob(os.path.join(os.path.dirname(base), mod + ".libs")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def bootstrap():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    _register_dll_dirs()
    os.makedirs(OUTDIR, exist_ok=True)
    import common
    common.DATA = DATA
    return common


common = bootstrap()
