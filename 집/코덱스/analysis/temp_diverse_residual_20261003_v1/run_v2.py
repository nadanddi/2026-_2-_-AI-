"""Execution repair: use the already installed EC CatBoost package. No model/rule change."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/코덱스/local/ec_model_packages_20261002_v1'))
import run
if __name__=='__main__':run.main()
