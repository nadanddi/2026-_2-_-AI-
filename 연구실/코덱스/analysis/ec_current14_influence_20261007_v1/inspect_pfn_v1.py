from pathlib import Path
import sys,json
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import numpy as np
p=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/components/DIAG10_0_pfn_1.npz'
with np.load(p,allow_pickle=False) as z:print([(n,z[n].shape,str(z[n].dtype)) for n in z.files])
