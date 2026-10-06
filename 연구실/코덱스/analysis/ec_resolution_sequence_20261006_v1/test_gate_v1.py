from pathlib import Path
import sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import pandas as pd,numpy as np
from stage3_4_v1 import eligible
a=pd.DataFrame(dict(act_vent_tdz=[.8,.799,.8,1.,float('nan'),1.],act_circfan_tdm=[9.,0.,10.,9.999,0.,float('nan')]))
np.testing.assert_array_equal(eligible(a),[True,False,False,True,False,False])
a['sub_ec']=999.;a['day']=999;a['risk']=-999
np.testing.assert_array_equal(eligible(a),[True,False,False,True,False,False])
print('PASS_PREFIX_GATE_BOUNDARIES_AND_LABEL_EXCLUSION',flush=True)
