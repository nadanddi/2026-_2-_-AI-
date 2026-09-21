import unittest
import numpy as np
import pandas as pd
from causal_features import RAW, build_all


class CausalityTests(unittest.TestCase):
    def frame(self):
        d=pd.DataFrame({'time':range(100),'farm':'F13','row_id':['a'+str(i) for i in range(100)]})
        for i,c in enumerate(RAW):d[c]=np.arange(100,dtype=float)+i
        return d

    def test_future_invariance_and_no_targets(self):
        d=self.frame(); a,_=build_all(d)
        changed=d.copy(); changed.loc[changed.time>50,RAW]=-9999
        changed['sub_temp']=123456; changed['sub_ec']=-123456
        b,_=build_all(changed)
        pd.testing.assert_frame_equal(a.loc['a0':'a50'],b.loc['a0':'a50'])

    def test_elapsed_hour_gaps(self):
        d=self.frame(); d=d[d.time!=49]
        a,_=build_all(d)
        self.assertTrue(np.isnan(a.loc['a50','in_temp_lag1']))
        self.assertEqual(a.loc['a50','in_temp_lag2'],d.loc[d.time==48,'in_temp'].iloc[0])

    def test_cross_farm_isolation(self):
        d=self.frame(); second=d.copy(); second.farm='F47'; second.row_id='b'+second.time.astype(str)
        second[RAW]=second[RAW]*10
        a,_=build_all(d); combined,_=build_all(pd.concat([d,second]))
        pd.testing.assert_frame_equal(a,combined.loc[a.index])

    def test_permutation_invariance(self):
        d=self.frame(); a,_=build_all(d); b,_=build_all(d.sample(frac=1,random_state=7))
        pd.testing.assert_frame_equal(a,b)


if __name__=='__main__':unittest.main()
