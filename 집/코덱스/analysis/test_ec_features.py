import unittest
import pandas as pd
import numpy as np
from test_causal_features import CausalityTests
from ec_features import add_ec_features


class ECFeatureTests(unittest.TestCase):
    def test_future_invariance(self):
        d=CausalityTests().frame(); a,_=add_ec_features(d)
        changed=d.copy(); future=changed.time>50
        changed.loc[future,changed.columns.difference(['farm','time','row_id'])]=999
        changed['sub_ec']=777
        b,_=add_ec_features(changed)
        pd.testing.assert_frame_equal(a.loc['a0':'a50'],b.loc['a0':'a50'])

    def test_day_start_is_not_carried_between_days(self):
        d=CausalityTests().frame(); d=d[d.time.ne(48)]
        a,_=add_ec_features(d)
        self.assertTrue(np.isnan(a.loc['a49','in_temp_day_start']))
        self.assertEqual(a.loc['a47','in_temp_day_start'],d.loc[d.time==24,'in_temp'].iloc[0])

    def test_vpd_units_and_saturation(self):
        d=CausalityTests().frame(); d['in_temp']=20.; d['in_hum']=50.
        a,_=add_ec_features(d)
        self.assertAlmostEqual(a.iloc[0].in_vpd_kpa,1.169,places=3)
        d['in_hum']=100.
        a,_=add_ec_features(d)
        self.assertAlmostEqual(a.iloc[0].in_vpd_kpa,0.)
        self.assertAlmostEqual(a.iloc[0].in_dewpoint_c,20.)


if __name__=='__main__':unittest.main()
