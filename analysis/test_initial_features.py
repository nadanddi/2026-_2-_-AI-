import unittest
import numpy as np
import pandas as pd
from test_causal_features import CausalityTests
from initial_features import build


class InitialFeaturesTests(unittest.TestCase):
    def test_future_targets_order_and_farms(self):
        d=CausalityTests().frame();a,groups=build(d)
        e=d.copy();e.loc[e.time>50,a.columns.intersection(d.columns)]=999
        e['sub_ec']=999;e['sub_temp']=-999
        b,_=build(e);pd.testing.assert_frame_equal(a.loc['a0':'a50'],b.loc['a0':'a50'])
        second=d.copy();second.farm='F47';second.row_id='b'+second.time.astype(str)
        second['in_temp']=-20
        b,_=build(pd.concat([d,second]).sample(frac=1,random_state=5))
        pd.testing.assert_frame_equal(a,b.loc[a.index])
        self.assertTrue(all(len(v)==len(set(v)) for v in groups.values()))

    def test_gap_and_curtain_direction(self):
        d=CausalityTests().frame();d=d[d.time!=49].copy()
        d['act_shade']=0;d['act_thermal']=0;d['act_heating']=50
        a,_=build(d)
        self.assertTrue(np.isnan(a.loc['a50','in_temp_lag1']))
        self.assertTrue(np.isnan(a.loc['a50','rad_energy3']))
        self.assertEqual(a.loc['a50','rad_open'],0)
        self.assertEqual(a.loc['a50','heating_closed'],50)
        d['act_shade']=100;d['act_thermal']=100
        b,_=build(d)
        self.assertEqual(b.loc['a50','rad_open'],d.loc[d.time==50,'out_rad'].iloc[0])
        self.assertEqual(b.loc['a50','heating_closed'],0)


if __name__=='__main__':unittest.main()
