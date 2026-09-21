import unittest
import numpy as np
import pandas as pd
from causal_features import RAW
from hypothesis_features import build,WEATHER


def fixture():
    d=pd.DataFrame({'time':range(120),'farm':'F13','row_id':['a'+str(i) for i in range(120)]})
    for i,c in enumerate(RAW):d[c]=np.arange(120,dtype=float)+i
    for c in WEATHER:d[c]=d.time%24
    return d


class HypothesisTests(unittest.TestCase):
    def test_future_and_label_invariance(self):
        d=fixture();a,_=build(d)
        e=d.copy();e.loc[e.time>50,RAW]=-123
        e['sub_temp']=123;e['sub_ec']=456
        b,_=build(e)
        pd.testing.assert_frame_equal(a.loc['a0':'a50'],b.loc['a0':'a50'])

    def test_bridge_and_gaps(self):
        d=fixture();a,_=build(d)
        self.assertEqual(a.loc['a48','in_temp_hyp_bridge1'],d.loc[23,'in_temp'])
        self.assertEqual(a.loc['a49','in_temp_hyp_bridge1'],d.loc[48,'in_temp'])
        self.assertEqual(a.loc['a49','in_temp_hyp_bridge3'],d.loc[22,'in_temp'])
        self.assertEqual(a.loc['a48','in_temp_hyp_lag48'],d.loc[0,'in_temp'])
        b,_=build(d[d.time!=23])
        self.assertTrue(np.isnan(b.loc['a48','in_temp_hyp_bridge1']))

    def test_prefix_updates_without_future(self):
        d=fixture();d.loc[30,'out_temp']+=1;a,_=build(d)
        self.assertEqual(a.loc['a29','hyp_weather_prefix_complete_match'],1)
        self.assertEqual(a.loc['a30','hyp_weather_prefix_complete_match'],0)
        self.assertEqual(a.loc['a31','hyp_weather_prefix_complete_match'],0)
        b,_=build(d[d.time!=25])
        self.assertEqual(b.loc['a26','hyp_weather_prefix_complete_match'],0)

    def test_farm_and_order_isolation(self):
        d=fixture();a,_=build(d);e=d.copy();e.farm='F47';e.row_id='b'+e.time.astype(str);e[RAW]*=2
        b,cols=build(pd.concat([d,e]).sample(frac=1,random_state=1))
        pd.testing.assert_frame_equal(a,b.loc[a.index])
        for target in cols:
            for names in cols[target].values():self.assertEqual(len(names),len(set(names)))


if __name__=='__main__':unittest.main()
