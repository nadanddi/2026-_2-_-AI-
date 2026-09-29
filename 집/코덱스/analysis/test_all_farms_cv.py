import unittest
from train_all_farms_cv import features, folds, RAW19, np, pd


class AllFarmTests(unittest.TestCase):
    def frame(self):
        pieces=[]
        for farm in ['F01','F13','F47']:
            t=np.arange(60*24);day=t//24;hour=t%24
            d=pd.DataFrame({c:t*.001+i for i,c in enumerate(RAW19)})
            d['farm']=farm;d['day']=day;d['hour']=hour;d['time']=t
            d['row_id']=[f'{farm}_{a:03}_{b:02}' for a,b in zip(day,hour)]
            d['in_temp']=20+np.sin(t/24);d['in_hum']=70.
            if farm=='F01':d.loc[:,[c for c in RAW19 if c not in ['in_temp','in_hum','in_co2','in_rad']]]=np.nan
            pieces.append(d)
        return pd.concat(pieces,ignore_index=True)

    def test_split_coverage_and_weather_purge(self):
        d=self.frame();assignment,splits=folds(d);m=d.set_index('row_id')
        self.assertTrue((sum(v.astype(int) for _,v in splits)==1).all())
        for tr,va in splits:
            self.assertFalse((tr&va).any())
            target=m.farm.isin(['F13','F47'])
            self.assertFalse(set(m.loc[tr&target,'day'])&set(m.loc[va&target,'day']))
            for farm in m.farm.unique():
                vd=m.loc[va&m.farm.eq(farm),'day']
                td=m.loc[tr&m.farm.eq(farm),'day']
                self.assertFalse(set(td)&set(vd+1))
                self.assertFalse(set(td)&set(vd-1))

    def test_missingness_and_future_invariance(self):
        d=self.frame();a,g=features(d)
        changed=d.copy();changed.loc[changed.time>80,RAW19]=999
        changed['sub_temp']=-999;changed['sub_ec']=999
        b,_=features(changed)
        ids=d.loc[d.time<=80,'row_id']
        pd.testing.assert_frame_equal(a.loc[ids],b.loc[ids])
        ids=d.loc[d.farm.eq('F01'),'row_id']
        self.assertTrue(a.loc[ids,'out_temp'].isna().all())
        self.assertTrue(a.loc[ids,'out_temp_absent'].eq(1).all())
        self.assertTrue(set(RAW19)<=set(g['full']))


if __name__=='__main__':unittest.main()
