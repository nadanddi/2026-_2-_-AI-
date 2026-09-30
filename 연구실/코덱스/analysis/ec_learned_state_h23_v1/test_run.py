import unittest
import importlib.util
from pathlib import Path
P=Path(__file__).with_name('run.py')

class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not P.exists():raise AssertionError('H23 implementation does not exist yet')
        spec=importlib.util.spec_from_file_location('h23',P)
        cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def raw(self):
        m=self.m;rows=[]
        for farm in ('F13','F47'):
            for day in range(10,28):
                for h in range(8):
                    r={'row_id':f'{farm}_{day:03d}_{h:02d}'}
                    for j,c in enumerate(m.RAW):r[c]=10+j+.2*h+.1*day
                    r['in_hum']=65+h;r['in_co2']=400+day+3*h
                    rows.append(r)
        return m.pd.DataFrame(rows)

    def test_future_other_farm_order_and_prefix(self):
        m=self.m;x=self.raw();f=m.features(x)
        ids=[f'F13_012_{h:02d}' for h in range(5)]
        z=x.copy();change=z.row_id.str.startswith('F47')|z.row_id.isin([f'F13_012_{h:02d}' for h in range(5,8)])
        z.loc[change,m.RAW]=777
        cols=m.DIRECT+m.AUX+m.TARGET
        m.pd.testing.assert_frame_equal(f.set_index('row_id').loc[ids,cols],m.features(z).set_index('row_id').loc[ids,cols])
        m.pd.testing.assert_frame_equal(f,m.features(x.sample(frac=1,random_state=3)))
        m.pd.testing.assert_frame_equal(f.set_index('row_id').loc[ids,cols],m.features(x[x.row_id.isin(ids)]).set_index('row_id').loc[ids,cols])

    def test_current_targets_not_teacher_inputs(self):
        m=self.m;x=self.raw();rid='F13_012_04';a=m.features(x).set_index('row_id')
        x.loc[x.row_id.eq(rid),['in_hum','in_co2']]=[99,999]
        b=m.features(x).set_index('row_id')
        m.np.testing.assert_allclose(a.loc[rid,m.AUX].astype(float),b.loc[rid,m.AUX].astype(float),equal_nan=True)
        self.assertNotEqual(a.loc[rid,'dah'],b.loc[rid,'dah'])

    def test_day_reset_missing_hour_and_unknown_innovation(self):
        m=self.m;x=self.raw();x=x[~x.row_id.eq('F13_012_03')]
        f=m.features(x);z=f.set_index('row_id')
        self.assertTrue(m.np.isnan(z.loc['F13_012_04','dah']))
        self.assertTrue(m.np.isnan(z.loc['F13_012_00','in_hum_lag1']))
        pred=m.np.zeros((len(f),2));states=m.states(f,pred)
        self.assertTrue(m.np.isnan(states.loc[f.row_id.eq('F13_012_04'),'ah_innovation'].iloc[0]))
        self.assertEqual(states.loc[f.row_id.eq('F13_012_00'),'ah_expected_cum'].iloc[0],0)

    def test_purged_nested_crossfit(self):
        m=self.m;f=m.features(self.raw());exclude={('F13',12)}
        tr=f[m.split_mask(f,exclude)].copy()
        self.assertFalse(tr.farm.eq('F13').mul(tr.day.between(11,13)).any())
        p,audit=m.crossfit(tr,7,trees=8)
        self.assertEqual(p.shape,(len(tr),2));self.assertTrue(m.np.isfinite(p).all())
        for a in audit:
            held=set(map(tuple,a['held_days']));train=set(map(tuple,a['train_days']))
            self.assertFalse(train&{(f,d+k) for f,d in held for k in (-1,0,1)})

if __name__=='__main__':unittest.main()
