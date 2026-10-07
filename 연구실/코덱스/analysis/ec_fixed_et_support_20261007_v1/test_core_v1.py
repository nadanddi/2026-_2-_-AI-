import unittest
import numpy as np
from core_v1 import alpha,aggregate,shapley,coalitions,GROUPS
class CoreTests(unittest.TestCase):
    def test_prefix(self):
        p=np.arange(24)**2;expected=np.mean(.5*p+.5*np.cumsum(p)/np.arange(1,25))
        self.assertAlmostEqual(alpha(True)@p,expected,12)
    def test_support(self):
        t=np.array([[1,2],[1,3],[4,3]]);q=np.tile([[1,3]],(48,1));w=aggregate(t,q,True)
        np.testing.assert_allclose(w,np.tile([.25,.5,.25],(2,1)))
        np.testing.assert_allclose(aggregate(t,q,False),w)
    def test_all_orders(self):
        v=np.array([0,2,3,9.]);np.testing.assert_allclose(shapley(v,2),[4,5])
    def test_groups_and_endpoints(self):
        cols=sum(GROUPS.values(),[]);a=np.zeros((24,47));b=a.copy();b[:,cols.index('in_co2')]=2;b[:,cols.index('act_heating')]=3
        names,x=coalitions(a,b,cols);self.assertEqual(names,['co2_sensor','heating']);np.testing.assert_equal(x[-1],b)
if __name__=='__main__':unittest.main()
