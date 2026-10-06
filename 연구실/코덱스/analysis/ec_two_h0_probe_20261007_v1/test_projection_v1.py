import unittest
from core_v1 import project_columns
class ProjectionTest(unittest.TestCase):
    def test_only_two_fields_and_original_order(self):
        cols=['in_co2','act_heating','in_co2_h0','act_heating_h0','in_temp_h0','act_heating_tdm','heat_run','season'];original=cols.copy()
        self.assertEqual(project_columns(cols),['in_co2','act_heating','in_temp_h0','act_heating_tdm','heat_run','season'])
        self.assertEqual(cols,original)
    def test_incomplete_or_duplicate_input_rejected(self):
        for cols in [['in_co2_h0','season'],['in_co2_h0','act_heating_h0','season','season']]:
            with self.assertRaises(ValueError):project_columns(cols)
if __name__=='__main__':unittest.main()
