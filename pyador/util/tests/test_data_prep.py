import unittest

import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal
from pyador.util.data_prep import _numerical_check
from pyador.util.data_prep import _integrity_check
from pyador.util.data_prep import _cat_to_num
from pyador.util.data_prep import _missing_check

# test data
t_df = pd.DataFrame([[1, "194611010TRH", "NBA", 0],
                     [1, np.nan, "NBA", 1],
                     [2, "194611020CHS", "NBA", np.nan],
                     [2, "194611020CHS", "NBA", 1],
                     [3, "194611020DTF", "NBA", 0],
                     [3, "194611020DTF", "NBA", 1],
                     ], columns=["gameorder", "game_id", "lg_id", "iscopy"
                                 ])
t_num_df = pd.DataFrame([[1, 0, 1293.277],
                         [1, 1, 1306.723],
                         [2, 0, np.nan],
                         [np.nan, 1, 1297.071],
                         [3, 0, 1279.619],
                         [3, 1, 1320.381],
                         ], columns=["gameorder", "iscopy", "elo_n"
                                     ])
t_series = t_df['gameorder']

t_var = 23333
t_str = "23333"


class UtilTestCases(unittest.TestCase):
    def test_integrity(self):
        self.assertEqual(_integrity_check(t_df), True)

        with self.assertRaises(TypeError):
            _integrity_check(t_series)

        with self.assertRaises(TypeError):
            _integrity_check(t_var)

        with self.assertRaises(TypeError):
            _integrity_check(t_str)

    def test_numerical_check(self):
        self.assertEqual(_numerical_check(t_df), False)
        self.assertEqual(_numerical_check(t_num_df), True)

    # TODO: supplement more testcases for util functions
    def test_missing_check(self):
        exp_df = pd.DataFrame([[1, "194611010TRH", "NBA", 0],
                               [1, "NaN", "NBA", 1],
                               [2, "194611020CHS", "NBA", 0],
                               [2, "194611020CHS", "NBA", 1.0],
                               [3, "194611020DTF", "NBA", 0],
                               [3, "194611020DTF", "NBA", 1],
                               ], columns=["gameorder", "game_id", "lg_id",
                                           "iscopy"])
        assert_frame_equal(_missing_check(t_df), exp_df)

    def test_cat_to_num(self):
        self.assertEqual(True, True)

        if __name__ == '__main__':
            unittest.main()
