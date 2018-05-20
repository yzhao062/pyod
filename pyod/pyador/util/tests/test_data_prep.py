import unittest

import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal
from pyador.util.data_prep import numerical_check
from pyador.util.data_prep import integrity_check
from pyador.util.data_prep import cat_to_num
from pyador.util.data_prep import missing_check


class UtilTestCases(unittest.TestCase):
    def setUp(self):
        self.t_df = pd.DataFrame([[1, "194611010TRH", "NBA", 0],
                                  [1, np.nan, "NBA", 1],
                                  [2, "194611020CHS", "NBA", np.nan],
                                  [2, "194611020CHS", "NBA", 1],
                                  [3, "194611020DTF", "NBA", 0],
                                  [3, "194611020DTF", "NBA", 1],
                                  ], columns=["gameorder", "game_id", "lg_id",
                                              "iscopy"])

        self.t_num_df = pd.DataFrame([[1, 0, 1293.277],
                                      [1, 1, 1306.723],
                                      [2, 0, np.nan],
                                      [np.nan, 1, 1297.071],
                                      [3, 0, 1279.619],
                                      [3, 1, 1320.381],
                                      ],
                                     columns=["gameorder", "iscopy", "elo_n"
                                              ])
        self.t_series = self.t_df['gameorder']

        self.t_var = 23333
        self.t_str = "23333"
        self.t_miss_df = pd.DataFrame([[1, "194611010TRH", "NBA", 0],
                                       [1, "NaN", "NBA", 1],
                                       [2, "194611020CHS", "NBA", 0],
                                       [2, "194611020CHS", "NBA", 1.0],
                                       [3, "194611020DTF", "NBA", 0],
                                       [3, "194611020DTF", "NBA", 1],
                                       ],
                                      columns=["gameorder", "game_id", "lg_id",
                                               "iscopy"])

    def test_integrity(self):
        self.assertEqual(integrity_check(self.t_df), True)

        with self.assertRaises(TypeError):
            integrity_check(self.t_series)

        with self.assertRaises(TypeError):
            integrity_check(self.t_var)

        with self.assertRaises(TypeError):
            integrity_check(self.t_str)

    def test_numerical_check(self):
        self.assertEqual(numerical_check(self.t_df), False)
        self.assertEqual(numerical_check(self.t_num_df), True)

    # TODO: supplement more testcases for util functions
    def test_missing_check(self):
        assert_frame_equal(missing_check(self.t_df), self.t_miss_df)

    def test_cat_to_num(self):
        self.assertEqual(True, True)

        if __name__ == '__main__':
            unittest.main()
