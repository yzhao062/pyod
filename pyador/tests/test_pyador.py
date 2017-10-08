import os
import unittest

import pandas as pd
import numpy as np
from pyador import pyador

# from pyador.local import TEST_DIR, TEST_FILE
#
# test_file = os.path.join(TEST_DIR, TEST_FILE)
# test_file = os.path.abspath(os.path.join(os.pardir, test_file))
#
# print(test_file)
#
# df = pd.read_csv(test_file)

# test data
t_df = pd.DataFrame([[1, "194611010TRH", "NBA", 0],
                     [1, np.nan, "NBA", 1],
                     [2, "194611020CHS", "NBA", np.nan],
                     [2, "194611020CHS", "NBA", 1],
                     [3, "194611020DTF", "NBA", 0],
                     [3, "194611020DTF", "NBA", 1],
                     ], columns=["gameorder", "game_id", "lg_id", "iscopy"])


class PyadorTestCases(unittest.TestCase):
    def test_argument(self):
        t1 = pyador.Pyador(n=200)
        self.assertEqual(200, t1.n)

        t2 = pyador.Pyador(frac=0.2)
        self.assertEqual(0.2, t2.frac)

        with self.assertRaises(ValueError):
            pyador.Pyador(frac=1.2)

        with self.assertRaises(ValueError):
            pyador.Pyador(frac=-0.1)

        with self.assertRaises(ValueError):
            pyador.Pyador(200, 0.2)

    def test_data_quality(self):
        t1 = pyador.Pyador(200)
        X, num_X, le_dict = t1._data_check_fix(t_df)

        n_cat_var = t_df.select_dtypes(exclude=[np.number]).shape[1]
        (m, n) = t_df.shape

        # check the label encoder dictionary exist
        self.assertEqual(len(le_dict), n_cat_var)
        # check the converted dataframe
        self.assertEqual(X.shape, (m, n + n_cat_var))


if __name__ == '__main__':
    unittest.main()
