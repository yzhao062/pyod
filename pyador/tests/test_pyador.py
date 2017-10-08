import os
import unittest

import pandas as pd
from pyador import pyador
from pyador.local import TEST_DIR, TEST_FILE

test_file = os.path.join(TEST_DIR, TEST_FILE)
test_file = os.path.abspath(os.path.join(os.pardir, test_file))

print(test_file)

df = pd.read_csv(test_file)


class PyadorTestCases(unittest.TestCase):
    def test_argument(self):
        t1 = pyador.Pyador(df, 200)
        self.assertEqual(200, t1.n)

        t2 = pyador.Pyador(df, frac=0.2)
        self.assertEqual(0.2, t2.frac)

        with self.assertRaises(ValueError):
            pyador.Pyador(df, frac=1.2)

        with self.assertRaises(ValueError):
            pyador.Pyador(df, frac=-0.1)

        with self.assertRaises(ValueError):
            pyador.Pyador(df, 200, 0.2)


if __name__ == '__main__':
    unittest.main()
