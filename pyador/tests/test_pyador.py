import os
import unittest

import pandas as pd
from pyador import pyador
from pyador.local import DEV_FILE, DEV_DIR

dev_file = os.path.join(DEV_DIR, DEV_FILE)
df = pd.read_csv(dev_file)


class MyTestCase(unittest.TestCase):
    def test_argument(self):
        x = pyador.Pyador(df, 200)
        self.assertEqual(200, x.n)

        y = pyador.Pyador(df, frac=0.2)
        self.assertEqual(0.2, y.frac)

        with self.assertRaises(ValueError):
            pyador.Pyador(df, 200, 0.2)


if __name__ == '__main__':
    unittest.main()
