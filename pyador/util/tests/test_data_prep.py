import unittest

from pyador.util.data_prep import numerical_check


class MyTestCase(unittest.TestCase):
    def test_numerical_test(self):
        self.assertEqual(True, False)
        # test on series and dataframe

    def test_integrity_check(self):
        self.assertEqual(True, False)
        # test different dtypes, dataframes and series


if __name__ == '__main__':
    unittest.main()
