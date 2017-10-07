from pyador.util.data_prep import numerical_check
import os
import pandas as pd
from pyador import pyador
from pyador import local as const

if __name__ == "__main__":
    dev_file = os.path.join(const.DEV_DIR, const.DEV_FILE)
    df = pd.read_csv(dev_file)
    # x = 12
    # print(numerical_check(df[['gameorder', 'year_id']]))

    x = pyador.Pyador(df, 200)
    x.debug()

    y = pyador.Pyador(df, frac=0.2)
    y.debug()

    z = pyador.Pyador(df, 200, 0.2)
    z.debug()
