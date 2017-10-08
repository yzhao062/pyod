import os
import pandas as pd
from pyador import pyador
from pyador import local as const

if __name__ == "__main__":
    dev_file = os.path.join(const.DEV_DIR, const.DEV_FILE)

    print (dev_file)
    df = pd.read_csv(dev_file)
    # x = 12
    # print(numerical_check(df[['gameorder', 'year_id']]))

    x = pyador.Pyador(df, 200)
    x.debug()

    print("do something")