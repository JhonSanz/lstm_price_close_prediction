import numpy as np
from termcolor import colored
import pandas as pd
import pandas_ta as ta
from constant import (
    CANDLES_PREDICTION, CANDLES_HISTORY,
)
from add_indicators import add_indicators


def get_data():
    df = pd.read_csv('labeled_renko.csv')
    df["prediction"] = df["uptrend"].shift(-CANDLES_PREDICTION)
    df = df[["Date", "High", "Open", "Close", "Low", "prediction"]]

    print(colored('Adding indicators...', 'yellow'))
    add_indicators(df)
    df.dropna(inplace=True)

    print(colored(
        f'Predict {CANDLES_PREDICTION} candles in the future', 'yellow'
    ))

    df.loc[df["prediction"] == True, "prediction"] = 1
    df.loc[df["prediction"] == False, "prediction"] = 0
    df = df.iloc[CANDLES_HISTORY:-1, :]
    df.reset_index(inplace=True, drop=True)
    df.to_csv("resources/test.csv", index=False)
    print(colored('Data created successfully in resources/test.csv', 'green'))

get_data()
