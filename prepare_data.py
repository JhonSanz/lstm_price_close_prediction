import numpy as np
from termcolor import colored
import pandas as pd
import pandas_ta as ta
from constant import (
    CANDLES_PREDICTION, CANDLES_HISTORY,
)
from add_indicators import add_indicators


def get_data():
    df = pd.read_csv(
        'data.csv',
        names=[
            "Date", "Time", "High", "Open", "Close",
            "Low", "ZigzagMax", "ZigzagMin"
        ]
    )
    df["Date"] = df["Date"] + " " + df["Time"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    df["Zigzag"] = df["ZigzagMax"] + df["ZigzagMin"]
    df.loc[df["Zigzag"] > 0, "Zigzag"] = 1

    df = df[["Date", "High", "Open", "Close", "Low", "Zigzag"]]
    print(colored('Adding indicators...', 'yellow'))
    df["timestamp"] = df["Date"].values.astype(np.int64) // 10 ** 9
    add_indicators(df)
    df.dropna(inplace=True)

    print(colored(
        f'Predict {CANDLES_PREDICTION} candles in the future', 'yellow'
    ))
    df["prediction"] = df["Close"].shift(-CANDLES_PREDICTION)
    df["Zigzag"] = df["Zigzag"].shift(-CANDLES_PREDICTION)

    df = df.iloc[CANDLES_HISTORY:-1, :]
    df.reset_index(inplace=True, drop=True)
    df.to_csv("resources/test.csv", index=False)
    print(colored('Data created successfully in resources/test.csv', 'green'))

get_data()
