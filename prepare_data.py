from numpy import append
from termcolor import colored
import pandas as pd
import pandas_ta as ta


CANDLES_PREDICTION = 1

def get_data():
    df = pd.read_csv(
        'data.csv',
        names=[
            "Date", "Time", "High", "Open", "Close",
            "Low", "ZigzagMax", "ZigzagMin"
        ]
    )
    df["Date"] = df["Date"] + " " + df["Time"]
    df = df[["Date", "High", "Open", "Close", "Low"]]
    print(colored('Adding indicators', 'yellow'))

    # df.ta.ao(
    #     high="High", low="Low", slow=500,
    #     fast=1, append=True
    # )
    df.ta.sma(length=14, append=True)
    df.ta.rsi(close="Close", append=True)
    df.ta.atr(close="Close", high="High", low="Low", append=True)
    df.ta.stdev(close="Close", append=True)

    print(colored(f'Predict {CANDLES_PREDICTION} candles in the future', 'yellow'))
    df["prediction"] = df["Close"].shift(-CANDLES_PREDICTION)
    df = df.iloc[500:-1, :]
    df.reset_index(inplace=True, drop=True)
    df.to_csv("resources/test.csv")
    print(colored('Data created successfully', 'green'))


get_data()