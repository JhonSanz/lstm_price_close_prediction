
import pandas as pd
from stocktrends import indicators


df = pd.read_csv(
    'data.csv',
    names=[
        "date", "Time", "high", "open", "close",
        "low", "ZigzagMax", "ZigzagMin"
    ]
)
df["date"] = df["date"] + " " + df["Time"]
df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d %H:%M")
df = df[["high", "open", "close", "low", "date"]]
print(df.tail())

renko = indicators.Renko(df)
print('\n\nRenko box calcuation based on periodic close')
renko.brick_size = 35.6
renko.chart_type = indicators.Renko.PERIOD_CLOSE
data = renko.get_ohlc_data()
print(data.tail())
data.to_csv("renko.csv")
