
import pandas as pd
from termcolor import colored


def get_trends(df):
    trends = df.copy()
    trends["uptrend"] = df["uptrend"].diff()
    trends = trends[trends["uptrend"] != 0]
    trends.reset_index(inplace=True)
    trends = trends[["date"]]
    return trends


def get_dataset(df_original, trends):
    df = df_original.copy()
    df["uptrend"] = None
    trends_ = list(trends["date"])
    for index, trend in enumerate(list(zip(trends_, trends_[1:]))):
        df.loc[
            (df["Date"] >= trend[0]) & (df["Date"] < trend[-1]),
            "uptrend"
        ] = index % 2 == 0
    df.dropna(inplace=True)
    return df


renko = pd.read_csv("renko.csv")
renko = renko.iloc[:, 1:]
renko.loc[renko["uptrend"] == True, "uptrend"] = 1
renko.loc[renko["uptrend"] == False, "uptrend"] = 0
trends = get_trends(renko)

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

df = df[["Date", "High", "Open", "Close", "Low"]]

df = get_dataset(df, trends)
df.to_csv(
    "labeled_renko.csv",
    columns=["Date", "High", "Open", "Close", "Low", "uptrend"],
    index=False
)