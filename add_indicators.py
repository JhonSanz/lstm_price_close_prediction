import pandas_ta as ta
from constant import CANDLES_HISTORY, COLUMNS_ORIGINAL


def add_indicators(df):
    # df.ta.sma(length=200, append=True)
    df.ta.sma(length=CANDLES_HISTORY, append=True)
    df.ta.sma(length=5, append=True)
    df.ta.stdev(length=CANDLES_HISTORY, close="Close", append=True)
    df.ta.rsi(length=CANDLES_HISTORY, close="Close", append=True)
    df.ta.stoch(
        k=CANDLES_HISTORY, close="Close", high="High",
        low="Low", append=True
    )
    return df


def get_column_indicators(df_columns):
    return list(set(df_columns).difference(COLUMNS_ORIGINAL))
