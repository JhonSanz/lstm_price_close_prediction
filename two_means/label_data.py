import pandas as pd
import pandas_ta as ta
from data_getter import DataGetter

class Labeler:
    OPERABLE = 1
    UNOPERABLE = 0

    def __init__(self, file_data, file_orders):
        self.file_data = file_data
        self.file_orders = file_orders
        self.labeled = None

    def read_data(self):
        self.file_orders = pd.read_csv(self.file_orders)
        self.file_data = DataGetter().get_initial_data(self.file_data)
        self.file_data["operable"] = self.OPERABLE
        self.file_data["sma_high"] = self.file_data.ta.sma(close=self.file_data["high"], length=200)
        self.file_data["sma_low"] = self.file_data.ta.sma(close=self.file_data["low"], length=200)
        self.file_data.dropna(inplace=True)
        self.file_data.reset_index(inplace=True, drop=True)

    def label_data(self):
        self.file_data.loc[
            (
                (self.file_data["sma_high"] <= self.file_data["open"]) &
                (self.file_data["open"] <= self.file_data["sma_low"])
            ) |
            (
                (self.file_data["sma_high"] <= self.file_data["high"]) &
                (self.file_data["high"] <= self.file_data["sma_low"])
            ) |
            (
                (self.file_data["sma_high"] <= self.file_data["low"]) &
                (self.file_data["low"] <= self.file_data["sma_low"])
            ) |
            (
                (self.file_data["sma_high"] <= self.file_data["close"]) &
                (self.file_data["close"] <= self.file_data["sma_low"])
            ),
            "operable"
        ] = self.UNOPERABLE
        for (date_open, date_close, diff) in zip (
            self.file_orders["date_open"], self.file_orders["date_close"], 
            self.file_orders["diff"]
        ):
            self.file_data.loc[
                (self.file_data["date"] >= date_open) &
                (self.file_data["date"] <= date_close) &
                (diff < 0),
                "operable"
            ] = self.UNOPERABLE
        
        columns = self.file_data.columns.tolist()
        columns = [x for x in columns if x != "operable"]
        self.file_data = self.file_data[columns + ["operable"]]
        return self.file_data

    def run(self):
        self.read_data()
        return self.label_data()
