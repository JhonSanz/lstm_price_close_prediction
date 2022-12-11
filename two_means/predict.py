import pandas as pd
import numpy as np
from label_data import Labeler
from sklearn.preprocessing import MinMaxScaler
from check_code import check_split
import math, random
from check_code import check_train_data_shape, check_total
from lstm_model import generate_model

class Clasify:
    CANDLES_HISTORY = 20
    PERCENTAGE_DATA = 0

    def __init__(self):
        self.model = None
        self.data = None
        self.scaler = None
        self.X = None
        self.Y = None
        self.x_train = None
        self.y_train = None
        self.y_test = None
        self.y_test = None

    def get_dataset(self, train_data):
        x_train = []
        for i in range(self.CANDLES_HISTORY, len(train_data)):
            x_train.append(train_data[i-self.CANDLES_HISTORY:i, :-1])
        return np.array(x_train)

    def get_scaled_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        X = self.get_dataset(scaled_data)
        Y = np.array(scaled_data[self.CANDLES_HISTORY:len(scaled_data), -1])
        assert check_split(data, scaler, self.CANDLES_HISTORY, X, Y)
        assert check_train_data_shape(X, Y)
        self.X = X
        self.Y = Y

    def zip_data(self, x, y):
        zipped = list(zip(x, y))
        random.shuffle(zipped)
        zipped_len_train = math.ceil(len(zipped) - len(zipped) * self.PERCENTAGE_DATA)
        zipped = zipped[:zipped_len_train]
        return zip(*zipped)

    def get_train_data(self):
        training_data_len = math.ceil(len(self.Y) * .80)
        x_train = self.X[:training_data_len]
        y_train = self.Y[:training_data_len]
        x_train, y_train = self.zip_data(x_train, y_train)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        assert check_train_data_shape(x_train, y_train)
        self.x_train = x_train
        self.y_train = y_train

    def get_test_data(self):
        training_data_len = math.ceil(len(self.Y) * .80)
        divition = training_data_len  # + (len(Y[training_data_len:]) / 5)
        shrink = math.ceil(divition)
        x_test = self.X[shrink:-1]
        y_test = self.Y[shrink:-1]
        x_test, y_test = self.zip_data(x_test, y_test)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        assert check_train_data_shape(x_test, y_test)
        self.x_test = x_test
        self.y_test = y_test
        assert check_total(
            y_test, self.Y, self.PERCENTAGE_DATA, training_data_len,
            self.y_train, shrink
        )

    def get_model(self):
        model = generate_model(self.X.shape)

    def run(self, data):
        self.get_scaled_data(data)
        self.get_test_data()
        self.get_train_data()
        self.get_model()

if __name__ == "main":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    data = Labeler(
        "NAS100_M10_201707030100_202209292350.csv", "200_200_totals.csv"
    ).run()
    useful_columns = [x for x in data.columns if x not in ["date", "spread"]]
    data = data[useful_columns]
    clasify = Clasify().run(data)

