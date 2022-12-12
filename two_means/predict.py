import pandas as pd
import numpy as np
from label_data import Labeler
from sklearn.preprocessing import MinMaxScaler
from check_code import check_split
import math, random
from check_code import check_train_data_shape, check_total
from lstm_model import generate_model, make_predictions

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

    def get_dataset_x(self, train_data):
        """
            Creates slices of len self.CANDLES_HISTORY, without including the
            last column. So, each position in x_train is a
            (self.CANDLES_HISTORY, len(data.columns) - 1) len vector

            The position x_train[0] corresponds to the first position in the
            dataframe. Notice that if we compute some indicator, some rows
            are going to be deleted because the .dropna()

            Example x_train, with self.CANDLES_HISTORY=3 ->
            [[5609.1    5612.1    5609.     5611.2    5631.742  5627.2265]
            [5611.1    5612.4    5607.5    5611.5    5631.479  5626.9825]
            [5611.4    5614.     5611.1    5613.1    5631.228  5626.7605]]

            Be aware of the scaler, it's going to convert the values from the 
            original data.
        """
        x_train = []
        for i in range(self.CANDLES_HISTORY, len(train_data)):
            x_train.append(train_data[i-self.CANDLES_HISTORY:i, :-1])
        return np.array(x_train)

    def get_dataset_y(self, scaled_data):
        """
            Here we are taking only the last colum of the dataset, which corresponds to the
            expected value. Notice that for each slice of data in every position of the
            x_train vector is only assigned a single value in the y_train vector.

            Example y_train[0] -> 1
            Corresponds to x_train[0] ->
            [[5609.1    5612.1    5609.     5611.2    5631.742  5627.2265]
            [5611.1    5612.4    5607.5    5611.5    5631.479  5626.9825]
            [5611.4    5614.     5611.1    5613.1    5631.228  5626.7605]]
        """
        return np.array(scaled_data[self.CANDLES_HISTORY:len(scaled_data), -1])

    def get_scaled_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        X = self.get_dataset_x(scaled_data)
        Y = self.get_dataset_y(scaled_data)
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
        """ Takes the 80% of the data as train data """
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
        """ Takes the 20% of the data as train data """
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
        make_predictions(model, {
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test
        })

    def run(self, data):
        self.get_scaled_data(data)
        self.get_train_data()
        self.get_test_data()
        self.get_model()

data = Labeler(
    "NAS100_M10_201707030100_202209292350.csv", "200_200_totals.csv"
).run()
useful_columns = [x for x in data.columns if x not in ["date", "spread"]]
data = data[useful_columns]
clasify = Clasify().run(data)

