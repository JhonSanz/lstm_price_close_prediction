import pandas as pd
import random, math

def check_split(df, scaler, CANDLES_HISTORY, X, Y):
    RANDOM_VALUE = random.randint(0, len(Y))
    check_split_df = pd.DataFrame(X[CANDLES_HISTORY + RANDOM_VALUE])
    check_split_df["prediction"] = Y[RANDOM_VALUE:CANDLES_HISTORY + RANDOM_VALUE]
    check_split_df = scaler.inverse_transform(check_split_df.values)
    check_split_df = pd.DataFrame(check_split_df)
    return all(
        check_split_df.loc[0].values.round() ==
        df.loc[CANDLES_HISTORY + RANDOM_VALUE].values.round()
    )

def check_train_data_shape(X, Y):
    return X.shape[0] == Y.shape[0]



def check_total(y_test, Y, PERCENTAGE_DATA, training_data_len, y_train, shrink):
    def get_len(samples):
        return math.ceil(len(samples) * PERCENTAGE_DATA)

    return (
        (y_test.shape[0] + get_len(Y[training_data_len:]))
        +
        (
            shrink - (
                y_train.shape[0] +
                get_len(Y[:training_data_len])
            )
        )
        +
        (y_train.shape[0] + get_len(Y[:training_data_len]))
    )
