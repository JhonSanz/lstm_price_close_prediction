from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def generate_model(shape):
    model = Sequential()
    model.add(LSTM(
        units=50,
        return_sequences=True,
        input_shape=(shape[1], shape[-1])
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(units=15, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=15, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=15, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    return model