import numpy as np
from keras.models import load_model
from helpers import landmark_flatten
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from data import get_data_from_file

class GestureClassifier:
    def __init__(self):
        self.model = None
    
    def load_model(self, load_path="gesture_classification_model.h5"):
        self.model = load_model(load_path)
    
    def train(self, data_path, save_path=None):
        X, y = get_data_from_file(data_path)
        one_hot_encoder = OneHotEncoder()
        y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).toarray()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        model = Sequential()
        model.add(Dense(32, input_dim=63, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, verbose=2, epochs=500, validation_data=(X_val, y_val))
        self.model = model
        if save_path:
            self.model.save(save_path)

    def pred(self, hand_landmarks):
        '''
        Return values:
            0 - paper
            1 - scissors
            2 - rock
        '''
        if not self.model:
            raise ValueError
        lm = landmark_flatten(hand_landmarks).reshape(1, -1)
        return np.argmax(self.model.predict(lm)[0])

# TODO: flipped or not?
if __name__ == '__main__':
    gc = GestureClassifier()
    data_path = 'data/'
    save_path = 'model/gc_v1.h5'
    gc.train(data_path, save_path)



