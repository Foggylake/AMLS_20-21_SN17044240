import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_label(label_path):
    df = pd.read_table(label_path)
    label = df.replace(-1, 0)
    return label


def import_train_data(label, data_path):
    train_image = []
    for i in tqdm(range(label.shape[0])):
        img = image.load_img(data_path + '/' + label['img_name'][i], target_size=(178, 178, 3))
        img = image.img_to_array(img)
        img = img / 255
        train_image.append(img)
    X = np.array(train_image)
    print(X.shape)
    return X

def import_test_data(label, data_path):
    test_image = []
    for i in tqdm(range(label.shape[0])):
        img = image.load_img(data_path + '/' + label['img_name'][i], target_size=(178, 178, 3))
        img = image.img_to_array(img)
        img = img / 255
        test_image.append(img)
    X = np.array(test_image)
    print(X.shape)
    return X


def y_a1(label):
    y = np.array(label.drop(['Unnamed: 0', 'img_name', 'smiling'], axis=1))
    print(y.shape)
    return y

def y_a2(label):
    y = np.array(label.drop(['Unnamed: 0', 'img_name', 'gender'], axis=1))
    print(y.shape)
    return y


def train_valid_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    return X_train, X_test, y_train, y_test


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(8, 8), activation="relu", input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(8, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(8, 8), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(8, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model

def train_save(model, X_train, X_test, y_train, y_test, model_path):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
    model.save(model_path)
    acc = history.history['accuracy']
    return acc[-1]

def evaluate(model_path, X, y):
    model = load_model(model_path)
    acc = model.evaluate(X, y)
    return acc[-1]
