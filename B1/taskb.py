import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from tqdm import tqdm


def import_train_data_b(label, data_path):
    train_image = []
    for i in tqdm(range(label.shape[0])):
        img = image.load_img(data_path + '/' + label['file_name'][i], target_size=(200, 200, 3))
        img = image.img_to_array(img)
        img = img / 255
        train_image.append(img)
    X = np.array(train_image)
    print(X.shape)
    return X


def import_test_data_b(label, data_path):
    test_image = []
    for i in tqdm(range(label.shape[0])):
        img = image.load_img(data_path + '/' + label['file_name'][i], target_size=(200, 200, 3))
        img = image.img_to_array(img)
        img = img / 255
        test_image.append(img)
    X = np.array(test_image)
    print(X.shape)
    return X


def y_b1(label):
    y = pd.get_dummies(label, columns=['face_shape'])
    y = np.array(y.drop(['Unnamed: 0', 'file_name', 'eye_color'], axis=1))
    print(y.shape)
    return y


def y_b2(label):
    y = pd.get_dummies(label, columns=['eye_color'])
    y = np.array(y.drop(['Unnamed: 0', 'file_name', 'face_shape'], axis=1))
    print(y.shape)
    return y


def build_model_b():
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
    model.add(Dense(5, activation='sigmoid'))
    print(model.summary())
    return model
