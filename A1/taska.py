import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Read the label file to python and convert all the -1s into 0s for simpler future binary training
def read_label(label_path):
    df = pd.read_table(label_path)
    label = df.replace(-1, 0)
    return label


# Import and preprocessing training data into a matrix
def import_train_data(label, data_path):
    train_image = []
    for i in tqdm(range(label.shape[0])):
        img = image.load_img(data_path + label['img_name'][i], target_size=(178, 178, 3))
        img = image.img_to_array(img)
        img = img / 255
        train_image.append(img)
    X = np.array(train_image)
    print(X.shape)
    return X


# Import and preprocessing testing data into a matrix
def import_test_data(label, data_path):
    test_image = []
    for i in tqdm(range(label.shape[0])):
        img = image.load_img(data_path + label['img_name'][i], target_size=(178, 178, 3))
        img = image.img_to_array(img)
        img = img / 255
        test_image.append(img)
    X = np.array(test_image)
    print(X.shape)
    return X


# Extract gender labels from label file, forming a matrix
def y_a1(label):
    y = np.array(label.drop(['Unnamed: 0', 'img_name', 'smiling'], axis=1))
    print(y.shape)
    return y


# Extract smiling labels from label file, forming a matrix
def y_a2(label):
    y = np.array(label.drop(['Unnamed: 0', 'img_name', 'gender'], axis=1))
    print(y.shape)
    return y


# Split the dataset into training and validation sets
def train_valid_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    return X_train, X_test, y_train, y_test


# Set up the model structure
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
    model.add(Dense(1, activation='sigmoid'))   #Last layer setting 1 for one category of gender or smiling
    # Last layer setting 1 for one category of gender or smiling. Using sigmoid for binary cases
    print(model.summary())
    return model


# Feed datasets into model for training and save the final model
def train_save(model, X_train, X_test, y_train, y_test, model_path):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
    model.save(model_path)
    acc = history.history['accuracy']
    return acc[-1]  # Return the accuracy of the final saved model


# Evaluate the saved model by the testing datasets
def evaluate(model_path, X, y):
    model = load_model(model_path)
    acc = model.evaluate(X, y)
    return acc[-1]  # Return the accuracy of the testing for the model
