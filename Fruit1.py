import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.layers import Dense,Dropout,Flatten,Conv2D

def get_images_labels(path):
    idxIn = 0
    labels = []
    images = np.zeros((500, 64, 64, 3))
    for filename in os.listdir(path):
        if filename.split('.')[1] == 'xml':
            tree = ET.parse(path + filename)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                fruit = o.find('name').text
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                img = Image.open(path+ filename[:-4] + ".jpg")
                img = img.crop((xmin, ymin, xmax, ymax))
                img = img.resize((64, 64), Image.ANTIALIAS)
                if np.asarray(img).shape == (64, 64, 3):
                    images[idxIn, :, :, :] = np.asarray(img)
                    labels.append(fruit)
                    idxIn += 1
    return images, labels

def get_model(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation="relu", input_shape=(64, 64, 3)))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="tanh"))
    model.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu"))
    model.add(Flatten())
    model.add(Dense(784, activation="tanh"))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation="softmax"))
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_val, y_val), verbose=2)

    plt.plot(hist.history["loss"], color="green", label="Train Loss")
    plt.plot(hist.history["val_loss"], color="red", label="Validation Loss")
    plt.title("Loss Plot")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss Values")
    plt.show()

    plt.plot(hist.history["accuracy"], color="black", label="Train Accuracy")
    plt.plot(hist.history["val_accuracy"], color="blue", label="Validation Accuracy")
    plt.title("Accuracy Plot")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy Values")
    plt.show()

    return model

if __name__ == '__main__':

    train_path = './train/'
    train_images, train_labels = get_images_labels(train_path)
    train_images =  train_images[0:len(train_labels)]
    train_labels = pd.get_dummies(train_labels).values
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2)
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    model = get_model(X_train, y_train, X_val, y_val)
    test_path = './test/'
    X_test, y_test = get_images_labels(test_path)
    X_test = X_test[0:len(y_test)]
    X_test = X_test / 255.0
    y_test = pd.get_dummies(y_test).values
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
