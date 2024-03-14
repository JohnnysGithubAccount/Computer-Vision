import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from datasets import load_az_dataset
from datasets import load_mnist_dataset

# from models import ResNet, EfficientNet
from models import ResNet34

from configs import az_path, model_path, plot_path, weight_path
from configs import epochs, batch_size, init_lr
from configs import input_shape

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from imutils import build_montages
import numpy as np
import argparse
import cv2
import string
from time import sleep


print("[INFO] loading datasets...")
(azData, azLabels) = load_az_dataset(az_path)
(digitsData, digitsLabels) = load_mnist_dataset()

azLabels += 10

print("[INFO] combining datasets...")
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

print("[INFO] transoformming dataset...")
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

data = np.expand_dims(data, axis=-1)
data /= 225.0

# Conver the labels from integers to vectors
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

classTotals = labels.sum(axis=0)
classWeight = {}

for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max()/classTotals[i]

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

print(testX[0])
print(len(testX[0]))

labelNames = "0123456789"
labelNames += string.ascii_lowercase
labelNames = [l for l in labelNames]

model = tf.keras.models.load_model(r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision\OpticalCharacterRecognition\BasicWithTensorflowKeras\TrainedModel\model_3333_64_512.h5")
model.load_weights(weight_path + r"\best_weight_3333_64_512.h5")
model.evaluate(testX, testY, batch_size=batch_size)
images = []

for i in np.random.choice(np.arange(0, len(testY)), size=(100,)):
    probs = model.predict([testX[np.newaxis, i]])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    image = (testX[i]*255).astype("uint8")
    color = (0, 255, 0)

    if prediction[0] != np.argmax(testY[i]):
        color = (0, 0, 255)

    image = cv2.merge([image]*3)
    image = cv2.resize(image, (75, 75), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    images.append(image)

montage = build_montages(images, (75, 75), (10, 10))[0]

cv2.imshow("OCR Results", montage)
cv2.waitKey(0)