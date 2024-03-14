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
from configs import weight_file_name, model_name_file, plot_file_name

import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import build_montages
import numpy as np
import argparse
import cv2
import string
from time import sleep



ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True, help=az_path)
ap.add_argument("-m", "--model", type=str, required=True, help=model_path)
ap.add_argument("-p", "--plot", type=str, default="plot.png", help=plot_path)
args = vars(ap.parse_args())

print("[INFO] loading datasets...")
(azData, azLabels) = load_az_dataset(path=args["az"])
(digitsData, digitsLabels) = load_mnist_dataset()

azLabels += 10

print("[INFO] combining datasets...")
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

print("[INFO] transformming dataset...")
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

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

print("[INFO] compiling model ...")
optimizer = SGD(lr=init_lr, decay=init_lr/epochs)
# optimizer = Adam(lr=init_lr, decay=init_lr/epochs)
# optimizer = Adamax(lr=2*init_lr, decay=init_lr/(epochs/2))
# optimizer = tf.keras.optimizers.RMSprop(lr=init_lr, decay=init_lr/epochs)

checkpoint = ModelCheckpoint(filepath=weight_path + weight_file_name,
                             monitor = 'val_accuracy',
                             save_best_only=True,
                             model='max')

model = ResNet34(inputShape=input_shape,
                 num_classes=len(le.classes_),
                 block_layers=[3, 3, 3, 3],
                 filters_size=64,
                 units=512)
model.summary()
# model.load_weights(weight_path + weight_file_name)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=["accuracy"])

print("[INFO] training the network...")
his = model.fit(
    aug.flow(
        trainX,
        trainY,
        batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX)//batch_size,
    epochs=epochs,
    class_weight=classWeight,
    callbacks=[checkpoint],
    verbose=1,)

labelNames = "0123456789"
labelNames += string.ascii_uppercase
labelNames = [l for l in labelNames]

predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames,
                            labels=list(set(testY.argmax(axis=1)))))

# model.save(args["model"], save_format="h5")
model.save(model_path + model_name_file)

N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, his.history['loss'], label='train_loss')
plt.plot(N, his.history['val_loss'], label='val_loss')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args['plot'])
plt.savefig(plot_path + plot_file_name)
