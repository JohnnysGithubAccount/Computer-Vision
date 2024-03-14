import tensorflow as tf

from configs import model_path, weight_path, image_path

import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np
import string


model_dir = model_path + r"\model_3333_64_512.h5"
weight_dir = weight_path + r"\best_weight_3333_64_512.h5"

print("[INFO] loading trained model")
model = tf.keras.models.load_model(model_dir)
model.load_weights(weight_dir)

# model.summary()

image_dir = image_path
# Original
image = cv2.imread(filename=image_dir)
# Gray - reduce noise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Blurred
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Edged detection map
edged = cv2.Canny(blurred, 30, 150)

# cv2.imshow("Image", edged)
# # Wait for a key press to close the window
# cv2.waitKey(0)
# # Destroy all windows
# cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method='left-to-right')[0]

chars = list()

for c in cnts:
    # Compute the box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    if (w>=5 and w<=150) and (h>=15 and w<=120):
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        if tW > tH:
            thresh = imutils.resize(thresh, width=32)

        else:
            thresh = imutils.resize(thresh, width=32)

        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)

        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0,0 ))
        padded = cv2.resize(padded, (32, 32))

        padded = padded.astype("float32")/255.0
        padded = np.expand_dims(padded, axis=-1)

        chars.append((padded, (x, y, w, h)))

boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype='float32')

preds = model.predict(chars)

labelNames = "0123456789"
labelNames += string.ascii_uppercase
labelNames = [l for l in labelNames]

for (pred, (x, y, w, h)) in zip(preds, boxes):
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]

    print(f"[INFOR] {label} - {prob * 100}%")
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Image", image)
cv2.waitKey(0)