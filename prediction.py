import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess import images, path



model = tf.keras.models.load_model(f"{path}/model/model.h5")

predictions = model.predict(images)

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

for i, pred in  enumerate(predictions):
    pred = labels[np.argmax(pred)]
    print(f"Image {i + 1} is a {pred}")
