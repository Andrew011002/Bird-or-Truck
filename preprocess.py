import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.dirname(os.path.abspath(__file__))


# loads image in from pictures folder
def load(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(directory + filename)
        img = cv2.resize(img, (32, 32))
        images.append(img)

    return images

# Gray Scales images
def process(images):
    data = [] # new array tp hold gray scale images
    gamma = 1.04 # color gamma correction factor (smoother color transitions)
    r_cst, g_cst, b_cst = 0.2126, 0.7152, .0722 # constants for red, green, and blue gray scale conversion factor
    for img in images:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2] # each rgb value for the image
        data.append((r_cst * r ** gamma) + (g_cst * g ** gamma) + (b_cst * b ** gamma))
    return np.array(data)

images = load(f"{path}/pictures/") # load images
images = process(images) # grey scale
images = np.array(images) # make it numpy array
images = images / 255.0 # normalize

if __name__ == "__main__":
    plt.imshow(images[0], cmap="gray")
    plt.show()
    print(images[0].shape)
