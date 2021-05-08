'''

Birthday Paradox Test

1. Randomly sample N images from generated results
2. Measure the Euclidean distance between images
3. Report top 20 image distances

'''

import os, random
import glob
from PIL import Image
from collections import Counter
import numpy as np

def L2Norm(H1,H2):
    distance =0
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    return np.sqrt(distance)

def convertImage(img1):
    reference_image_1 = Image.open(img1)
    reference_image_arr = np.asarray(reference_image_1)
    flat_array_1 = reference_image_arr.flatten()
    RH1 = Counter(flat_array_1)
    H1 = []
    for i in range(256):
        if i in RH1.keys():
            H1.append(RH1[i])
        else:
            H1.append(0)
    return H1

def compareImages(img1, img2):
    H1 = convertImage(img1)
    H2 = convertImage(img2)
    return L2Norm(H1, H2)

NUM_SAMPLED_IMAGES = 20
FILE_NAME = "/content/drive/MyDrive/Data/Test3/*.jpg"

selected_images = random.sample(glob.glob(FILE_NAME), NUM_SAMPLED_IMAGES)

for image in enumerate(selected_images, 1):
    print(image)