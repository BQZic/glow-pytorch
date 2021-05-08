'''

Birthday Paradox Test

1. Randomly sample N images from generated results
2. Measure the Euclidean distance between images
3. Report top 20 image distances

'''

import os, random
import glob

NUM_SAMPLED_IMAGES = 20
FILE_NAME = "/content/drive/MyDrive/Data/Test3/*.jpg"

selected_images = random.sample(glob.glob(FILE_NAME), NUM_SAMPLED_IMAGES)

for image in enumerate(selected_images, 1):
    continue