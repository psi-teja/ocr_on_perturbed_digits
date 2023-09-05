import numpy as np
import os
import keras
import cv2
import shutil

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# randomly pick n number of samples for each digit
# to make sure our dataset is equally distributed
n = 100

# number of lines (perturbations)
n_of_lines = 3

# number of smudges (perturbations)
n_of_smudges = 3

# N is used to keep note of number of samples already picked for each digit
N = np.zeros(10)

# Output directory to store the image
normal_dir = 'digits'
noised_dir = "digits_noised"

# deleting the dir of all ready exists
if os.path.exists(normal_dir):
    shutil.rmtree(normal_dir)
if os.path.exists(noised_dir):
    shutil.rmtree(noised_dir)

# creating output dir
os.mkdir(normal_dir)
os.mkdir(noised_dir)


# Function to add random lines and smudges to an image
def add_noise(image):
    # Add random lines
    for _ in range(3):  # You can adjust the number of lines
        x1, y1 = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        x2, y2 = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        color = np.random.randint(128, 256)  # the color of the line in grayscale can be adjusted here
        thickness = 1  # You can adjust the line thickness
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    # Add random smudges
    for _ in range(3):  # You can adjust the number of smudges
        x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        color = np.random.randint(128, 256)  # the color of the smudges in grayscale can be adjusted here
        radius = 2  # You can adjust the smudge size
        cv2.circle(image, (x, y), radius, color, -1)


i = 0

# run the while loop till we get n samples for each digit
while min(N) < n:

    # randomly picking a sample from training data
    j = np.random.randint(0, len(X_train))
    digit = y_train[j]

    # go to next sample if we already have enough number of samples for a digit
    if N[digit] == n:
        continue

    i += 1

    N[digit] += 1

    data = X_train[j]

    # converting all black pixels to white pixels and vice versa
    image = 255 - data

    # Define the image filename
    image_filename = f'{i}_{digit}.png'  # You can change the file format (e.g., .png, .jpg, .bmp)

    cv2.imwrite(f"{normal_dir}/{image_filename}", image)

    add_noise(image)

    cv2.imwrite(f"{noised_dir}/{image_filename}", image)

    print(f"{i} - Image created and saved!")
