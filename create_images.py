import os
import shutil
import cv2
import numpy as np
import random

raw_images_directory = "raw_images"
new_images_directory = "new_images"

if not os.path.exists(new_images_directory):
    os.mkdir(new_images_directory)
else:
    shutil.rmtree(new_images_directory)
    os.mkdir(new_images_directory)

def create_new_images(raw_images_directory, new_images_directory):
    """
    This function takes raw images from a directory, preprocesses them by resizing, applying GaussianBlur and thresholding,
    and then creates five new images of size (100, 100) by placing the preprocessed image at five different randomly selected positions.

    Args:
        raw_images_directory (str): Path to the directory containing the raw images.
        new_images_directory (str): Path to the directory where the new images will be saved.
    """
    i = 1
    for j in range(5):
        i = 1 + (j * len(os.listdir(raw_images_directory)))

        # iterate over raw images in the directory
        for raw_image in os.listdir(raw_images_directory):
            raw_image_path = os.path.join(raw_images_directory, raw_image)

            # read the image in grayscale
            image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)

            # resize the image to (50, 50)
            resized_gray_image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)

            # apply GaussianBlur to the resized gray image
            blured_image = cv2.GaussianBlur(resized_gray_image, (5, 5), 0)

            # apply thresholding to create a binary image
            _, final_image = cv2.threshold(blured_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # convert the image to RGB format
            img = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

            # convert the image to a numpy array
            img_array = np.array(img)

            # create a white blank image of size (100, 100)
            new_image = np.ones((100, 100, 3)) * 255

            # randomly select positions for placing the preprocessed image
            xrand_number = random.randint(0, 50) # x coordinate of the center of the main image
            yrand_number = random.randint(0, 50) # y coordinate of the center of the main image

            # place the preprocessed image on the new blank image at the selected positions
            new_image[xrand_number:xrand_number+50, yrand_number:yrand_number+50, :3] = img_array

            # save the new image with a formatted number as its name
            number = f"{i:02d}"
            file_path = f"{new_images_directory}/{number}.png"
            cv2.imwrite(file_path, new_image)

            i += 1

    # shuffle created images and rename them with sequential numbers
    # get a list of the new images file names 
    file_names = os.listdir(new_images_directory)
    random.shuffle(file_names)
    for i, file_name in enumerate(file_names):
        new_file_name = f"{i+1:04d}.png"
        current_file_path = os.path.join(new_images_directory, file_name)
        new_file_path = os.path.join(new_images_directory, new_file_name)
        os.rename(current_file_path, new_file_path)

create_new_images(raw_images_directory, new_images_directory)