import numpy as np
from PIL import Image
import glob

def Mean(img_path):

    # Initiation
    all_image_path = glob.glob(img_path)
    num_images = len(all_image_path)  # number of images present in the path.

    sum = 0  # sum of all array elements from every image

    for image in all_image_path:
        image_opened = Image.open(image)
        total_pixels = image_opened.size[0] * image_opened.size[1]
        # get the total number of pixels
        img_as_array = np.asarray(image_opened)
        # Convert opened image into array

        img_as_array = img_as_array / 255
        img_as_array = img_as_array.sum()
        img_as_array = img_as_array / total_pixels
        sum += img_as_array
        print(sum)

    mean = sum / num_images

    return mean

def StandardDeviation(img_path):

    # Initiation
    all_image_path = glob.glob(img_path)
    num_images = len(all_image_path)  # number of images present in the path.
    mean_value = Mean(img_path)

    square_sum = 0  # sum of (ith element - mean)^2

    for image in all_image_path:

        image_opened = Image.open(image)
        total_pixels = image_opened.size[0] * image_opened.size[1]
        # total number of pixels by multiplying width and depth of the image
        img_as_array = np.asarray(image_opened)

        img_as_array = img_as_array / 255
        img_as_array = img_as_array.sum()
        img_as_array = img_as_array / total_pixels

        ith_square = (img_as_array - mean_value) ** 2
        square_sum += ith_square

    stdev = square_sum / num_images
    stdev = (stdev) ** 0.5

    return stdev


if __name__ == '__main__':

    path_test = '../data/test/images/*.png'
    print(Mean(path_test))
    print(StandardDeviation(path_test))
