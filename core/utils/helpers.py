import cv2
import numpy as np
import itertools
import operator
import os, csv
import argparse
import tensorflow as tf

import time, datetime
import random
import skimage

from skimage import morphology, measure, transform, exposure


_RGB_MEAN = [123.68, 116.78, 103.94]


def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name


def search_largest_region(image):
    labeling = measure.label(image)
    regions = measure.regionprops(labeling)

    largest_region = None
    area_max = 0.
    for region in regions:
        if region.area > area_max:
            area_max = region.area
            largest_region = region

    return largest_region


def generate_largest_region(image):
    region = search_largest_region(image)
    bin_image = np.zeros_like(image)
    if region != None:
        for coord in region.coords:
            bin_image[coord[0], coord[1]] = 1

    return bin_image


def generate_largest_region_threshold(image, ratio=0.02):
    binary_image = np.zeros_like(image)
    labeling = measure.label(image)
    regions = measure.regionprops(labeling)

    for region in regions:
        if region.area > np.prod(image)*ratio:
            for coord in region.coords:
                binary_image[coord[0], coord[1], coord[2]] = 1

    return binary_image


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def load_image(path):
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_GRAY2BGR)
    return image


def load_image_batch(paths, gray=False):
    imgs = []
    for path in paths:
        if gray:
            imgs.append(np.expand_dims(load_image_gray(path), axis=2))
        else:
            img = np.expand_dims(load_image_gray(path), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            imgs.append(img)
            #imgs.append(load_image(path))
    assert np.max(np.array(imgs)) < 256 and np.min(np.array(imgs)) >= 0
    return np.asarray(imgs)


def filter_image(image, img_filter):
    assert len(image.shape) == 2
    return img_filter(image)


def load_image_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def resize_batch_image(batch_images, resize_factor):
    resized_batch_images = []
    for i in range(batch_images.shape[0]):
        image = batch_images[i]
        new_size = [image.shape[0]/resize_factor, image.shape[1]/resize_factor]
        resized_image = transform.resize(image, new_size)
        resized_batch_images.append(resized_image)

    return np.array(resized_batch_images)


def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_width)
        y = random.randint(0, image.shape[0] - crop_height)

        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')


def random_crop3d(image, label, crop_x, crop_y, crop_z):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]) or (image.shape[2]!=label.shape[2]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_x <= image.shape[0]) and (crop_y <= image.shape[1]) and (crop_z <= image.shape[2]):
        x = random.randint(0, image.shape[0] - crop_x)
        y = random.randint(0, image.shape[1] - crop_y)
        z = random.randint(0, image.shape[2] - crop_z)

        return image[x:x + crop_x, y:y + crop_y, z:z+crop_z], label[x:x + crop_x, y:y + crop_y, z:z+crop_z]
    else:
        raise Exception('Crop shape exceeds image dimensions!')


def data_augmentation(input_image, output_image, h_flip, v_flip, rotation, height, width, wb_contrast):
    # Data augmentation
    input_image, output_image = random_crop(input_image, output_image, height, width)

    if h_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if v_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)

    assert np.max(input_image) < 256 and np.min(input_image) >= 0
    if rotation:
        angle = random.uniform(-1*rotation, rotation)
    if rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    assert np.max(input_image) < 256 and np.min(input_image) >= 0
    if wb_contrast and random.randint(0, 1):
        black_hat = morphology.black_tophat(input_image[:, :, 1], selem=morphology.square(5))
        white_hat = morphology.white_tophat(input_image[:, :, 1], selem=morphology.square(5))
        input_image = np.add(np.add(black_hat, white_hat), input_image[:, :, 1])
        input_image = np.clip(input_image, 0, 255)
        input_images = []
        for i in range(3):
            input_images.append(input_image)
        input_image = np.transpose(np.array(input_images), [1, 2, 0])
    if len(output_image.shape) != 3:
        output_image = np.expand_dims(output_image, axis=2)

    assert np.max(input_image) < 256 and np.min(input_image) >= 0
    assert np.max(output_image) < 256 and np.min(output_image) >= 0
    return input_image, output_image


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# class_dict = get_class_dict("CamVid/class_dict.csv")
# y = cv2.imread("CamVid/test_labels/0001TP_007170_L.png",-1)
# y = reverse_one_hot(one_hot_it(y, class_dict))
# y = colour_code_segmentation(y, class_dict)

# file_name = "gt_test.png"
# cv2.imwrite(file_name,np.uint8(y))

def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This seg_model has %d trainable parameters"% (total_parameters))
