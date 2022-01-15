# -*- coding: utf-8 -*-

import os
from os.path import isfile, join, sep

import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import VGG19

from hdf5_writer import HDF5Writer


def get_image_paths(data_dir):
    return [full_path for f in os.listdir(data_dir) if isfile(full_path := join(data_dir, f))]


def get_data_paths_and_labels(data_dir):
    data_paths = get_image_paths(join(data_dir, 'man'))
    woman_paths = get_image_paths(join(data_dir, 'woman'))
    data_paths += woman_paths
    labels = [0 if f.split(sep)[-2] == 'man' else 1 for f in data_paths]

    return data_paths, labels


def write_data_to_hdf5(data_dir, output_dir):
    data_paths, labels = get_data_paths_and_labels(data_dir)
    dims = (len(data_paths), 4*4*512)

    index = np.arange(dims[0])
    np.random.shuffle(index)

    vgg19_base = VGG19(include_top=False, weights='imagenet', input_shape=(150,150,3))

    with HDF5Writer(output_dir, 32, dims) as writer:
        for i in index:
            image = load_img(data_paths[i], target_size=(150, 150), interpolation='bilinear')
            image = img_to_array(image, data_format='channels_last')
            image = preprocess_input(image).reshape(1, 150, 150, 3)
            features = vgg19_base.predict(image).reshape(4*4*512)
            writer.write(features, labels[i])
