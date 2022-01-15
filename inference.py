# -*- coding: utf-8 -*-

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def predict(model_path, image_path):
    # Load image from disk
    image = load_img(image_path, target_size=(150, 150), interpolation='bilinear')
    image = img_to_array(image, data_format='channels_last')
    image = preprocess_input(image).reshape(1, 150, 150, 3)

    # Extract feature using pre-trained VGG19
    vgg19_base = VGG19(include_top=False, weights='imagenet', input_shape=(150,150,3))
    features = vgg19_base.predict(image).reshape(1, 4*4*512)

    # Perform inference
    model = load_model(model_path)
    result = model.predict(features)
    print(result)
