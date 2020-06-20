# -*- coding: utf-8 -*-
# 
# # MIT License
# 
# Copyright (c) 2020 Mike Simms
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""tensorflow image classification script used to determine if someone is at my front door."""

import argparse
import cv2
import io
import math
import os
import signal
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

if sys.version_info[0] < 3:
    import ConfigParser as configparser
else:
    import configparser

x = 640
y = 360
depth = 3
quitting = False
rate = 1000.0 # rate (in ms) at which to sample to the RTSP stream


def signal_handler(signal, frame):
    global quitting

    print("Quitting...")
    quitting = True

def post_to_slack(config, message):
    """Post a message to the the slack channel specified in the configuration file."""
    try:
        key = config.get('Slack', 'key')
        channel = config.get('Slack', 'channel')

        from slacker import Slacker
        slack = Slacker(key)
        slack.chat.post_message(channel, message)
    except configparser.NoOptionError:
        pass
    except configparser.NoSectionError:
        pass
    except ImportError:
        print("Failed ot import Slacker. Cannot post to Slack. Either install the module or remove the Slack section from the configuration file.")
    except:
        pass

def show_training_images(train_label1_dir, train_label2_dir):
    """For debuggging purposes, shows the images used to fit the model."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    train_label1_names = os.listdir(train_label1_dir)
    train_label2_names = os.listdir(train_label2_dir)

    # Parameters for our graph; we'll output images in a 4x4 configuration.
    nrows = 4
    ncols = 4

    # Index for iterating over images.
    pic_index = 0

    # Set up matplotlib fig, and size it to fit 4x4 pics.
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_person_pix = [os.path.join(train_label1_dir, fname) for fname in train_label1_names[pic_index-8:pic_index]]
    next_non_person_pix = [os.path.join(train_label2_dir, fname) for fname in train_label2_names[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_person_pix + next_non_person_pix):

        # Set up subplot; subplot indices start at 1.
        sp = plt.subplot(nrows, ncols, i + 1)

        # Don't show axes (or gridlines).
        sp.axis('Off')

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

def build_model(input_dir, validation_dir, train_label1_dir, train_label2_dir):
    global x, y, depth

    my_callbacks = []
    my_callbacks.append(EarlyStopping(monitor='loss', patience=3))

    train_label1_names = os.listdir(train_label1_dir)
    train_label2_names = os.listdir(train_label2_dir)
    batch_size = 1
    num_samples = len(train_label1_names) + len(train_label2_names)
    steps_per_epoch = math.ceil(num_samples / batch_size)

    model = tf.keras.models.Sequential([
        # First convolution.
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(x,y,depth)),
        tf.keras.layers.MaxPooling2D(2,2),
        # Second convolution.
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Third convolution.
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Fourth convolution.
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Fifth convolution.
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN.
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer.
        tf.keras.layers.Dense(512, activation='relu'),
        # Just one output neuron.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    # All images will be rescaled by 1/255.
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator.
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(input_dir, target_size=(x, y), batch_size=batch_size, class_mode='binary', color_mode="rgb")
    train_generator.shuffle = True

    # Flow validation images in batches of 32.
    if len(validation_dir) > 0:
        print("Loading validation data...")
        validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(x, y), batch_size=batch_size, class_mode='binary', color_mode="rgb")
        validation_generator.shuffle = True
    else:
        validation_generator = None

    # Fit the model.
    print("Fitting the model...")
    num_validation_steps = 8
    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=16, verbose=1, validation_data=validation_generator, validation_steps=num_validation_steps, callbacks=my_callbacks)

    # Evaluate the model.
    print("Evaluating the model...")
    print(model.evaluate(x=validation_generator, steps=num_validation_steps))

    return model

def show_test_image(data):
    """For debuggging purposes, shows the images used to fit the model."""
    import matplotlib.pyplot as plt

    plt.imshow(data)
    plt.show()

def predict_from_img_data(model, config, img):
    """Score raw image data against the model."""
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    images = np.vstack([img_array])
    classes = model.predict(images, batch_size=1)
    score = classes[0]
    if score > 0.5:
        print("Person detected! " + str(score))
        post_to_slack(config, "Person detected! " + str(score))
    else:
        print("Person not detected. " + str(score))

def predict_from_file(model, config, file_name, show_image):
    """Score a file against the model."""
    print("Testing " + file_name + "...")

    global x, y, depth
    global quitting
    global rate

    img = image.load_img(file_name, target_size=(x, y, depth))
    predict_from_img_data(model, config, img)
    if show_image:
        show_test_image(img)

def predict_from_dir(model, config, dir_name, show_images):
    """Score a directory of files against the model."""
    print("Testing " + dir_name + "...")

    items = os.listdir(dir_name)
    for item in items:
        file_name = os.path.join(dir_name, item)
        if os.path.isfile(file_name):
            predict_from_file(model, config, file_name, show_images)

def predict_from_rtsp(model, config, url, show_images):
    """Score samples from an RTSP stream against the model."""
    print("Connecting to RTSP stream " + url + "...")

    global x, y, depth
    global quitting
    global rate

    cap = cv2.VideoCapture(url)
    while cap.isOpened() and not quitting:
        _, original_image = cap.read()
        #resized_image = original_image.reshape(x, y, depth)
        #gray_image = tf.image.rgb_to_grayscale(resized_image)
        predict_from_img_data(model, config, original_image)
        if show_images:
            show_test_image(resized_image)
        time.sleep(rate / 1000.0)
    cap.release()

def load_config(config_file_name):
    """Loads the configuration file."""
    config = configparser.RawConfigParser(allow_no_value=True)
    if sys.version_info[0] < 3:
        with open(config_file_name) as f:
            sample_config = f.read()
        config.readfp(io.BytesIO(sample_config))
    else:
        config.read(config_file_name)
    return config

def main():
    """Entry point for the app."""

    # Parse command line options.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="", help="Directory containing the input files used to train the model.", required=False)
    parser.add_argument("--validation-dir", default="", help="Directory containing the validation files used to validate the model.", required=False)
    parser.add_argument("--predict-file", default="", help="Test the specified file against the model.", required=False)
    parser.add_argument("--predict-dir", default="", help="Test the specified files against the model.", required=False)
    parser.add_argument("--predict-rtsp", default="", help="Test samples from the RTSP stream against the model.", required=False)
    parser.add_argument("--show-images", action="store_true", default=False, help="Show images used for training.", required=False)
    parser.add_argument("--config", type=str, action="store", default="", help="The configuration file.", required=True)

    try:
        args = parser.parse_args()
    except IOError as e:
        parser.error(e)
        sys.exit(1)

    train_label1_dir = os.path.join(args.input_dir, 'person')
    train_label2_dir = os.path.join(args.input_dir, 'not_person')

    # Register the signal handler.
    signal.signal(signal.SIGINT, signal_handler)

    # Parse the config file.
    config = load_config(args.config)

    # Either train or load the model.
    if len(args.input_dir) > 0:

        # For debugging/demonstration purposes.
        if args.show_images:
            show_training_images(train_label1_dir, train_label2_dir)

        # Train the model.
        model = build_model(args.input_dir, args.validation_dir, train_label1_dir, train_label2_dir)

        # Save it so we don't have to do this again.
        model.save(config.get('General', 'Model'))

    # Load the model from file.
    else:
        model = tf.keras.models.load_model(config.get('General', 'Model'))

    # Test the model against real data.
    if len(args.predict_file) > 0:
        predict_from_file(model, config, args.predict_file, args.show_images)
    if len(args.predict_dir) > 0:
        predict_from_dir(model, config, args.predict_dir, args.show_images)
    if len(args.predict_rtsp) > 0:
        predict_from_rtsp(model, config, args.predict_rtsp, args.show_images)

if __name__ == "__main__":
    main()
