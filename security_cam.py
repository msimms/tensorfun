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

import argparse
import cv2
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x = 640
y = 360
depth = 3

class my_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if ('accuracy' in logs and logs.get('accuracy') > 0.6):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

def show_training_images(train_label1_dir, train_label2_dir):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    train_person_names = os.listdir(train_label1_dir)
    train_not_person_names = os.listdir(train_label2_dir)

    # Parameters for our graph; we'll output images in a 4x4 configuration.
    nrows = 4
    ncols = 4

    # Index for iterating over images.
    pic_index = 0

    # Set up matplotlib fig, and size it to fit 4x4 pics.
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_person_pix = [os.path.join(train_label1_dir, fname) for fname in train_person_names[pic_index-8:pic_index]]
    next_non_person_pix = [os.path.join(train_label2_dir, fname) for fname in train_not_person_names[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_person_pix + next_non_person_pix):
        # Set up subplot; subplot indices start at 1.
        sp = plt.subplot(nrows, ncols, i + 1)

        # Don't show axes (or gridlines).
        sp.axis('Off')

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

def build_model(input_dir, validation_dir, train_label1_dir, train_label2_dir):
    callbacks = my_callback()

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
    print("Training data...")
    train_generator = train_datagen.flow_from_directory(input_dir, target_size=(x, y), batch_size=128, class_mode='binary')

    # Flow validation images in batches of 32.
    if len(validation_dir) > 0:
        print("Validation data...")
        validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(x, y), batch_size=32, class_mode='binary')
    else:
        validation_generator = None

    # Fit the model.
    print("Fitting model...")
    history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1, validation_data=validation_generator, validation_steps=8)

    return model

def predict_from_file(model, file_name):
    print("Testing " + file_name + "...")

    img = image.load_img(file_name, target_size=(x, y))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    images = np.vstack([img_array])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(file_name + " is a person.")
    else:
        print(file_name + " is not a person.")

def predict_from_rtsp(mode, url):
    print("Connecting to RTSP stream " + url)
    cap = cv2.VideoCapture(url)
    while cap.isOpened():
        ret, frame = cap.read()
    cap.release()

def main():
    """Entry point for the app."""

    # Parse command line options.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="", help="Directory containing the input files used to train the model.", required=False)
    parser.add_argument("--validation-dir", default="", help="Directory containing the validation files used to validate the model.", required=False)
    parser.add_argument("--model", default="", help="File name for either saving or loading the model.", required=False)
    parser.add_argument("--predict-file", default="", help="Test the specified file against the model.", required=False)
    parser.add_argument("--predict-rtsp", default="", help="Test samples from the RTSP stream against the model.", required=False)
    parser.add_argument("--show-images", action="store_true", default=False, help="Show images used for training.", required=False)

    try:
        args = parser.parse_args()
    except IOError as e:
        parser.error(e)
        sys.exit(1)

    train_label1_dir = os.path.join(args.input_dir, 'person')
    train_label2_dir = os.path.join(args.input_dir, 'not_person')

    # For debugging/demonstration purposes.
    if args.show_images:
        show_training_images(train_label1_dir, train_label2_dir)

    # Either train or load the model.
    if len(args.input_dir) > 0:

        # Train the model.
        model = build_model(args.input_dir, args.validation_dir, train_label1_dir, train_label2_dir)

        # Save it so we don't have to do this again.
        if len(args.model) > 0:
            model.save(args.model)
    
    # Load the model from file.
    else:
        model = tf.keras.models.load_model(args.model)

    # Test the model against real data.
    if len(args.predict_file) > 0:
        predict_from_file(model, args.predict_file)
    if len(args.predict_rtsp) > 0:
        predict_from_file(model, args.predict_rtsp)

if __name__ == "__main__":
    main()
