import tensorflow as tf 
import os
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np

img_size = 128
channels = 3
input_dim = img_size*img_size*channels

#Helper function to get all image paths from the given directory   
def get_image_paths(img_dir):
    file=[]
    for subdir, dirs, files in os.walk(img_dir):
        for i in files: 
            i = os.path.join(img_dir, i)
            file.append(i)
        return file

#Helper function to convert paht of the image to numpy array representation of the image
def path_to_numpy(img_dir):
    image = load_img(img_dir, target_size=(img_size, img_size))
    image = img_to_array(image)
    return image

#Helper function to generate the net batch of images
def next_batch(dataset, i, batch_size, ctr):
    x = np.zeros((batch_size, img_size, img_size, channels), dtype = np.float32)
    for k in range(batch_size):
        x[k] = path_to_numpy(dataset[i+k+ctr])
        x[k] = np.multiply(x[k], 1.0/255.0)
    return x

#Helper function to generate the next test image
def next_image(dataset, i, batch_size):
    x = np.zeros((1, img_size, img_size, channels), dtype = np.float32)
    x[0] = path_to_numpy(dataset[i+batch_size+1])
    x[0] = np.multiply(x[0], 1.0/255.0)
    return x

