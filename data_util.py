import numpy as np
import os
import tensorflow as tf
from imageio import imread, imsave
from PIL import Image
import random


def get_two_combination_data_valrs(root_dir, num_sample=100, num_op=20, num_trans=2):
    encoder_target, decoder_target = np.zeros((num_sample, 1), dtype=float), np.zeros((num_sample, num_trans), dtype=int),
    encoder_input = np.random.randint(low=0, high=num_op, size=(num_sample, num_trans), dtype=int)
    gt_table = np.zeros((num_sample, num_op * num_op), dtype=float)
    image_path = tf.gfile.Glob(os.path.join(root_dir, '*'))
    image_path = [os.path.basename(path) for path in image_path]
    return encoder_input, encoder_target, decoder_target, image_path, gt_table

def image_processing_wrap(root_dir):
    def image_processing(image_path, *other):
        image_root = tf.constant(root_dir, dtype=tf.string)
        image = tf.read_file(tf.string_join([image_root, image_path], separator="/"))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (299, 299))
        image.set_shape([299, 299, 3])
        image = tf.cast(image, tf.float32) / 127.5 - 1
        output = image, image_path, *other
        return output
    return image_processing


def collect_variables(scope=None, underired_scope=None):
    var_list = tf.trainable_variables(scope=scope)
    filtered_var_list = []
    for var in var_list:
        if not underired_scope or not var.op.name.startswith(underired_scope):
            filtered_var_list.append(var)
    return filtered_var_list

def modify_variables(var_list, add=None, remove=None):
    def name_in_checkpoint(var):
        if add:
            return add + var.op.name
        else:
            return var.op.name.replace(remove, "")
    var_list = {name_in_checkpoint(var): var for var in var_list}
    return var_list

def load_images_with_noise(image_path_exist, num_noise, input_dir='./data/dev_images'):
    num_batch = len(image_path_exist)
    noise_images = np.zeros([num_batch, num_noise, 299, 299, 3], dtype=float)
    filepaths = tf.gfile.Glob(os.path.join(input_dir, '*'))
    for idx, exist in enumerate(image_path_exist):
        filepaths_ = filepaths.copy()
        full_exist = os.path.join(input_dir, exist)
        filepaths_.remove(full_exist)
        chosen_filepaths = random.sample(filepaths_, num_noise)
        for i, chosen_filepath in enumerate(chosen_filepaths):
            with tf.gfile.Open(chosen_filepath, 'rb') as f:
                image = imread(f, pilmode='RGB').astype(np.float) / 255.0
            noise_images[idx, i, :, :, :] = image * 2.0 - 1.0
    return noise_images
