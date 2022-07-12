import numpy as np
import os
import tensorflow as tf
from imageio import imread, imsave
import skimage.transform
import random
import copy


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


def load_images_with_noise(image_path, image_path_list, num_noise):
    noise_images = np.zeros([1, num_noise, 299, 299, 3], dtype=float)
    root_dir = './data/dev_images'
    full_path = os.path.join(root_dir, image_path)
    with tf.gfile.Open(full_path, 'rb') as f:
        image = imread(f, pilmode='RGB').astype(np.float) / 255.0
    image_shape = image.shape
    if image_shape[0] < 100 or image_shape[1] < 100 or image_shape[2] != 3:
        return False, None, None
    base_image = skimage.transform.resize(image, [299, 299])
    base_image = base_image * 2.0 - 1.0
    base_image = np.expand_dims(base_image, axis=0)
    image_path_list_ = image_path_list.copy()
    image_path_list_.remove(image_path)
    chosen_filepaths = random.sample(image_path_list_, num_noise)
    for i, chosen_filepath in enumerate(chosen_filepaths):
        with tf.gfile.Open(os.path.join(root_dir, chosen_filepath), 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
            image_shape = image.shape
            if image_shape[0] < 100 or image_shape[1] < 100 or image_shape[2] != 3:
                return False, None, None
            image = skimage.transform.resize(image, [299, 299])
            noise_images[0, i, :, :, :] = image * 2.0 - 1.0
    return True, base_image, noise_images


def parse_dataset(file_path):
    if not os.path.exists(file_path):
        file = open(file_path, 'w')
        file.close()
    with open(file_path, 'r') as file:
        lines = file.readlines()
    img_paths = []
    op_list = []
    value_list = []
    for line in lines:
        items = line.split()
        img_path, value = items[0], items[-1]
        ops = [int(item) for item in items[1:-1]]
        img_paths.append(img_path)
        op_list.append(ops)
        value_list.append([float(value)])
    return img_paths, np.array(op_list), np.array(value_list)

def parse_dataset_valrs(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    img_paths = []
    op_list = []
    value_list = []
    valrs_root = './data/val_rs'
    valrs_images = os.listdir(valrs_root)
    for line in lines:
        img_path, op_1, op_2, value = line.split()
        if img_path in valrs_images:
            img_paths.append(img_path)
            op_list.append([int(op_1), int(op_2)])
            value_list.append([float(value)])
    return img_paths, np.array(op_list), np.array(value_list)


def parse_dataset_optim(file_path, num_sample):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    img_paths = []
    op_list = []
    value_list = []
    lines = random.sample(lines, num_sample)
    for line in lines:
        items = line.split()
        img_path, value = items[0], items[-1]
        ops = [int(item) for item in items[1:-1]]
        img_paths.append(img_path)
        op_list.append(ops)
        value_list.append([float(value)])
    return img_paths, np.array(op_list), np.array(value_list)




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


def prepare_dataset(image_path, op_list, value_list, batch_size, root_dir):
    tf.logging.error('Data size : {}'.format(len(image_path)))
    N = len(image_path)
    encoder_input = op_list
    decoder_target = copy.copy(op_list)
    encoder_target = value_list
    encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int32)
    encoder_target = tf.convert_to_tensor(encoder_target, dtype=tf.float32)
    decoder_target = tf.convert_to_tensor(decoder_target, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image_path, encoder_input, encoder_target, decoder_target))
    dataset = dataset.shuffle(buffer_size=N)
    dataset = dataset.map(image_processing_wrap(root_dir))
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_one_shot_iterator()
    image, image_path, encoder_input, encoder_target, decoder_target = iterator.get_next()
    assert encoder_input.shape.ndims == 2
    assert encoder_target.shape.ndims == 2
    assert decoder_target.shape.ndims == 2
    assert image.shape.ndims == 4
    return image, image_path, encoder_input, encoder_target, decoder_target