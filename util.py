import pandas as pd
import numpy as np
import tensorflow as tf
from imageio import imread, imsave
from PIL import Image
import os
import time
import sys
import random

def load_labels(untarget, dataset='imagenet'):
    if dataset == 'imagenet':
        file_name = './data/dev_dataset.csv'
        dev = pd.read_csv(file_name)
        label = 'TrueLabel' if untarget else 'TargetClass'
        f2l = {dev.iloc[i]['ImageId']+'.png': dev.iloc[i][label] for i in range(len(dev))}
    elif dataset == 'imagenet_valrs':
        file_name = './data/val_rs.csv'
        dev = pd.read_csv(file_name)
        assert untarget is True
        label = 'label'
        f2l = {dev.iloc[i]['filename']: dev.iloc[i][label] for i in range(len(dev))}
    return f2l


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def load_images_with_noise(input_dir, batch_shape, num_noise):
    images = np.zeros(batch_shape)
    noise_images = np.expand_dims(images, axis=1).repeat(num_noise, axis=1)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    filepaths = tf.gfile.Glob(os.path.join(input_dir, '*'))
    for filepath in filepaths:
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filepaths_ = filepaths.copy()
        filepaths_.remove(filepath)
        chosen_filepaths = random.sample(filepaths_, num_noise)
        for i, chosen_filepath in enumerate(chosen_filepaths):
            with tf.gfile.Open(chosen_filepath, 'rb') as f:
                image = imread(f, pilmode='RGB').astype(np.float) / 255.0
            noise_images[idx, i, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images, noise_images
            filenames = []
            images = np.zeros(batch_shape)
            noise_images = np.expand_dims(images, axis=1).repeat(num_noise, axis=1)
            idx = 0
    if idx > 0:
        yield filenames, images, noise_images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, ((images[i, :, :, :] + 1.0) * 0.5 * 255).astype(np.uint8), format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f