"""Implementation of MIFGSM attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import tensorflow as tf
from stn import spatial_transformer_network as transformer

import sys
sys.path.append("..")
from util import load_labels, load_images, load_images_with_noise, save_images, check_or_create_dir, progress_bar
from model import id_to_arg_scope, id_to_scope, id_to_saverscope, id_to_model, id_to_checkpoint

slim = tf.contrib.slim

TI_FLAG = False
NI_FLAG = False
VI_FLAG = False
MOMENTUM = 1.0
KERNEL_SIZE = 7
NUMBER = 20
BETA = 1.5

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

kernel = gkern(KERNEL_SIZE, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def build_attack_graph(params):
    eps = 2.0 * params['max_epsilon'] / 255.0
    num_classes = 1001
    batch_shape = [1, params['image_height'], params['image_width'], 3]
    sign_flag = 1.0 if params['untarget'] else -1.0

    model_list = params['attack_model'].split('_')

    def input_transform(input_tensor, noise_tensor, i, op_chosen):
        ops_function = {'Admix': input_admix, 'Scale': input_scale, 'Admix_and_Scale': input_admix_and_scale, 'Brightness': input_brightness, 'Color': input_color, 'Contrast': input_contrast, 'Sharpness': input_sharpness, 'Invert': input_invert, 'Hue': input_hue, 'Saturation': input_saturation, 'Gamma': input_gamma, 'Crop': input_crop, 'Resize': input_resize, 'Rotate': input_rotate, 'ShearX': input_shearX, 'ShearY': input_shearY, 'TranslateX': input_translateX, 'TranslateY': input_translateY, 'Reshape': input_reshape, 'Cutout': input_cutout}

        ops_arg = {'Admix': (noise_tensor, i), 'Scale': (), 'Admix_and_Scale': (noise_tensor, i), 'Brightness': (0.5, ), 'Color': (0.5, ), 'Contrast': (0.5, ), 'Sharpness': (0.5, ), 'Invert': (), 'Hue': (0.2, ), 'Saturation': (0.5, ), 'Gamma': (0.4, ), 'Crop': (), 'Resize': (), 'Rotate': (np.pi/6, ), 'ShearX': (0.5, ), 'ShearY': (0.5, ), 'TranslateX': (0.4, ), 'TranslateY': (0.4, ), 'Reshape': (0.5, ), 'Cutout': (60, -1)}

        ops_list = ['Admix', 'Scale', 'Admix_and_Scale', 'Brightness', 'Color', 'Contrast', 'Sharpness', 'Invert', 'Hue', 'Saturation', 'Gamma', 'Crop', 'Resize', 'Rotate', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Reshape', 'Cutout']

        def f0(input_tensor):
            return input_admix(input_tensor, noise_tensor, i)
        def f1(input_tensor):
            return input_scale(input_tensor)
        def f2(input_tensor):
            return input_admix_and_scale(input_tensor, noise_tensor, i)
        def f3(input_tensor):
            return input_brightness(input_tensor, 0.5)
        def f4(input_tensor):
            return input_color(input_tensor, 0.5)
        def f5(input_tensor):
            return input_contrast(input_tensor, 0.5)
        def f6(input_tensor):
            return input_sharpness(input_tensor, 0.5)
        def f7(input_tensor):
            return input_invert(input_tensor)
        def f8(input_tensor):
            return input_hue(input_tensor, 0.2)
        def f9(input_tensor):
            return input_saturation(input_tensor, 0.5)
        def f10(input_tensor):
            return input_gamma(input_tensor, 0.4)
        def f11(input_tensor):
            return input_crop(input_tensor)
        def f12(input_tensor):
            return input_resize(input_tensor)
        def f13(input_tensor):
            return input_rotate(input_tensor, np.pi/6)
        def f14(input_tensor):
            return input_shearX(input_tensor, 0.5)
        def f15(input_tensor):
            return input_shearY(input_tensor, 0.5)
        def f16(input_tensor):
            return input_translateX(input_tensor, 0.4)
        def f17(input_tensor):
            return input_translateY(input_tensor, 0.4)
        def f18(input_tensor):
            return input_reshape(input_tensor, 0.5)
        def f19(input_tensor):
            return input_cutout(input_tensor, 60, -1)
        def f_(input_tensor):
            return input_tensor
        num_op = op_chosen.shape[2]
        for op in range(num_op):
            op_id = op_chosen[0, i, op]
            input_tensor = tf.switch_case(op_id, branch_fns={
                0: lambda: f0(input_tensor), 
                1: lambda: f1(input_tensor),
                2: lambda: f2(input_tensor),
                3: lambda: f3(input_tensor),
                4: lambda: f4(input_tensor),
                5: lambda: f5(input_tensor),
                6: lambda: f6(input_tensor),
                7: lambda: f7(input_tensor),
                8: lambda: f8(input_tensor),
                9: lambda: f9(input_tensor),
                10: lambda: f10(input_tensor),
                11: lambda: f11(input_tensor),
                12: lambda: f12(input_tensor),
                13: lambda: f13(input_tensor),
                14: lambda: f14(input_tensor),
                15: lambda: f15(input_tensor),
                16: lambda: f16(input_tensor),
                17: lambda: f17(input_tensor),
                18: lambda: f18(input_tensor),
                19: lambda: f19(input_tensor),
                }, default=lambda: f_(input_tensor))

        return input_tensor

    def input_admix_and_scale(input_tensor, noise_tensor, i, prob=1.0):
        with tf.variable_scope('admix_and_scale', reuse=tf.AUTO_REUSE):
            SCALE_FACTOR = tf.cast(tf.random_uniform(shape=[1], minval=0, maxval=5, dtype=tf.int32), dtype=tf.float32)[0]
            return (input_tensor + 0.2 * noise_tensor[:, i]) / 2 ** SCALE_FACTOR

    def input_admix(input_tensor, noise_tensor, i, prob=1.0):
        with tf.variable_scope('admix', reuse=tf.AUTO_REUSE):
            processed_image = input_tensor + 0.2 * noise_tensor[:, i]
            return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda: processed_image, lambda: input_tensor)

    def input_scale(input_tensor, prob=1.0):
        with tf.variable_scope('scale', reuse=tf.AUTO_REUSE):
            SCALE_FACTOR = tf.cast(tf.random_uniform(shape=[1], minval=0, maxval=5, dtype=tf.int32), dtype=tf.float32)[0]
            processed_image = input_tensor / 2 ** SCALE_FACTOR
            return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda: processed_image, lambda: input_tensor)

    def blend(image_1, image_2, factor):
        processed_image = image_1 * (1 - factor) + image_2 * factor
        return tf.clip_by_value(processed_image, -1.0, 1.0)

    def input_brightness(input_tensor, factor_delta=0.5, prob=1.0):
        with tf.variable_scope('brightness', reuse=tf.AUTO_REUSE):
            factor = tf.random_uniform(shape=[1], minval=1-factor_delta, maxval=1+factor_delta)[0]
            degenerate = tf.zeros_like(input_tensor)
            processed_image = blend(degenerate, input_tensor, factor)
            return processed_image

    def input_color(input_tensor, factor_delta=0.5, prob=1.0):
        with tf.variable_scope('color', reuse=tf.AUTO_REUSE):
            factor = tf.random_uniform(shape=[1], minval=1-factor_delta, maxval=1+factor_delta)[0]
            degenerate = 0.2989 * input_tensor[..., 0] + 0.5870 * input_tensor[..., 1] + 0.1140 * input_tensor[..., 2]
            degenerate = tf.tile(tf.expand_dims(degenerate, axis=-1), [1,1,1,3])
            processed_image = blend(degenerate, input_tensor, factor)
            return processed_image

    def input_contrast(input_tensor, factor_delta=0.5, prob=1.0):
        with tf.variable_scope('contrast', reuse=tf.AUTO_REUSE):
            factor = tf.random_uniform(shape=[1], minval=1-factor_delta, maxval=1+factor_delta)[0]
            degenerate = 0.2989 * input_tensor[..., 0] + 0.5870 * input_tensor[..., 1] + 0.1140 * input_tensor[..., 2]
            mean = tf.reduce_mean(degenerate, axis=[1, 2], keep_dims=True)
            mean = tf.expand_dims(mean, axis=-1)
            degenerate = tf.ones_like(input_tensor) * mean
            processed_image = blend(degenerate, input_tensor, factor)
            return processed_image

    sharpness_kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32)
    sharpness_kernel = sharpness_kernel / np.sum(sharpness_kernel)
    sharpness_kernel = np.stack([sharpness_kernel, sharpness_kernel, sharpness_kernel]).swapaxes(2, 0)
    sharpness_kernel = np.expand_dims(sharpness_kernel, 3)
    def input_sharpness(input_tensor, factor_delta=0.5, prob=1.0):
        with tf.variable_scope('sharpness', reuse=tf.AUTO_REUSE):
            factor = tf.random_uniform(shape=[1], minval=1-factor_delta, maxval=1+factor_delta)[0]
            degenerate = tf.nn.depthwise_conv2d(input_tensor, sharpness_kernel, strides=[1, 1, 1, 1], padding='SAME')
            processed_image = blend(degenerate, input_tensor, factor)
            return processed_image

    def input_shearX(input_tensor, delta=0.5, prob=1.0):
        with tf.variable_scope('shearX', reuse=tf.AUTO_REUSE):
            factor = tf.random_uniform(shape=[1], minval=-delta, maxval=delta)[0]
            matrix = [1, factor, 0, 0, 1, 0]
            processed_image = transformer(input_tensor, [matrix] * params['batch_size'])
            return processed_image

    def input_shearY(input_tensor, delta=0.5, prob=1.0):
        with tf.variable_scope('shearY', reuse=tf.AUTO_REUSE):
            factor = tf.random_uniform(shape=[1], minval=-delta, maxval=delta)[0]
            matrix = [1, 0, 0, factor, 1, 0]
            processed_image = transformer(input_tensor, [matrix] * params['batch_size'])
            return processed_image

    def input_translateX(input_tensor, delta=0.4, prob=1.0):
        with tf.variable_scope('translateX', reuse=tf.AUTO_REUSE):
            factor = tf.random_uniform(shape=[1], minval=-delta, maxval=delta)[0]
            matrix = [1, 0, factor, 0, 1, 0]
            processed_image = transformer(input_tensor, [matrix] * params['batch_size'])
            return processed_image

    def input_translateY(input_tensor, delta=0.4, prob=1.0):
        with tf.variable_scope('translateY', reuse=tf.AUTO_REUSE):
            factor = tf.random_uniform(shape=[1], minval=-delta, maxval=delta)[0]
            matrix = [1, 0, 0, 0, 1, factor]
            processed_image = transformer(input_tensor, [matrix] * params['batch_size'])
            return processed_image

    def input_reshape(input_tensor, delta=0.5, prob=1.0):
        with tf.variable_scope('reshape', reuse=tf.AUTO_REUSE):
            scale_x = tf.random_uniform(shape=[1], minval=1-delta, maxval=1+delta)[0]
            scale_y = tf.random_uniform(shape=[1], minval=1-delta, maxval=1+delta)[0]
            shear_x = tf.random_uniform(shape=[1], minval=-delta, maxval=delta)[0]
            shear_y = tf.random_uniform(shape=[1], minval=-delta, maxval=delta)[0]
            translate_x = tf.random_uniform(shape=[1], minval=-delta, maxval=delta)[0]
            translate_y = tf.random_uniform(shape=[1], minval=-delta, maxval=delta)[0]
            matrix = [scale_x, shear_x, translate_x, shear_y, scale_y, translate_y]
            processed_image = transformer(input_tensor, [matrix] * params['batch_size'])
            return processed_image

    def input_rotate(input_tensor, theta=np.pi/6, prob=1.0):
        with tf.variable_scope('rotate', reuse=tf.AUTO_REUSE):
            random_angles = tf.random_uniform(shape=(tf.shape(input_tensor)[0],), minval=-theta, maxval=theta)
            distorted_image = tf.contrib.image.transform(
                input_tensor,
                tf.contrib.image.angles_to_projective_transforms(
                    random_angles, tf.cast(tf.shape(input_tensor)[1], tf.float32), tf.cast(tf.shape(input_tensor)[2], tf.float32)
                ))
            return distorted_image

    def input_crop(input_tensor, prob=1.0):
        with tf.variable_scope('crop', reuse=tf.AUTO_REUSE):
            rnd = tf.random_uniform((), 279, params['image_width'], dtype=tf.int32)
            croped = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            h_rem = params['image_height'] - rnd
            w_rem = params['image_width'] - rnd
            pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
            pad_bottom = h_rem - pad_top
            pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
            pad_right = w_rem - pad_left
            padded = tf.pad(croped, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
            padded.set_shape((input_tensor.shape[0], params['image_height'], params['image_width'], 3))
            return padded

    def input_resize(input_tensor, prob=1.0):
        with tf.variable_scope('resize', reuse=tf.AUTO_REUSE):
            rnd = tf.random_uniform((), params['image_width'], params['image_resize'], dtype=tf.int32)
            rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            h_rem = params['image_resize'] - rnd
            w_rem = params['image_resize'] - rnd
            pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
            pad_bottom = h_rem - pad_top
            pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
            pad_right = w_rem - pad_left
            padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
            padded = tf.image.resize_images(padded, [params['image_height'], params['image_width']], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            padded.set_shape((input_tensor.shape[0], params['image_height'], params['image_width'], 3))
            return padded

    def input_cutout(input_tensor, pad_size=60, replace=-1, prob=1.0):
        with tf.variable_scope('cutout', reuse=tf.AUTO_REUSE):
            image_height = tf.shape(input_tensor)[1]
            image_width = tf.shape(input_tensor)[2]
            mask_list = []
            for i in range(params['batch_size']):
                cutout_center_height = tf.random_uniform(
                    shape=[], minval=0, maxval=image_height,
                    dtype=tf.int32)
                cutout_center_width = tf.random_uniform(
                    shape=[], minval=0, maxval=image_width,
                    dtype=tf.int32)

                lower_pad = tf.maximum(0, cutout_center_height - pad_size)
                upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
                left_pad = tf.maximum(0, cutout_center_width - pad_size)
                right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

                cutout_shape = [image_height - (lower_pad + upper_pad),
                                image_width - (left_pad + right_pad)]
                padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
                mask = tf.pad(
                    tf.zeros(cutout_shape, dtype=input_tensor.dtype),
                    padding_dims, constant_values=1)
                mask = tf.expand_dims(mask, -1)
                mask = tf.tile(mask, [1, 1, 3])
                mask_list.append(mask)
            mask = tf.stack(mask_list, axis=0)
            processed_image = input_tensor * mask
            return processed_image

    def input_invert(input_tensor, prob=1.0):
        with tf.variable_scope('invert', reuse=tf.AUTO_REUSE):
            processed_image = -input_tensor
            return processed_image

    def hsv_to_rgb(tensor):
        h = tensor[..., 0]
        s = tensor[..., 1]
        v = tensor[..., 2]
        c = s * v
        m = v - c
        dh = h * 6
        h_category = tf.cast(dh, tf.int32)
        fmodu = dh - tf.floor(dh / 2) * 2
        x = c * (1 - tf.abs(fmodu - 1))
        component_shape = tf.shape(tensor)[:-1]
        dtype = tensor.dtype
        rr = tf.zeros(component_shape, dtype=dtype)
        gg = tf.zeros(component_shape, dtype=dtype)
        bb = tf.zeros(component_shape, dtype=dtype)
        h0 = tf.equal(h_category, 0)
        rr = tf.where(h0, c, rr)
        gg = tf.where(h0, x, gg)
        h1 = tf.equal(h_category, 1)
        rr = tf.where(h1, x, rr)
        gg = tf.where(h1, c, gg)
        h2 = tf.equal(h_category, 2)
        gg = tf.where(h2, c, gg)
        bb = tf.where(h2, x, bb)
        h3 = tf.equal(h_category, 3)
        gg = tf.where(h3, x, gg)
        bb = tf.where(h3, c, bb)
        h4 = tf.equal(h_category, 4)
        rr = tf.where(h4, x, rr)
        bb = tf.where(h4, c, bb)
        h5 = tf.equal(h_category, 5)
        rr = tf.where(h5, c, rr)
        bb = tf.where(h5, x, bb)
        r = rr + m
        g = gg + m
        b = bb + m
        return tf.stack([r, g, b], axis=-1)

    def rgb_to_hsv(tensor):
        r = tensor[..., 0]
        g = tensor[..., 1]
        b = tensor[..., 2]
        x_max = tf.reduce_max(tensor, axis=-1, keep_dims=False)
        x_min = tf.reduce_min(tensor, axis=-1, keep_dims=False)
        v = x_max + 1e-10
        c = x_max - x_min + 1e-10
        l = (x_max + x_min) / 2
        component_shape = tf.shape(tensor)[:-1]
        dtype = tensor.dtype
        h = tf.zeros(component_shape, dtype=dtype)
        h1 = tf.equal(v, r)
        h11 = tf.greater(g, b)
        h = tf.where(tf.logical_and(h1, h11), (g-b) / c / 6 , h)
        h = tf.where(tf.logical_and(h1, tf.logical_not(h11)), 1 + (g-b) / c / 6 , h)
        h2 = tf.equal(v, g)
        h = tf.where(h2, (2 + (b-r) / c) / 6, h)
        h3 = tf.equal(v, b)
        h = tf.where(h3, (4 + (r-g) / c) / 6, h)
        s = tf.zeros(component_shape, dtype=dtype)
        s1 = tf.equal(v, 0)
        s = tf.where(s1, s, c/v)
        return tf.stack([h, s, v], axis=-1)

    def input_hue(input_tensor, delta=0.2, prob=1.0):
        with tf.variable_scope('hue', reuse=tf.AUTO_REUSE):
            random_delta = tf.random_uniform(shape=(tf.shape(input_tensor)[0], 1, 1), minval=-delta, maxval=delta)
            processed_image = (input_tensor + 1.0) / 2
            processed_image = rgb_to_hsv(processed_image)
            mask_shape = tf.shape(input_tensor)[:-1]
            dtype = input_tensor.dtype
            mask = tf.ones(mask_shape, dtype=dtype) * random_delta
            mask = tf.stack([mask, tf.zeros(mask_shape, dtype=dtype), tf.zeros(mask_shape, dtype=dtype)], axis=-1)
            processed_image = processed_image + mask
            processed_image = tf.clip_by_value(processed_image, 0.0, 1.0)
            processed_image = hsv_to_rgb(processed_image)
            processed_image = processed_image * 2.0 - 1.0
            return processed_image

    def input_saturation(input_tensor, delta=0.5, prob=1.0):
        with tf.variable_scope('saturation', reuse=tf.AUTO_REUSE):
            random_delta = tf.random_uniform(shape=(tf.shape(input_tensor)[0], 1, 1), minval=1-delta, maxval=1+delta)
            processed_image = (input_tensor + 1.0) / 2
            processed_image = rgb_to_hsv(processed_image)
            mask_shape = tf.shape(input_tensor)[:-1]
            dtype = input_tensor.dtype
            mask = tf.ones(mask_shape, dtype=dtype) * random_delta
            mask = tf.stack([tf.ones(mask_shape, dtype=dtype), mask, tf.ones(mask_shape, dtype=dtype)], axis=-1)
            processed_image = processed_image * mask
            processed_image = tf.clip_by_value(processed_image, 0.0, 1.0)
            processed_image = hsv_to_rgb(processed_image)
            processed_image = processed_image * 2.0 - 1.0
            return processed_image

    def input_gamma(input_tensor, delta=0.4, prob=1.0):
        with tf.variable_scope('gamma', reuse=tf.AUTO_REUSE):
            random_delta = tf.random_uniform(shape=(tf.shape(input_tensor)[0], 1, 1, 1), minval=1-delta, maxval=1+delta)
            processed_image = (input_tensor + 1.0) / 2 + 1e-10
            processed_image = tf.pow(processed_image, random_delta)
            processed_image = tf.clip_by_value(processed_image, 0.0, 1.0)
            processed_image = processed_image * 2.0 - 1.0
            return processed_image

    def graph(x, noise_x, y, i, x_max, x_min, grad, variance, op_chosen):
        eps = 2.0 * params['max_epsilon'] / 255.0
        num_iter = params['num_iter']
        alpha = eps / num_iter
        momentum = MOMENTUM
        num_classes = 1001

        if NI_FLAG:
            x_in = x + momentum * alpha * grad
        else:
            x_in = x

        noise_list = []
        for model_id in model_list:
            for ix in range(params['num_noise']):
                with slim.arg_scope(id_to_arg_scope[model_id]):
                    logits, end_points = id_to_model[model_id](
                        input_transform(x_in, noise_x, ix, op_chosen), num_classes=num_classes, is_training=False, scope=id_to_scope[model_id], reuse=tf.AUTO_REUSE)
                    cross_entropy = tf.losses.softmax_cross_entropy(y, logits, label_smoothing=0.0, weights=1.0)
                    if model_id in ['1', '2', '3', '4', '5', '6', '7']:
                        auxlogits = end_points['AuxLogits']
                        cross_entropy += tf.losses.softmax_cross_entropy(y, auxlogits, label_smoothing=0.0, weights=0.4)
                    noise = tf.gradients(cross_entropy, x)[0]
                    noise_list.append(noise)
        noise = tf.reduce_mean(tf.stack(noise_list, axis=4), axis=4)
        
        if VI_FLAG:
            iter = tf.constant(0)
            max_iter = tf.constant(NUMBER)
            _, _, _, _, _, global_grad = tf.while_loop(grad_finish, batch_grad, [x, y, iter, max_iter, eps*BETA, tf.zeros_like(noise)])
            new_grad = tf.identity(noise)
            noise = noise + variance
            variance = global_grad / (1. * NUMBER) / len(model_list)  - new_grad

        if TI_FLAG:
            noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        noise = momentum * grad + noise
        x = x + alpha * tf.sign(noise) * sign_flag
        x = tf.clip_by_value(x, x_min, x_max)
        i = tf.add(i, 1)
        return x, noise_x, y, i, x_max, x_min, noise, variance, op_chosen


    def stop(x, noise_x, y, i, x_max, x_min, grad, variance, op_prob):
        num_iter = params['num_iter']
        return tf.less(i, num_iter)


    
    def grad_finish(x, y, i, max_iter, alpha, grad):
        return tf.less(i, max_iter)

    def batch_grad(x, y, i, max_iter, alpha, grad):
        x_neighbor = x + tf.random.uniform(x.shape, minval=-alpha, maxval=alpha)

        for model_id in model_list:
            with slim.arg_scope(id_to_arg_scope[model_id]):
                logits, end_points = id_to_model[model_id](
                    x_neighbor, num_classes=1001, is_training=False, scope=id_to_scope[model_id], reuse=tf.AUTO_REUSE)
                cross_entropy = tf.losses.softmax_cross_entropy(y, logits, label_smoothing=0.0, weights=1.0)
                if model_id in ['1', '2', '3', '4', '5', '6', '7']:
                    auxlogits = end_points['AuxLogits']
                    cross_entropy += tf.losses.softmax_cross_entropy(y, auxlogits, label_smoothing=0.0, weights=0.4)
                grad += tf.gradients(cross_entropy, x_neighbor)[0]
        i = tf.add(i, 1)
        return x, y, i, max_iter, alpha, grad

    tf.logging.error('Building attack graph')
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:1'):
            x_input = tf.placeholder(tf.float32, shape=batch_shape)
            noise_x_input = tf.placeholder(tf.float32, shape=[1, params['num_noise'], params['image_height'], params['image_width'], 3])
            x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
            x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

            x_labels = tf.placeholder(tf.int32, shape=[1])
            op_chosen = tf.placeholder(tf.int32, shape=[1, params['num_noise'], params['encoder_length']])
            y = tf.one_hot(x_labels, num_classes)

            i = tf.constant(0)
            grad = tf.zeros(shape=batch_shape)
            variance = tf.zeros(shape=batch_shape)
            x_adv, _, _, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, noise_x_input, y, i, x_max, x_min, grad, variance, op_chosen])

            return g, x_input, noise_x_input, x_labels, x_adv, op_chosen

