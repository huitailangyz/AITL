from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
import json
import math
import AITL

from data_util import load_images_with_noise, get_two_combination_data_valrs
from transform_attack import build_attack_graph_single_step

import sys
from util import load_labels, save_images, check_or_create_dir, progress_bar
from model import id_to_arg_scope, id_to_scope, id_to_saverscope, id_to_model, id_to_checkpoint
slim = tf.contrib.slim

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='models')

parser.add_argument('--AITL_num_seed_arch', type=int, default=1000)

parser.add_argument('--AITL_encoder_num_layers', type=int, default=1)

parser.add_argument('--AITL_encoder_hidden_size', type=int, default=96)

parser.add_argument('--AITL_encoder_emb_size', type=int, default=32)

parser.add_argument('--AITL_mlp_num_layers', type=int, default=0)

parser.add_argument('--AITL_mlp_hidden_size', type=int, default=200)

parser.add_argument('--AITL_decoder_num_layers', type=int, default=1)

parser.add_argument('--AITL_decoder_hidden_size', type=int, default=96)

parser.add_argument('--AITL_encoder_length', type=int, default=2)

parser.add_argument('--AITL_decoder_length', type=int, default=2)

parser.add_argument('--AITL_encoder_dropout', type=float, default=0.1)

parser.add_argument('--AITL_mlp_dropout', type=float, default=0.1)

parser.add_argument('--AITL_decoder_dropout', type=float, default=0.0)

parser.add_argument('--AITL_weight_decay', type=float, default=1e-4)

parser.add_argument('--AITL_encoder_vocab_size', type=int, default=12)

parser.add_argument('--AITL_decoder_vocab_size', type=int, default=12)

parser.add_argument('--AITL_batch_size', type=int, default=1)

parser.add_argument('--AITL_predict_lambda', type=float, default=1)

parser.add_argument('--AITL_predict_num_steps', type=int, default=1)

parser.add_argument('--AITL_predict_num_seeds', type=int, default=1)

parser.add_argument('--AITL_image_hidden_size', type=int, default=128)

parser.add_argument('--AITL_num_sample', type=int, default=100)

parser.add_argument('--AITL_model_id', type=str, default='')

parser.add_argument('--attack_max_epsilon', type=float, default=16)

parser.add_argument('--attack_image_height', type=int, default=299)

parser.add_argument('--attack_image_width', type=int, default=299)

parser.add_argument('--attack_image_resize', type=int, default=330)

parser.add_argument('--attack_model_list', type=str, default="1")

parser.add_argument('--attack_num_iter', type=int, default=10)

parser.add_argument('--attack_num_noise', type=int, default=15)

parser.add_argument('--attack_untarget', type=bool, default=True)

parser.add_argument('--attack_output_dir', type=str, default='')


def predict():
    params = get_AITL_params()
    data_root = "./data/val_rs"
    dataset = "imagenet_valrs"
    encoder_input, encoder_target, decoder_target, image_path, gt_table = get_two_combination_data_valrs(data_root, num_sample=params['num_sample'], num_op=20, num_trans=params['encoder_length'])
    params['batches_per_epoch'] = math.ceil(len(encoder_input) / params['batch_size'])

    with tf.Graph().as_default() as g:
        tf.logging.info('Generating new transform using gradient descent with step size {} and step num {}'.format(params['predict_lambda'], params['predict_num_steps']))
        tf.logging.info('Preparing data')
        N = len(encoder_input)
        image_pred, image_path_pred, encoder_pred_input, encoder_pred_target, decoder_pred_target = AITL.input_fn(
            image_path,
            encoder_input,
            encoder_target,
            decoder_target,
            params['batch_size'],
            1,
            data_root,
        )
        model_id = params['model_id']
        with slim.arg_scope(id_to_arg_scope[model_id]):
            logits, end_points = id_to_model[model_id](
                image_pred, num_classes=1001, is_training=False, scope=id_to_scope[model_id], reuse=tf.AUTO_REUSE)
            image = end_points['PreLogits']
            image_emb_pred = tf.squeeze(image, [1, 2], name='SpatialSqueeze')
        predict_value, sample_id, debug_dict = AITL.get_predict_ops(image_emb_pred, encoder_pred_input, params)
        data_ops = [image_pred, image_emb_pred, image_path_pred, encoder_pred_target, decoder_pred_target]

        attack_params = get_attack_params()
        os.makedirs(attack_params['output_dir'], exist_ok=True)
        x_input, noise_x_input, x_labels, op_chosen, grad, variance, x_adv, grad_adv, variance_adv = build_attack_graph_single_step(g, attack_params)

        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        f2l = load_labels(attack_params['untarget'], dataset)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            s = tf.train.Saver()
            s.restore(sess, tf.train.latest_checkpoint(params['model_dir']))
            for batch in range(params['batches_per_epoch']):
                tf.logging.info("Batch %d" % batch)
                image_v, image_emb_v, image_path_v, encoder_target_v, decoder_target_v = sess.run(data_ops)
                op_chosen_v = np.zeros((len(image_path_v), params['predict_num_seeds'], params['encoder_length']), dtype=int)
                for seed_id in range(params['predict_num_seeds']):
                    encoder_input_v = np.random.randint(low=0, high=params['encoder_vocab_size'], size=(len(image_path_v), params['encoder_length']), dtype=int)
                    arch_emb_v = sess.run(debug_dict['arch_emb'], feed_dict={
                            image_emb_pred: image_emb_v,
                            encoder_pred_input: encoder_input_v,
                        })
                    for step in range(params['predict_num_steps']):
                        predict_value_v,  new_arch_emb_v = sess.run([predict_value, debug_dict['new_arch_emb']], feed_dict={
                            image_emb_pred: image_emb_v,
                            encoder_pred_input: encoder_input_v,
                            debug_dict['arch_emb']: arch_emb_v,
                        })
                        new_sample_id_v, new_predict_value_v = sess.run([sample_id, predict_value], feed_dict={
                            image_emb_pred: image_emb_v,
                            encoder_pred_input: encoder_input_v,
                            debug_dict['arch_emb']: new_arch_emb_v,
                        })
                        arch_emb_v = new_arch_emb_v


                    for i in range(len(image_path_v)):
                        image_path = image_path_v[i]
                        op_chosen_v[i, seed_id] = new_sample_id_v[i]

                image_path_v = [bytes.decode(filename) for filename in image_path_v]
                noise_image = load_images_with_noise(image_path_v, attack_params['num_noise'], input_dir=data_root)
                labels = [f2l[filename] for filename in image_path_v]

                image_in = image_v
                grad_in = np.zeros([attack_params['batch_size'], attack_params['image_height'], attack_params['image_width'], 3])
                variance_in = np.zeros([attack_params['batch_size'], attack_params['image_height'], attack_params['image_width'], 3])

                for step in range(attack_params['num_iter']):
                    image_out, grad_out, variance_out = sess.run([x_adv, grad_adv, variance_adv], feed_dict={
                        x_input: image_in,
                        noise_x_input: noise_image,
                        x_labels: labels,
                        op_chosen: op_chosen_v,
                        grad: grad_in,
                        variance: variance_in})
                    image_in, grad_in, variance_in = image_out, grad_out, variance_out
                adv_images = image_out

                save_images(adv_images, image_path_v, attack_params['output_dir'])


def get_AITL_params():
    params = {
        'model_dir': os.path.join(FLAGS.output_dir, 'AITL'),
        'encoder_num_layers': FLAGS.AITL_encoder_num_layers,
        'encoder_hidden_size': FLAGS.AITL_encoder_hidden_size,
        'encoder_emb_size': FLAGS.AITL_encoder_emb_size,
        'mlp_num_layers': FLAGS.AITL_mlp_num_layers,
        'mlp_hidden_size': FLAGS.AITL_mlp_hidden_size,
        'decoder_num_layers': FLAGS.AITL_decoder_num_layers,
        'decoder_hidden_size': FLAGS.AITL_decoder_hidden_size,
        'encoder_length': FLAGS.AITL_encoder_length,
        'decoder_length': FLAGS.AITL_decoder_length,
        'encoder_dropout': FLAGS.AITL_encoder_dropout,
        'mlp_dropout': FLAGS.AITL_mlp_dropout,
        'decoder_dropout': FLAGS.AITL_decoder_dropout,
        'weight_decay': FLAGS.AITL_weight_decay,
        'encoder_vocab_size': FLAGS.AITL_encoder_vocab_size,
        'decoder_vocab_size': FLAGS.AITL_decoder_vocab_size,
        'batch_size': FLAGS.AITL_batch_size,
        'predict_lambda': FLAGS.AITL_predict_lambda,
        'predict_num_steps': FLAGS.AITL_predict_num_steps,
        'predict_num_seeds': FLAGS.AITL_predict_num_seeds,
        'image_hidden_size': FLAGS.AITL_image_hidden_size,
        'num_sample': FLAGS.AITL_num_sample,
        'model_id': FLAGS.AITL_model_id,
    }
    return params

def get_attack_params():
    params = {
        'model_dir': os.path.join(FLAGS.output_dir, 'AITL'),
        'encoder_length': FLAGS.AITL_encoder_length,
        'batch_size': FLAGS.AITL_batch_size,
        'max_epsilon': FLAGS.attack_max_epsilon,
        'image_height': FLAGS.attack_image_height,
        'image_width': FLAGS.attack_image_width,
        'image_resize': FLAGS.attack_image_resize,
        'model_list': FLAGS.attack_model_list,
        'num_iter': FLAGS.attack_num_iter,
        'num_noise': FLAGS.attack_num_noise,
        'untarget': FLAGS.attack_untarget,
        'output_dir': FLAGS.attack_output_dir,
    }
    return params


def main(_):
    predict()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
