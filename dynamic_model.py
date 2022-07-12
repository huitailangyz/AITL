from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import time

import encoder
import decoder
from data_util import collect_variables, modify_variables

import sys
sys.path.append("..")
from model import id_to_arg_scope, id_to_scope, id_to_saverscope, id_to_model, id_to_checkpoint
slim = tf.contrib.slim



def get_train_ops(image_emb_train, encoder_train_input, encoder_train_target, decoder_train_target, params,
                  reuse=tf.AUTO_REUSE):
    with tf.variable_scope('EPD', reuse=reuse):
        my_encoder = encoder.Model(image_emb_train, encoder_train_input, encoder_train_target, params, tf.estimator.ModeKeys.TRAIN,
                                    'Encoder', reuse)
        encoder_outputs = my_encoder.arch_emb
        encoder_outputs.set_shape([None, params['encoder_hidden_size']])
        my_decoder = decoder.Model(encoder_outputs, decoder_train_target, params, tf.estimator.ModeKeys.TRAIN, 'Decoder', reuse)
        encoder_loss = my_encoder.loss
        decoder_loss = my_decoder.loss
        decoder_correct_rate = my_decoder.correct_rate
        mse = encoder_loss
        cross_entropy = decoder_loss

        total_loss = params['trade_off'] * encoder_loss + (1 - params['trade_off']) * decoder_loss + params['weight_decay'] * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    
        tf.summary.scalar('training_loss', total_loss)

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(params['lr'])
        if params['optimizer'] == "sgd":
            learning_rate = tf.cond(
                global_step < params['start_decay_step'],
                lambda: learning_rate,
                lambda: tf.train.exponential_decay(
                    learning_rate,
                    (global_step - params['start_decay_step']),
                    params['decay_steps'],
                    params['decay_factor'],
                    staircase=True),
                name="calc_learning_rate")
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif params['optimizer'] == "adam":
            assert float(params['lr']) <= 0.001, "! High Adam learning rate %g" % params['lr']
            opt = tf.train.AdamOptimizer(learning_rate)
        elif params['optimizer'] == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        tf.summary.scalar("learning_rate", learning_rate)
  
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            var_list = collect_variables(underired_scope=id_to_saverscope[params["model_id"]])
            gradients, variables = zip(*opt.compute_gradients(total_loss, var_list))
            grad_norm = tf.global_norm(gradients)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])
            train_op = opt.apply_gradients(
                zip(clipped_gradients, variables), global_step=global_step)
        debug_dict = {
            'predict_value': my_encoder.predict_value,
            'arch_emb': my_encoder.arch_emb,
            'image_emb': my_encoder.image_emb,
            'logits': my_decoder.logits,
            'encoder_input': my_encoder.x,
            'decoder_target': my_decoder.target,
        }
        return mse, cross_entropy, total_loss, learning_rate, train_op, global_step, grad_norm, decoder_correct_rate, debug_dict


def get_evaluate_ops(image_emb_eval, encoder_eval_input, encoder_eval_target, decoder_eval_target, params,
                  reuse=tf.AUTO_REUSE):
    with tf.variable_scope('EPD', reuse=reuse):
        my_encoder = encoder.Model(image_emb_eval, encoder_eval_input, encoder_eval_target, params, tf.estimator.ModeKeys.EVAL, 'Encoder', reuse)
        encoder_outputs =  my_encoder.arch_emb
        encoder_outputs.set_shape([None, params['decoder_hidden_size']])
        my_decoder = decoder.Model(encoder_outputs, decoder_eval_target, params, tf.estimator.ModeKeys.EVAL, 'Decoder', reuse)
        encoder_loss = my_encoder.loss
        decoder_loss = my_decoder.loss
        decoder_correct_rate = my_decoder.correct_rate
        mse = encoder_loss
        cross_entropy = decoder_loss

        total_loss = params['trade_off'] * encoder_loss + (1 - params['trade_off']) * decoder_loss + params['weight_decay'] * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        debug_dict = {
            'predict_value': my_encoder.predict_value,
            'encoder_input': my_encoder.x,
            'target_value': my_encoder.y,
        }    
        return mse, cross_entropy, total_loss, decoder_correct_rate, debug_dict


def get_predict_ops(image_emb_pred, encoder_pred_input, params, reuse=tf.AUTO_REUSE):
    encoder_pred_target = None
    decoder_pred_target = None
    with tf.variable_scope('EPD', reuse=reuse):
        my_encoder = encoder.Model(image_emb_pred, encoder_pred_input, encoder_pred_target, params, tf.estimator.ModeKeys.PREDICT, 'Encoder', reuse)
        encoder_outputs = my_encoder.arch_emb
        my_decoder = decoder.Model(encoder_outputs, decoder_pred_target, params, tf.estimator.ModeKeys.PREDICT, 'Decoder', reuse)
        arch_emb, predict_value, new_arch_emb = my_encoder.infer()
        sample_id = my_decoder.decode()
        return predict_value, sample_id, arch_emb, new_arch_emb


def build_model_train_graph(params):
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:0'):
            tf.logging.error('Building model train graph')
            image_ph = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
            encoder_input_ph = tf.placeholder(shape=[None, params['encoder_length']], dtype=tf.int32)
            encoder_target_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            decoder_target_ph = tf.placeholder(shape=[None, params['decoder_length']], dtype=tf.int32)
            model_id = params['model_id']
            with slim.arg_scope(id_to_arg_scope[model_id]):
                logits, end_points = id_to_model[model_id](
                    image_ph, num_classes=1001, is_training=False, scope=id_to_scope[model_id], reuse=tf.AUTO_REUSE)
                image = end_points['PreLogits']
                image_emb = tf.squeeze(image, [1, 2], name='SpatialSqueeze')
            train_mse, train_cross_entropy, train_loss, learning_rate, train_op, global_step, grad_norm, decoder_correct_rate, debug_dict = get_train_ops(
                image_emb, encoder_input_ph, encoder_target_ph, decoder_target_ph, params)
            merged_summary = tf.summary.merge_all()
            run_ops = [
                        train_mse,
                        train_cross_entropy,
                        train_loss,
                        learning_rate,
                        train_op,
                        global_step,
                        grad_norm,
                        decoder_correct_rate,
                        merged_summary,
                    ]
            return g, image_ph, encoder_input_ph, encoder_target_ph, decoder_target_ph, run_ops


def build_model_evaluate_graph(params):
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:0'):
            tf.logging.error('Building model evaluate graph')
            image_ph = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
            encoder_input_ph = tf.placeholder(shape=[None, params['encoder_length']], dtype=tf.int32)
            encoder_target_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            decoder_target_ph = tf.placeholder(shape=[None, params['decoder_length']], dtype=tf.int32)
            model_id = params['model_id']
            with slim.arg_scope(id_to_arg_scope[model_id]):
                logits, end_points = id_to_model[model_id](
                    image_ph, num_classes=1001, is_training=False, scope=id_to_scope[model_id], reuse=tf.AUTO_REUSE)
                image = end_points['PreLogits']
                image_emb = tf.squeeze(image, [1, 2], name='SpatialSqueeze')
            eval_mse, eval_cross_entropy, eval_loss, decoder_correct_rate, debug_dict = get_evaluate_ops(
                image_emb, encoder_input_ph, encoder_target_ph, decoder_target_ph, params)
            run_ops = [
                        eval_mse,
                        eval_cross_entropy,
                        eval_loss,
                        decoder_correct_rate,
                    ]
            return g, image_ph, encoder_input_ph, encoder_target_ph, decoder_target_ph, run_ops, debug_dict


def build_model_optim_graph(params):
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:0'):
            tf.logging.error('Building model optimizer graph')
            image_ph = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
            encoder_input_ph = tf.placeholder(shape=[None, params['encoder_length']], dtype=tf.int32)
            encoder_target_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            decoder_target_ph = tf.placeholder(shape=[None, params['decoder_length']], dtype=tf.int32)
            model_id = params['model_id']
            with slim.arg_scope(id_to_arg_scope[model_id]):
                logits, end_points = id_to_model[model_id](
                    image_ph, num_classes=1001, is_training=False, scope=id_to_scope[model_id], reuse=tf.AUTO_REUSE)
                image = end_points['PreLogits']
                image_emb = tf.squeeze(image, [1, 2], name='SpatialSqueeze')
            predict_value, sample_id, arch_emb, new_arch_emb = get_predict_ops(image_emb, encoder_input_ph, params)
            return g, image_ph, encoder_input_ph, encoder_target_ph, decoder_target_ph, predict_value, sample_id, image_emb, arch_emb, new_arch_emb