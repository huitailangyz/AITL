from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import time

import encoder
import decoder
from data_util import image_processing_wrap, collect_variables, modify_variables

import sys
sys.path.append("..")
from model import id_to_arg_scope, id_to_scope, id_to_saverscope, id_to_model, id_to_checkpoint
slim = tf.contrib.slim

def input_fn(image_path, encoder_input, encoder_target, decoder_target, batch_size, num_epochs=1, root_dir='./data/dev_images', shuffle=True):
    shape = np.array(encoder_input).shape
    tf.logging.info('Data size : {}, {}'.format(shape, np.array(encoder_target).shape))
    N = shape[0]
    encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int32)
    encoder_target = tf.convert_to_tensor(encoder_target, dtype=tf.float32)
    decoder_target = tf.convert_to_tensor(decoder_target, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image_path, encoder_input, encoder_target, decoder_target))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=N)
    dataset = dataset.map(image_processing_wrap(root_dir))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_one_shot_iterator()
    image, image_path, encoder_input, encoder_target, decoder_target = iterator.get_next()
    assert encoder_input.shape.ndims == 2
    assert encoder_target.shape.ndims == 2
    assert decoder_target.shape.ndims == 2
    assert image.shape.ndims == 4
    return image, image_path, encoder_input, encoder_target, decoder_target


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
    
        debug_dict = {
            'arch_emb': arch_emb,
            'new_arch_emb': new_arch_emb,
            'image_emb': my_encoder.image_emb,
        }
        return predict_value, sample_id, debug_dict

def train(params, image_path, encoder_input, encoder_target, decoder_target):
    with tf.Graph().as_default():
        tf.logging.info('Training Encoder-Predictor-Decoder')
        tf.logging.info('Preparing data')
        shape = np.array(encoder_input).shape
        image_train, _, encoder_train_input, encoder_train_target, decoder_train_target = input_fn(
            image_path,
            encoder_input,
            encoder_target,
            decoder_target,
            params['batch_size'],
            None,
        )
        tf.logging.info('Building model')
        model_id = params['model_id']
        with slim.arg_scope(id_to_arg_scope[model_id]):
            logits, end_points = id_to_model[model_id](
                image_train, num_classes=1001, is_training=False, scope=id_to_scope[model_id], reuse=tf.AUTO_REUSE)
            image = end_points['PreLogits']
            image_emb_train = tf.squeeze(image, [1, 2], name='SpatialSqueeze')
        train_mse, train_cross_entropy, train_loss, learning_rate, train_op, global_step, grad_norm, decoder_correct_rate, debug_dict = get_train_ops(
            image_emb_train, encoder_train_input, encoder_train_target, decoder_train_target, params)
        saver = tf.train.Saver(max_to_keep=10)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            params['model_dir'], save_steps=params['batches_per_epoch'] * params['save_frequency'], saver=saver)
        hooks = [checkpoint_saver_hook]
        merged_summary = tf.summary.merge_all()
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)

        s = tf.train.Saver(slim.get_model_variables(scope=id_to_saverscope[model_id]))
        with tf.train.SingularMonitoredSession(config=config, hooks=hooks, checkpoint_dir=params['model_dir']) as sess:
            s.restore(sess, id_to_checkpoint[model_id])
            writer = tf.summary.FileWriter(params['model_dir'], sess.graph)
            start_time = time.time()
            for step in range(params['train_epochs'] * params['batches_per_epoch']):
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
                train_mse_v, train_cross_entropy_v, train_loss_v, learning_rate_v, _, global_step_v, gn_v, decoder_cr_v, summary, *other = sess.run(run_ops)
                writer.add_summary(summary, global_step_v)
                epoch = (global_step_v+1) // params['batches_per_epoch']
                
                curr_time = time.time()
                if (global_step_v+1) % 100 == 0:
                    log_string = "epoch={:<6d} ".format(epoch)
                    log_string += "step={:<6d} ".format(global_step_v+1)
                    log_string += "mse={:<6f} ".format(train_mse_v)
                    log_string += "cross_entropy={:<6f} ".format(train_cross_entropy_v)
                    log_string += "loss={:<6f} ".format(train_loss_v)
                    log_string += "learning_rate={:<8.6f} ".format(learning_rate_v)
                    log_string += "|gn|={:<8.4f} ".format(gn_v)
                    log_string += "decode_correct_rate={:<8.4f} ".format(decoder_cr_v)
                    log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
                    tf.logging.info(log_string)


def evaluate(params, image_path, encoder_input, encoder_target, decoder_target):
    with tf.Graph().as_default():
        tf.logging.info('Evaluate Encoder-Predictor-Decoder')
        tf.logging.info('Preparing data')
        shape = np.array(encoder_input).shape
        image_eval, _, encoder_eval_input, encoder_eval_target, decoder_eval_target = input_fn(
            image_path,
            encoder_input,
            encoder_target,
            decoder_target,
            params['batch_size'],
            1,
        )
        tf.logging.info('Building model')
        model_id = params['model_id']
        with slim.arg_scope(id_to_arg_scope[model_id]):
            logits, end_points = id_to_model[model_id](
                image_eval, num_classes=1001, is_training=False, scope=id_to_scope[model_id], reuse=tf.AUTO_REUSE)
            image = end_points['PreLogits']
            image_emb_eval = tf.squeeze(image, [1, 2], name='SpatialSqueeze')
        eval_mse, eval_cross_entropy, eval_loss, decoder_correct_rate, debug_dict = get_evaluate_ops(
            image_emb_eval, encoder_eval_input, encoder_eval_target, decoder_eval_target, params)
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        eval_mse_list, eval_cross_entropy_list, eval_loss_list, decoder_correct_rate_list = [], [], [], []
        with tf.train.SingularMonitoredSession(
            config=config, checkpoint_dir=params['model_dir']) as sess:
            for step in range(params['batches_per_epoch']):
                run_ops = [
                    eval_mse,
                    eval_cross_entropy,
                    eval_loss,
                    decoder_correct_rate,
                ]
                eval_mse_v, eval_cross_entropy_v, eval_loss_v, decoder_cr_v = sess.run(run_ops)
                eval_mse_list.append(eval_mse_v)
                eval_cross_entropy_list.append(eval_cross_entropy_v)
                eval_loss_list.append(eval_loss_v)
                decoder_correct_rate_list.append(decoder_cr_v)
            
            log_string = "\n[Evaluate]\n"
            log_string += "mse={:<6f} ".format(np.mean(eval_mse_list))
            log_string += "cross_entropy={:<6f} ".format(np.mean(eval_cross_entropy_list))
            log_string += "loss={:<6f} ".format(np.mean(eval_loss_list))
            log_string += "decode_correct_rate={:<8.4f} ".format(np.mean(decoder_correct_rate_list))
            tf.logging.info(log_string)
