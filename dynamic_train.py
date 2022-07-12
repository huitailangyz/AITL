from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import copy
import json
import math
import time
import multiprocessing
import random


from dynamic_util import load_images_with_noise, parse_dataset, prepare_dataset, parse_dataset_optim
from dynamic_attack import build_attack_graph
from dynamic_evaluate import build_attack_evaluate_graph
from dynamic_model import build_model_train_graph, build_model_optim_graph

slim = tf.contrib.slim

import sys
sys.path.append("..")
from model import id_to_arg_scope, id_to_scope, id_to_saverscope, id_to_model, id_to_checkpoint, id_to_name



parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='model')

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

parser.add_argument('--AITL_encoder_vocab_size', type=int, default=20)

parser.add_argument('--AITL_decoder_vocab_size', type=int, default=20)

parser.add_argument('--AITL_trade_off', type=float, default=0.8)

parser.add_argument('--AITL_train_epochs', type=int, default=300)

parser.add_argument('--AITL_save_frequency', type=int, default=10)

parser.add_argument('--AITL_batch_size', type=int, default=100)

parser.add_argument('--AITL_lr', type=float, default=0.001)

parser.add_argument('--AITL_optimizer', type=str, default='adam')

parser.add_argument('--AITL_start_decay_step', type=int, default=100)

parser.add_argument('--AITL_decay_steps', type=int, default=1000)

parser.add_argument('--AITL_decay_factor', type=float, default=0.9)

parser.add_argument('--AITL_max_gradient_norm', type=float, default=5.0)

parser.add_argument('--AITL_predict_lambda', type=float, default=1)

parser.add_argument('--AITL_predict_num_steps', type=int, default=1)

parser.add_argument('--AITL_predict_num_seeds', type=int, default=1)

parser.add_argument('--AITL_image_hidden_size', type=int, default=128)

parser.add_argument('--AITL_num_sample_eval', type=int, default=100)

parser.add_argument('--AITL_num_sample_append', type=int, default=100)

parser.add_argument('--AITL_model_id', type=str, default="")

parser.add_argument('--attack_max_epsilon', type=float, default=16)

parser.add_argument('--attack_image_height', type=int, default=299)

parser.add_argument('--attack_image_width', type=int, default=299)

parser.add_argument('--attack_image_resize', type=int, default=330)

parser.add_argument('--attack_model', type=str, default="1")

parser.add_argument('--attack_num_iter', type=int, default=10)

parser.add_argument('--attack_num_noise', type=int, default=15)

parser.add_argument('--attack_untarget', type=bool, default=True)

parser.add_argument('--attack_init_num_sample', type=int, default=500)

parser.add_argument('--attack_pool_num_sample', type=int, default=500)

parser.add_argument('--evaluate_model', type=str, default='')


def get_AITL_params():
    params = {
        'model_dir': os.path.join(FLAGS.output_dir, 'AITL'),
        'sample_file': os.path.join(FLAGS.output_dir, 'sample.txt'),
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
        'trade_off': FLAGS.AITL_trade_off,
        'train_epochs': FLAGS.AITL_train_epochs,
        'save_frequency': FLAGS.AITL_save_frequency,
        'batch_size': FLAGS.AITL_batch_size,
        'lr': FLAGS.AITL_lr,
        'optimizer': FLAGS.AITL_optimizer,
        'start_decay_step': FLAGS.AITL_start_decay_step,
        'decay_steps': FLAGS.AITL_decay_steps,
        'decay_factor': FLAGS.AITL_decay_factor,
        'max_gradient_norm': FLAGS.AITL_max_gradient_norm,
        'predict_lambda': FLAGS.AITL_predict_lambda,
        'predict_num_seeds': FLAGS.AITL_predict_num_seeds,
        'predict_num_steps': FLAGS.AITL_predict_num_steps,
        'image_hidden_size': FLAGS.AITL_image_hidden_size,
        'num_sample_eval': FLAGS.AITL_num_sample_eval,
        'num_sample_append': FLAGS.AITL_num_sample_append,
        'model_id': FLAGS.AITL_model_id,
        'init_num_sample': FLAGS.attack_init_num_sample,
    }
    return params

def get_attack_params():
    params = {
        'sample_file': os.path.join(FLAGS.output_dir, 'sample.txt'),
        'encoder_length': FLAGS.AITL_encoder_length,
        'max_epsilon': FLAGS.attack_max_epsilon,
        'image_height': FLAGS.attack_image_height,
        'image_width': FLAGS.attack_image_width,
        'image_resize': FLAGS.attack_image_resize,
        'attack_model': FLAGS.attack_model,
        'num_iter': FLAGS.attack_num_iter,
        'num_noise': FLAGS.attack_num_noise,
        'untarget': FLAGS.attack_untarget,
        'init_num_sample': FLAGS.attack_init_num_sample,
        'pool_num_sample': FLAGS.attack_pool_num_sample,
        'batch_size': 1,
    }
    return params


def get_evaluate_params():
    params = {
        'eval_model': FLAGS.evaluate_model,
    }
    return params


def model_process(queue_lock, writer_lock, files):
    tf.logging.error("Enter model process")
    params = get_AITL_params()
    os.makedirs(params['model_dir'], exist_ok=True)
    flag = False
    while not flag:
        queue_lock.acquire()
        img_paths, op_list, value_list = parse_dataset(params['sample_file'])
        queue_lock.release()
        if len(img_paths) < params['init_num_sample']:
            time.sleep(60)
        else:
            flag = True


    epoch = 0
    while True:
        tf.logging.error("[Epoch %d] Start training" % epoch)
        queue_lock.acquire()
        img_paths, op_list, value_list = parse_dataset(params['sample_file'])
        queue_lock.release()
        g_train, image_ph, encoder_input_ph, encoder_target_ph, decoder_target_ph, run_ops = build_model_train_graph(params)
        with g_train.as_default():
            run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            image, image_path, encoder_input, encoder_target, decoder_target = prepare_dataset(img_paths, op_list, value_list, params['batch_size'], root_dir='./data/dev_images')
            data_ops = [image, image_path, encoder_input, encoder_target, decoder_target]
            saver = tf.train.Saver()
            checkpoint_saver_hook = tf.train.CheckpointSaverHook(params['model_dir'], save_steps=1000, saver=saver)
            hooks = [checkpoint_saver_hook]
            s = tf.train.Saver(slim.get_model_variables(scope=id_to_saverscope[params['model_id']]))
            start_time = time.time()
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.train.SingularMonitoredSession(config=config, hooks=hooks, checkpoint_dir=params['model_dir']) as sess:
                s.restore(sess, id_to_checkpoint[params['model_id']])
                batch_per_epoch = math.ceil(len(img_paths) / params['batch_size'])
                for step in range(batch_per_epoch):
                    image_v, image_path_v, encoder_input_v, encoder_target_v, decoder_target_v = sess.run(data_ops, options=run_opts)
                    train_mse_v, train_cross_entropy_v, train_loss_v, learning_rate_v, _, global_step_v, gn_v, decoder_cr_v, summary, *other = sess.run(run_ops, options=run_opts, feed_dict={
                        image_ph: image_v,
                        encoder_input_ph: encoder_input_v,
                        encoder_target_ph: encoder_target_v,
                        decoder_target_ph: decoder_target_v,
                    })
                    curr_time = time.time()
                    if (global_step_v+1) % 100 == 0 or step == batch_per_epoch - 1:
                        log_string = "epoch={:<6d} ".format(epoch)
                        log_string += "step={:<6d} ".format(global_step_v+1)
                        log_string += "mse={:<6f} ".format(train_mse_v)
                        log_string += "cross_entropy={:<6f} ".format(train_cross_entropy_v)
                        log_string += "loss={:<6f} ".format(train_loss_v)
                        log_string += "learning_rate={:<8.6f} ".format(learning_rate_v)
                        log_string += "|gn|={:<8.4f} ".format(gn_v)
                        log_string += "decode_correct_rate={:<8.4f} ".format(decoder_cr_v)
                        log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
                        tf.logging.error(log_string)


        tf.logging.error("[Epoch %d] Start optimizing" % epoch)
        queue_lock.acquire()
        num_sample = len(files)
        queue_lock.release()

        if num_sample < params['num_sample_append'] * 2:
            img_label_file = './data/dev_dataset.csv'
            dev = pd.read_csv(img_label_file)
            img_to_label = {dev.iloc[i]['ImageId']+'.png': dev.iloc[i]['TrueLabel'] for i in range(len(dev))}
            queue_lock.acquire()
            tf.logging.error("Now queue sample: %d" % num_sample)
            img_paths, op_list, value_list = parse_dataset_optim(params['sample_file'], params['num_sample_append'])
            queue_lock.release()

            g_optim, image_ph, encoder_input_ph, encoder_target_ph, decoder_target_ph, predict_value, sample_id, image_emb, arch_emb, new_arch_emb = build_model_optim_graph(params)
            with g_optim.as_default():
                image, image_path, encoder_input, encoder_target, decoder_target = prepare_dataset(img_paths, op_list, value_list, params['batch_size'], root_dir='./data/dev_images')
                data_ops = [image, image_path, encoder_input, encoder_target, decoder_target]

                with tf.train.SingularMonitoredSession(config=config, checkpoint_dir=params['model_dir']) as sess:
                    batch_per_epoch = math.ceil(len(img_paths) / params['batch_size'])
                    for step in range(batch_per_epoch):
                        image_v, image_path_v, encoder_input_v, encoder_target_v, decoder_target_v = sess.run(data_ops)

                        arch_emb_v = sess.run(arch_emb, feed_dict={
                                image_ph: image_v,
                                encoder_input_ph: encoder_input_v,
                            })
                        for step in range(params['predict_num_steps']):
                            arch_emb_v = sess.run(new_arch_emb, feed_dict={
                                image_ph: image_v,
                                encoder_input_ph: encoder_input_v,
                                arch_emb: arch_emb_v,
                            })
                        new_sample_id_v = sess.run(sample_id, feed_dict={
                                image_ph: image_v,
                                encoder_input_ph: encoder_input_v,
                                arch_emb: arch_emb_v,
                            })
                        for ix, image_path_ in enumerate(image_path_v):
                            queue_lock.acquire()
                            files.append([bytes.decode(image_path_), [new_sample_id_v[ix][i] for i in range(params['encoder_length'])], img_to_label[bytes.decode(image_path_)]])
                            queue_lock.release()
        epoch += 1



def attack_eval_process(queue_lock, writer_lock, files):
    tf.logging.error("Enter attack and eval process")
    img_label_file = './data/dev_dataset.csv'
    dev = pd.read_csv(img_label_file)
    img_to_label = {dev.iloc[i]['ImageId']+'.png': dev.iloc[i]['TrueLabel'] for i in range(len(dev))}
    imgs = list(img_to_label.keys())

    control_params = get_AITL_params()
    attack_params = get_attack_params()
    eval_params = get_evaluate_params()

    g_attack, x_input_attack, noise_x_input, x_labels, x_adv, op_chosen = build_attack_graph(attack_params)
    g_eval, x_input_eval, pred_list = build_attack_evaluate_graph(eval_params)

    tf.logging.error("Load attack checkpoint")
    with g_attack.as_default():
        sess_attack = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        model_list = attack_params['attack_model'].split('_')
        for model_id in model_list:
            s = tf.train.Saver(slim.get_model_variables(scope=id_to_saverscope[model_id]))
            s.restore(sess_attack, id_to_checkpoint[model_id])

    tf.logging.error("Load evaluate checkpoint")
    with g_eval.as_default():
        sess_eval = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        model_list = eval_params['eval_model'].split('_')
        for model_id in model_list:
            s = tf.train.Saver(slim.get_model_variables(scope=id_to_saverscope[model_id]))
            s.restore(sess_eval, id_to_checkpoint[model_id])

    def generate_samples(num):
        tf.logging.error("Attack and eval process acquire lock")
        queue_lock.acquire()
        for i in range(num):
            random_img = random.choice(imgs)
            random_op = [random.randint(0, control_params['encoder_vocab_size']-1) for _ in range(control_params['encoder_length'])]
            random_label = img_to_label[random_img]
            files.append([random_img, random_op, random_label])
        queue_lock.release()
        tf.logging.error("Attack and eval process release lock")


    generate_samples(attack_params['init_num_sample'])

    tf.logging.error("Start generate sample")
    writer = open(attack_params['sample_file'], 'a')

    while True:
        if len(files) == 0:
            generate_samples(attack_params['pool_num_sample'])

        queue_lock.acquire()
        random_img, random_op, random_label = files.pop()
        queue_lock.release()
        flag, images, noise_images = load_images_with_noise(random_img, imgs, attack_params['num_noise'])
        if not flag:
            continue
        labels = np.array([random_label])
        op_chosen_v = np.array([[random_op]])

        adv_images = sess_attack.run(x_adv, feed_dict={
                        x_input_attack: images,
                        noise_x_input: noise_images,
                        x_labels: labels,
                        op_chosen: op_chosen_v})

        pred = sess_eval.run(pred_list, feed_dict={x_input_eval: adv_images})
        success_count = 0
        for i in range(len(model_list)):
            if pred[i][0] != labels[0]:
                success_count += 1
        writer.write("%s " % (random_img))
        for i in range(len(random_op)):
            writer.write("%d " % (random_op[i]))
        writer.write("%.4f\n" % (success_count / len(model_list)))
        writer.flush()


def main(_):
    server = multiprocessing.Manager()
    queue_lock = server.Lock()
    writer_lock = server.Lock()
    files = server.list([])
    processes = []
    processes.append(multiprocessing.Process(target=model_process, args=(queue_lock, writer_lock, files), name='Model_Process'))
    processes.append(multiprocessing.Process(target=attack_eval_process, args=(queue_lock, writer_lock, files), name='Attack_Eval_Process'))

    for i in range(len(processes)):
        processes[i].start()

    for i in range(len(processes)):
        processes[i].join()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
