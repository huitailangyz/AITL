# coding=utf-8
"""Evaluate the attack success rate under 13 models including normally trained models and adversarial training models"""

import os
import numpy as np
import logging
import tensorflow as tf
from util import load_labels, load_images, progress_bar
from model import id_to_arg_scope, id_to_scope, id_to_saverscope, id_to_model, id_to_checkpoint, id_to_name

slim = tf.contrib.slim


tf.flags.DEFINE_string('evaluate_model', '', 'The models used to evaluate.')
tf.flags.DEFINE_string('dataset', 'imagenet', 'The dataset used to attack.')
tf.flags.DEFINE_bool('untarget', True, 'untarget of target attack.')
tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_string('exp_name', '', 'Name of the experiment.')
tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')
FLAGS = tf.flags.FLAGS

def main(_):
    f2l = load_labels(FLAGS.untarget, FLAGS.dataset)
    input_dir = os.path.join(FLAGS.output_dir, FLAGS.exp_name, "adv_image")
    model_list = FLAGS.evaluate_model.split('_')
    batch_shape = [FLAGS.batch_size, 299, 299, 3]
    num_classes = 1001
    tf.logging.set_verbosity(tf.logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    
    handler = logging.FileHandler(os.path.join(FLAGS.output_dir, FLAGS.exp_name, "evaluate.txt"), mode='a')
    handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)

    pred_list = []
    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        for model_id in model_list:
            with slim.arg_scope(id_to_arg_scope[model_id]):
                logits, end_points = id_to_model[model_id](
                    x_input, num_classes=num_classes, is_training=False, scope=id_to_scope[model_id])
                pred = tf.argmax(end_points['Predictions'], 1)
                pred_list.append(pred)

        num_sample = 0
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            for model_id in model_list:
                s = tf.train.Saver(slim.get_model_variables(scope=id_to_saverscope[model_id]))
                s.restore(sess, id_to_checkpoint[model_id])
          
            success_count = np.zeros(len(model_list))
            idx = 0
            for filenames, images in load_images(input_dir, batch_shape):
                pred = sess.run(pred_list, feed_dict={x_input: images})

                for i_num, filename in enumerate(filenames):
                    label = f2l[filename]
                    for ix, model_id in enumerate(model_list):
                        if pred[ix][i_num] != label:
                            success_count[ix] += 1
                progress_bar(idx, 1000/FLAGS.batch_size)
                idx += 1
                num_sample += len(filenames)
            for ix, model_id in enumerate(model_list):
                logger.info("Attack Success Rate for {0} : {1:.1f}%".format(id_to_name[model_id], success_count[ix] * 1. / num_sample * 100))



def build_attack_evaluate_graph(params):
    pred_list = []
    model_list = params['eval_model'].split('_')
    batch_shape = [1, 299, 299, 3]
    num_classes = 1001
    tf.logging.error('Building attack evalute graph')
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:2'):
            x_input = tf.placeholder(tf.float32, shape=batch_shape)
            for model_id in model_list:
                with slim.arg_scope(id_to_arg_scope[model_id]):
                    logits, end_points = id_to_model[model_id](
                        x_input, num_classes=num_classes, is_training=False, scope=id_to_scope[model_id])
                    pred = tf.argmax(end_points['Predictions'], 1)
                    pred_list.append(pred)
    return g, x_input, pred_list