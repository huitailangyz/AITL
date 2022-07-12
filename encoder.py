from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf

import sys
sys.path.append("..")
from model import id_to_arg_scope, id_to_scope, id_to_saverscope, id_to_model, id_to_checkpoint
slim = tf.contrib.slim


_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5


class Encoder(object):
    def __init__(self, params, mode, W_emb):
        self.num_layers = params['encoder_num_layers']
        self.hidden_size = params['encoder_hidden_size']
        self.emb_size = params['encoder_emb_size']
        self.mlp_num_layers = params['mlp_num_layers']
        self.mlp_hidden_size = params['mlp_hidden_size']
        self.mlp_dropout = params['mlp_dropout']
        self.encoder_length = params['encoder_length']
        self.vocab_size = params['encoder_vocab_size']
        self.dropout = params['encoder_dropout']
        self.image_hidden_size = params['image_hidden_size']
        self.model_id = "1"
        self.W_emb = W_emb
        self.mode = mode

    def build_encoder(self, image_emb, x, batch_size, is_training):
        # process x
        self.batch_size = batch_size
        assert x.shape.ndims == 2, '[batch_size, length]'
        x = tf.gather(self.W_emb, x) # [batch_size, length, emb_size]
        x = tf.reshape(x, [-1, self.encoder_length * self.emb_size])

        for i in range(self.num_layers):
            name = 'encoder/mlp_{}'.format(i)
            x = tf.layers.dense(x, self.hidden_size, activation=tf.nn.relu, name=name)
            x = tf.layers.dropout(x, self.dropout, training=is_training)
            x = tf.layers.batch_normalization(
                x, axis=1,
                momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                center=True, scale=True, training=is_training, fused=True
            )
        x = tf.nn.l2_normalize(x, dim=-1)
        self.arch_emb = x

        image = tf.layers.dense(image_emb, self.image_hidden_size, activation=tf.nn.relu, name='image/fc')
        image = tf.nn.l2_normalize(image, dim=-1)
        self.image_emb = image

        # process predictor
        x = tf.concat([self.arch_emb, image], axis=1)
        
        for i in range(self.mlp_num_layers):
            name = 'predictor/mlp_{}'.format(i)
            x = tf.layers.dense(x, self.mlp_hidden_size, activation=tf.nn.relu, name=name)
            x = tf.layers.dropout(x, self.mlp_dropout, training=is_training)

        self.predict_value = tf.layers.dense(x, 1, activation=tf.sigmoid, name='regression')
        return {
            'arch_emb' : self.arch_emb,
            'image_emb' : self.image_emb,
            'predict_value' : self.predict_value,
        }


class Model(object):
    def __init__(self, image, x, y, params, mode, scope='Encoder', reuse=tf.AUTO_REUSE):
        self.image = image
        self.x = x
        self.y = y
        self.params = params
        self.batch_size = tf.shape(x)[0]
        self.vocab_size = params['encoder_vocab_size']
        self.emb_size = params['encoder_emb_size']
        self.hidden_size = params['encoder_hidden_size']
        self.encoder_length = params['encoder_length']
        self.weight_decay = params['weight_decay']
        self.mode = mode
        self.is_training = self.mode == tf.estimator.ModeKeys.TRAIN

        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        tf.get_variable_scope().set_initializer(initializer)
        self.build_graph(scope=scope, reuse=reuse)

  
    def build_graph(self, scope=None, reuse=tf.AUTO_REUSE):
        tf.logging.info("# creating %s graph ..." % self.mode)
        # Encoder
        with tf.variable_scope(scope, reuse=reuse):
            self.W_emb = tf.get_variable('W_emb', [self.vocab_size, self.emb_size])
            self.arch_emb, self.predict_value = self.build_encoder()
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                self.compute_loss()
            else:
                self.loss = None
                self.total_loss = None

    def build_encoder(self):
        encoder = Encoder(self.params, self.mode, self.W_emb)
        res = encoder.build_encoder(self.image, self.x, self.batch_size, self.is_training)
        self.image_emb = res['image_emb']
        return res['arch_emb'], res['predict_value']

    def compute_loss(self):
        weights = 1 - tf.cast(tf.equal(self.y, -1.0), tf.float32) 
        mean_squared_error = tf.losses.mean_squared_error(
            labels=self.y, 
            predictions=self.predict_value,
            weights=weights)
        
        tf.summary.scalar('mean_squared_error', mean_squared_error)
        self.loss = tf.identity(mean_squared_error, name='squared_error')
        total_loss = mean_squared_error + self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.total_loss = total_loss

    def train(self):
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.constant(self.params['lr'])
        if self.params['optimizer'] == "sgd":
            self.learning_rate = tf.cond(
                self.global_step < self.params['start_decay_step'],
                lambda: self.learning_rate,
                lambda: tf.train.exponential_decay(
                        self.learning_rate,
                        (self.global_step - self.params['start_decay_step']),
                        self.params['decay_steps'],
                        self.params['decay_factor'],
                        staircase=True),
                name="learning_rate")
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.params['optimizer'] == "adam":
            assert float(self.params['lr']) <= 0.001, "! High Adam learning rate %g" % self.params['lr']
            opt = tf.train.AdamOptimizer(self.learning_rate)
        elif self.params['optimizer'] == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*opt.compute_gradients(self.total_loss))
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.params['max_gradient_norm'])
            self.train_op = opt.apply_gradients(
                zip(clipped_gradients, variables), global_step=self.global_step)

        tf.identity(self.learning_rate, 'learning_rate')
        tf.summary.scalar("learning_rate", self.learning_rate),
        tf.summary.scalar("total_loss", self.total_loss),
    
        return {
            'train_op' : self.train_op,
            'loss' : self.total_loss,
        }

    def eval(self):
        assert self.mode == tf.estimator.ModeKeys.EVAL
        return {
            'loss': self.total_loss,
        }

    def infer(self):
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        grads_on_outputs = tf.gradients(self.predict_value, self.arch_emb)[0]
        new_arch_emb = self.arch_emb - self.params['predict_lambda'] * grads_on_outputs
        new_arch_emb = tf.nn.l2_normalize(new_arch_emb, dim=-1)
        return self.arch_emb, self.predict_value, new_arch_emb