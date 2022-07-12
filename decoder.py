from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

INF=1<<16

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
class Decoder():
    def __init__(self, params):
        self.num_layers = params['decoder_num_layers']
        self.hidden_size = params['decoder_hidden_size']
        self.length = params['decoder_length']
        self.source_length = params['encoder_length']
        self.vocab_size = params['decoder_vocab_size']
        self.dropout = params['decoder_dropout']

    def build_decoder(self, encoder_outputs, batch_size, is_training):
        self.batch_size = batch_size
        x = encoder_outputs
        for i in range(self.num_layers):
            name = 'decoder/mlp_{}'.format(i)
            x = tf.layers.dense(x, self.hidden_size, activation=tf.nn.relu, name=name)
            x = tf.layers.dropout(x, self.dropout, training=is_training)
            x = tf.layers.batch_normalization(
                x, axis=1,
                momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                center=True, scale=True, training=is_training, fused=True
            )
        x = tf.layers.dense(x, self.length * self.vocab_size, name='decoder/output')
        x = tf.reshape(x, [-1, self.length, self.vocab_size])
        logits = x
        sample_id = tf.argmax(x, axis=-1, output_type=tf.int32)

        return logits, sample_id


class Model(object):
    def __init__(self,
                encoder_outputs,
                target,
                params,
                mode,
                scope=None,
                reuse=tf.AUTO_REUSE):
        self.params = params
        self.encoder_outputs = encoder_outputs
        self.target = target
        self.batch_size = tf.shape(self.encoder_outputs)[0]
        self.mode = mode
        self.vocab_size = params['decoder_vocab_size']
        self.num_layers = params['decoder_num_layers']
        self.decoder_length = params['decoder_length']
        self.hidden_size = params['decoder_hidden_size']
        self.weight_decay = params['weight_decay']
        self.is_training = mode == tf.estimator.ModeKeys.TRAIN

        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        tf.get_variable_scope().set_initializer(initializer)

        self.build_graph(scope=scope, reuse=reuse)

    def build_graph(self, scope=None, reuse=tf.AUTO_REUSE):
        tf.logging.info("# creating %s graph ..." % self.mode)
        with tf.variable_scope(scope, reuse=reuse):
            self.logits, self.sample_id = self.build_decoder()

            if self.mode != tf.estimator.ModeKeys.PREDICT:
                self.compute_loss()
            else:
                self.loss = None
                self.total_loss = None
  
    def build_decoder(self):
        decoder = Decoder(self.params)
        logits, sample_id = decoder.build_decoder(
            self.encoder_outputs, self.batch_size, self.is_training)
        return logits, sample_id

    def compute_loss(self):
        target_output = self.target
        crossent = tf.losses.sparse_softmax_cross_entropy(
            labels=target_output, logits=self.logits)
        self.loss = tf.identity(crossent, 'cross_entropy')
        total_loss = crossent + self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.total_loss = total_loss
        self.correct_rate = tf.reduce_mean(tf.to_float(tf.equal(target_output, self.sample_id)))

  
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
        tf.summary.scalar("train_loss", self.total_loss),

        return {
            'train_op': self.train_op, 
            'loss' : self.total_loss
        }

    def eval(self):
        assert self.mode == tf.estimator.ModeKeys.EVAL
        return {
            'loss' : self.total_loss,
        }

    def infer(self):
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        return {
            'logits' : self.logits,
            'sample_id' : self.sample_id,
        }

    def decode(self):
        res = self.infer()
        sample_id = res['sample_id']
        return sample_id