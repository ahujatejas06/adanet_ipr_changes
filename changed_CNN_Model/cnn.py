import tensorflow as tf
import numpy
import sys, os
import layers as L

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('keep_prob_hidden', 0.5, "dropout rate")
tf.app.flags.DEFINE_float('lrelu_a', 0.1, "lrelu slope")
tf.app.flags.DEFINE_boolean('top_bn', False, "")

def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    h = x
    rng = numpy.random.RandomState(seed)

    # Reduced initial channel size: 3 -> 64
    h = L.conv(h, ksize=3, stride=1, f_in=3, f_out=64, seed=rng.randint(123456), name='c1')
    h = L.lrelu(L.bn(h, 64, is_training=is_training, update_batch_stats=update_batch_stats, name='b1'), FLAGS.lrelu_a)

    # Using Depthwise Separable Convolutions for c3, c4, c5
    h = L.depthwise_separable_conv(h, ksize=3, stride=1, f_in=64, f_out=64, seed=rng.randint(123456), name='c2')
    h = L.lrelu(L.bn(h, 64, is_training=is_training, update_batch_stats=update_batch_stats, name='b2'), FLAGS.lrelu_a)
    h = L.depthwise_separable_conv(h, ksize=3, stride=1, f_in=64, f_out=128, seed=rng.randint(123456), name='c3')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b3'), FLAGS.lrelu_a)
    h = L.depthwise_separable_conv(h, ksize=3, stride=1, f_in=128, f_out=128, seed=rng.randint(123456), name='c4')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b4'), FLAGS.lrelu_a)

    h = L.max_pool(h, ksize=2, stride=2)

    # Further layers with reduced channels and without dropout
    h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=256, seed=rng.randint(123456), name='c5')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b5'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=512, seed=rng.randint(123456), padding="VALID", name='c6')
    h = L.lrelu(L.bn(h, 512, is_training=is_training, update_batch_stats=update_batch_stats, name='b6'), FLAGS.lrelu_a)

    # Global Average Pooling
    h1 = tf.reduce_mean(h, axis=[1, 2])  # Apply GAP for spatial invariance; use as intermediate feature representation

    # Final fully connected layer with dropout
    h = L.fc(h1, 512, 10, seed=rng.randint(123456), name='fc')
    h = tf.nn.dropout(h, rate=1 - FLAGS.keep_prob_hidden, seed=rng.randint(123456)) if stochastic else h

    if FLAGS.top_bn:
        h = L.bn(h, 10, is_training=is_training, update_batch_stats=update_batch_stats, name='bfc')

    return h, h1  # Return both the final output and the intermediate feature representation
