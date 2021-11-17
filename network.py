### semi_resnet network
import numpy as np

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from utils import *


def generator_simplified_api(inputs, is_train=True, reuse=False):
    image_size = 256
    k = 4
    # 128, 64, 32, 16，8，4
    s2, s4, s8, s16, s32, s64 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(
        image_size / 16), int(image_size / 32), int(image_size / 64)

    batch_size = 64
    gf_dim = 16  # Dimension of gen filters in first conv layer. [64]

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim * 32 * s64 * s64, W_init=w_init,
                            act=tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s64, s64, gf_dim * 32], name='g/h0/reshape')

        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim * 32, (k, k), out_size=(s64, s64), strides=(1, 1),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h1/batch_norm')
        net_h1 = tl.layers.ElementwiseLayer([net_h1, net_h0], combine_fn=tf.add, name='res_add1')

        net_h2 = DeConv2d(net_h1, gf_dim * 16, (k, k), out_size=(s32, s32), strides=(2, 2),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim * 16, (k, k), out_size=(s32, s32), strides=(1, 1),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h3/batch_norm')
        net_h3 = tl.layers.ElementwiseLayer([net_h3, net_h2], combine_fn=tf.add, name='res_add3')

        net_h4 = DeConv2d(net_h3, gf_dim * 8, (k, k), out_size=(s16, s16), strides=(2, 2),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h4/batch_norm')

        net_h5 = DeConv2d(net_h4, gf_dim * 8, (k, k), out_size=(s16, s16), strides=(1, 1),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h5/decon2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h5/batch_norm')
        net_h5 = tl.layers.ElementwiseLayer([net_h5, net_h4], combine_fn=tf.add, name='res_add5')

        net_h6 = DeConv2d(net_h5, gf_dim * 4, (k, k), out_size=(s8, s8), strides=(2, 2),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h6/decon2d')
        net_h6 = BatchNormLayer(net_h6, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h6/batch_norm')

        net_h7 = DeConv2d(net_h6, gf_dim * 4, (k, k), out_size=(s8, s8), strides=(1, 1),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h7/decon2d')
        net_h7 = BatchNormLayer(net_h7, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h7/batch_norm')
        net_h7 = tl.layers.ElementwiseLayer([net_h7, net_h6], combine_fn=tf.add, name='res_add7')

        net_h8 = DeConv2d(net_h7, gf_dim * 2, (k, k), out_size=(s4, s4), strides=(2, 2),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h8/decon2d')
        net_h8 = BatchNormLayer(net_h8, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h8/batch_norm')

        net_h9 = DeConv2d(net_h8, gf_dim * 2, (k, k), out_size=(s4, s4), strides=(1, 1),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h9/decon2d')
        net_h9 = BatchNormLayer(net_h9, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h9/batch_norm')
        net_h9 = tl.layers.ElementwiseLayer([net_h9, net_h8], combine_fn=tf.add, name='res_add9')

        net_h10 = DeConv2d(net_h9, gf_dim, (k, k), out_size=(s2, s2), strides=(2, 2),
                           padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h10decon2d')
        net_h10 = BatchNormLayer(net_h10, act=tf.nn.relu, is_train=is_train,
                                 gamma_init=gamma_init, name='g/h10/batch_norm')

        net_h11 = DeConv2d(net_h10, gf_dim, (k, k), out_size=(s2, s2), strides=(1, 1),
                           padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h11/decon2d')
        net_h11 = BatchNormLayer(net_h11, act=tf.nn.relu, is_train=is_train,
                                 gamma_init=gamma_init, name='g/h11/batch_norm')
        net_h11 = tl.layers.ElementwiseLayer([net_h11, net_h10], combine_fn=tf.add, name='res_add11')
        #
        net_h12 = DeConv2d(net_h11, 3, (k, k), out_size=(image_size, image_size), strides=(2, 2),
                           padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h12/decon2d')
        logits = net_h12.outputs
        net_h12.outputs = tf.nn.tanh(net_h12.outputs)
    return net_h12, logits





def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    k = 5
    df_dim = 16  # Dimension of discrim filters in first conv layer. [64]
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')

        net_h0 = Conv2d(net_in, df_dim, (k, k), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                        padding='SAME', W_init=w_init, name='d/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim, (k, k), (1, 1), act=None,
                        padding='SAME', W_init=w_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')
        net_h1 = tl.layers.ElementwiseLayer([net_h1, net_h0], combine_fn=tf.add, name='res_add1')

        net_h2 = Conv2d(net_h1, df_dim * 2, (k, k), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim * 2, (k, k), (1, 1), act=None,
                        padding='SAME', W_init=w_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')
        net_h3 = tl.layers.ElementwiseLayer([net_h3, net_h2], combine_fn=tf.add, name='res_add3')

        net_h4 = Conv2d(net_h3, df_dim * 4, (k, k), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='d/h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h4/batch_norm')

        net_h5 = Conv2d(net_h4, df_dim * 4, (k, k), (1, 1), act=None,
                        padding='SAME', W_init=w_init, name='d/h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h5/batch_norm')
        net_h5 = tl.layers.ElementwiseLayer([net_h5, net_h4], combine_fn=tf.add, name='res_add5')

        net_h6 = Conv2d(net_h5, df_dim * 8, (k, k), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='d/h6/conv2d')
        net_h6 = BatchNormLayer(net_h6, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h6/batch_norm')

        net_h7 = Conv2d(net_h6, df_dim * 8, (k, k), (1, 1), act=None,
                        padding='SAME', W_init=w_init, name='d/h7/conv2d')
        net_h7 = BatchNormLayer(net_h7, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h7/batch_norm')
        net_h7 = tl.layers.ElementwiseLayer([net_h7, net_h6], combine_fn=tf.add, name='res_add7')

        net_h8 = Conv2d(net_h7, df_dim * 16, (k, k), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='d/h8/conv2d')
        net_h8 = BatchNormLayer(net_h8, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h8/batch_norm')

        net_h9 = Conv2d(net_h8, df_dim * 16, (k, k), (1, 1), act=None,
                        padding='SAME', W_init=w_init, name='d/h9/conv2d')
        net_h9 = BatchNormLayer(net_h9, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d/h9/batch_norm')
        net_h9 = tl.layers.ElementwiseLayer([net_h9, net_h8], combine_fn=tf.add, name='res_add9')

        net_h10 = Conv2d(net_h9, df_dim * 32, (k, k), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='d/h10/conv2d')
        net_h10 = BatchNormLayer(net_h10, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='d/h10/batch_norm')

        net_h11 = Conv2d(net_h10, df_dim * 32, (k, k), (1, 1), act=None,
                         padding='SAME', W_init=w_init, name='d/h11/conv2d')
        net_h11 = BatchNormLayer(net_h11, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='d/h11/batch_norm')
        net_h11 = tl.layers.ElementwiseLayer([net_h11, net_h10], combine_fn=tf.add, name='res_add11')

        global_max1_f = MaxPool2d(net_h7, filter_size=(4, 4), strides=None, padding='SAME', name='maxpool1_f')
        global_max1_f = FlattenLayer(global_max1_f, name='d/h7/flatten')
        global_max2_f = MaxPool2d(net_h9, filter_size=(2, 2), strides=None, padding='SAME', name='maxpool2_f')
        global_max2_f = FlattenLayer(global_max2_f, name='d/h9/flatten')
        global_max3_f = FlattenLayer(net_h11, name='d/h11/flatten')

        feature = ConcatLayer(layers=[global_max1_f, global_max2_f, global_max3_f], name='d/concat_layer1')
        net_h12 = DenseLayer(feature, n_units=1, act=tf.identity,
                             W_init=w_init, name='d/h12/lin_sigmoid')
        logits = net_h12.outputs
        net_h12.outputs = tf.nn.sigmoid(net_h12.outputs)

        net_h13 = DenseLayer(feature, n_units=45, act=tf.identity,
                             W_init=w_init, name='d/h13/lin_sigmoid')



    return net_h12, logits, feature.outputs, net_h13.outputs # change net_h12 to net_h13



