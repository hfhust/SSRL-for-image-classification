import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from utils import *
from network import *

# import tflib as lib
# import tflib.plot
#import tflib.inception_score


pp = pprint.PrettyPrinter()

"""


Usage : see README.md
"""

flags = tf.app.flags
flags.DEFINE_integer("epoch", 121, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 200000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 50, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "train6x80", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "sampless", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS





def main(_):
    pp.pprint(flags.FLAGS.__flags)
    os.chdir('/home/admin1/data1/hefeng/mySemi/NWM2-Res-80ratio-FiveFolder')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    z_dim = 100

    # with tf.device("/gpu:0"): # <-- if you have a GPU machine
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    real_images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim],
                                 name='real_images')
    label = tf.placeholder(tf.float32, [FLAGS.batch_size, 45], name='real_labels')
    d_logits_real = tf.placeholder(tf.float32, [FLAGS.batch_size, 45], name='real_labels')
    d_logits_false = tf.placeholder(tf.float32, [FLAGS.batch_size, 45], name='false_labels')

    # z --> generator for training
    net_g, g_logits = generator_simplified_api(z, is_train=True, reuse=False)
    # generated fake images --> discriminator
    net_d, d_logits_f, feature_fake, d_logits_f45 = discriminator_simplified_api(net_g.outputs, is_train=True,
                                                                                 reuse=False)
    # real images --> discriminator
    net_d_r, d_logits_r, feature_real, d_logits_r45 = discriminator_simplified_api(real_images, is_train=True,
                                                                                   reuse=True)

    mixup_lam = tf.placeholder(tf.float32, [64, 1, 1, 1], name='mixup_lam')
    mixup_lam2 = tf.placeholder(tf.float32, [64, 1], name='mixup_lam2')
    x_mixup = real_images * mixup_lam + net_g.outputs * (1 - mixup_lam)
    net_d_mixup, d_logits_mixup, feature_mixup, d_logits_45_mixup = discriminator_simplified_api(x_mixup,
                                                                                                 is_train=True,
                                                                                                 reuse=True)
    real_images_labels = tf.ones_like(d_logits_r)
    net_g_outputs_d_logits = tf.zeros_like(d_logits_f)
    y_mixup1 = real_images_labels * mixup_lam2 + net_g_outputs_d_logits * (1 - mixup_lam2)
    d_loss_mixup1 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_mixup, labels=y_mixup1))
    ############将D对于生成图片的预测输出d_logits_f45作为假图片的标签，与真实图片进行mixup
    real_images_labels2 = d_logits_real
    ##求每一行最大值所在的列
    d_logits_pred = tf.argmax(d_logits_f45, axis=1)

    # 对label进行one-hot编码
    #  net_g_outputs_d_logits2 = keras.utils.to_categorical(d_logits_pred, num_classes=45)
    number = tf.size(d_logits_pred)
    d_logits_pred = tf.expand_dims(d_logits_pred, 1)
    indices = tf.expand_dims(tf.range(0, number, 1), 1)
    d_logits_pred = tf.to_int32(d_logits_pred, name='ToInt32')
    concated = tf.concat([indices, d_logits_pred], 1)
    net_g_outputs_d_logits2 = tf.sparse_to_dense(concated, tf.stack([number, 45]), 1.0, 0.0)

    y_mixup2 = real_images_labels2 * mixup_lam2 + net_g_outputs_d_logits2 * (1 - mixup_lam2)
    d_loss_mixup2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_45_mixup, labels=y_mixup2))
    ############################################
    d_loss_mixup = d_loss_mixup1  +d_loss_mixup2
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_r, labels=tf.ones_like(d_logits_r)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f, labels=tf.zeros_like(d_logits_f)))
    d_loss_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_r45, labels=d_logits_real))

    d_loss = d_loss_real + d_loss_fake + d_loss_sup + d_loss_mixup

    g_loss1 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f, labels=tf.ones_like(d_logits_f)))
    g_loss2 = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (FLAGS.image_size * FLAGS.image_size)
    g_loss = g_loss1 + g_loss2

    # sample_z --> generator for evaluation, set is_train to False
    # so that BatchNormLayer behave differently
    net_g2, g2_logits = generator_simplified_api(z, is_train=False, reuse=True)

    net_d3, d3_logits, _, d3_logits_r45 = discriminator_simplified_api(real_images, is_train=False, reuse=True)

    # trainable parameters for updating discriminator and generator
    g_vars = net_g.all_params  # only updates the generator
    d_vars = net_d.all_params  # only updates the discriminator

    net_g.print_params(False)
    print("---------------")
    net_d.print_params(False)

    # optimizers for updating discriminator and generator
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
        .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
        .minimize(g_loss, var_list=g_vars)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # this gpu order is same as nvidia-smi showed
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.InteractiveSession(config=config)

    n = 5
    for i in range(n):
        sess.run(tf.initialize_all_variables())

        if not os.path.exists(FLAGS.checkpoint_dir + '%d' % (i + 1)):
            os.makedirs(FLAGS.checkpoint_dir + '%d' % (i + 1))

        # load checkpoints
        print("[*] Loading checkpoints...")
        model_dir = "%s_%s_%s" % (FLAGS.dataset, 64, FLAGS.output_size)
        save_dir = os.path.join(FLAGS.checkpoint_dir + '%d' % (i + 1), model_dir)
        # load the latest checkpoints
        # for num in xrange(70, 71):
        net_g_name = os.path.join(save_dir, 'net_g.npz')
        net_d_name = os.path.join(save_dir, 'net_d.npz')

        print(net_g_name, net_d_name)

        if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
            print("[!] Loading checkpoints failed!")
        else:
            net_g_loaded_params = tl.files.load_npz(name=net_g_name)
            net_d_loaded_params = tl.files.load_npz(name=net_d_name)
            tl.files.assign_params(sess, net_g_loaded_params, net_g)
            tl.files.assign_params(sess, net_d_loaded_params, net_d)
            print("[*] Loading checkpoints SUCCESS!")

        # TODO: use minbatch to shuffle and iterate
        # data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
        data_files = glob(
            os.path.join("/home/admin1/dataset/NWPU-RESISC45/folder%d/" % (i + 1), FLAGS.dataset, "*.jpg"))

        # load labels
        NUM_STYLE_LABELS = 45
        style_label_file = './style_names.txt'
        style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
        if NUM_STYLE_LABELS > 0:
            style_labels = style_labels[:NUM_STYLE_LABELS]

        if FLAGS.is_train:

            # iter_counter = 0
            for epoch in range(FLAGS.epoch):
                # shuffle data
                shuffle(data_files)
                print("[*]Dataset shuffled!")
                # load labels
                batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size
                lens = batch_idxs * FLAGS.batch_size
                print(lens)
                y = np.zeros((lens, 45), dtype=np.uint8)
                for m in range(lens):
                    for j in range(len(style_labels)):
                        if style_labels[j] in data_files[m]:
                            y[m][j] = 1
                            break

                # load image data

                for idx in range(batch_idxs):
                    batch_files = data_files[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]
                    # get real images
                    batch = [
                        get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size,
                                  is_grayscale=0) for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
                    # get real labels
                    batch_labels = y[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]
                    batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)

                    mixup_alpha = 1
                    batch_mixup_lam = np.random.beta(mixup_alpha, mixup_alpha, size=64)
                    batch_mixup_lam = batch_mixup_lam.reshape(64, 1, 1, 1)
                    batch_mixup_lam2 = batch_mixup_lam.reshape(64, 1)
                    start_time = time.time()
                    # updates the discriminator
                    errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images,
                                                                     d_logits_real: batch_labels,
                                                                     mixup_lam: batch_mixup_lam,
                                                                     mixup_lam2: batch_mixup_lam2})
                    # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
                    for _ in range(2):
                        errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z, real_images: batch_images,
                                                                         d_logits_real: batch_labels,
                                                                         mixup_lam: batch_mixup_lam,
                                                                         mixup_lam2: batch_mixup_lam2})
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, FLAGS.epoch, idx, batch_idxs,
                             time.time() - start_time, errD, errG))
                    sys.stdout.flush()

                if np.mod(epoch, 5) == 0 and epoch > 89:
                    print(epoch)
                    # save current network parameters
                    print("[*] Saving checkpoints...")
                    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
                    save_dir = os.path.join(FLAGS.checkpoint_dir + '%d' % (i + 1), model_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # the latest version location
                    net_g_name = os.path.join(save_dir, str(epoch) + 'net_g.npz')
                    net_d_name = os.path.join(save_dir, str(epoch) + 'net_d.npz')

                    tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                    tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)

                    print("[*] Saving checkpoints SUCCESS!")



if __name__ == '__main__':
        tf.app.run()