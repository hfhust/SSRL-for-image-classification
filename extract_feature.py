import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from random import shuffle
from tensorlayer.layers import *
from utils import *
from network import *

pp = pprint.PrettyPrinter()

"""
TensorLayer implementation of DCGAN to generate face image.

Usage : see README.md
"""

flags = tf.app.flags
flags.DEFINE_integer("epoch", 121, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 200000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 50, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "train80", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "sampless", "Directory name to save the image samples [samples]")
flags.DEFINE_string("feature_dir", "features", "Directory name to save features")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    #os.chdir('/home/admin1/data1/hefeng/mySemi/UcMix1-FiveFolder')

    real_images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim],
                                 name='real_images')
    # z --> generator for training

    net_d, d_logits, features, _ = discriminator_simplified_api(real_images, is_train=FLAGS.is_train,
                                                                           reuse=False)
    # net_d, d_logits, features = discriminator_simplified_api(real_images, is_train=FLAGS.is_train,reuse=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # this gpu order is same as nvidia-smi showed
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.InteractiveSession(config=config)

    # sess=tf.Session()
    # tl.utils.set_gpu_fraction( gpu_fraction=0.88)
    sess.run(tf.initialize_all_variables())

    n = 5
    for i in range(n):
        # load checkpoints
        print("[*] Loading checkpoints...")
        model_dir = "%s_%s_%s" % (FLAGS.dataset, 64, FLAGS.output_size)
        save_dir = os.path.join(FLAGS.checkpoint_dir + '%d'%(i + 1), model_dir)
        # print save_dir
        # load the latest checkpoints

        nums = [ 90, 95,100,105,110,115,120]
        # 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
        #
        for num in nums:
            net_g_name = os.path.join(save_dir, '%dnet_g.npz' % num)
            net_d_name = os.path.join(save_dir, '%dnet_d.npz' % num)

            print(net_g_name, net_d_name)

            if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
                print("[!] Loading checkpoints failed!")
                exit()
            else:
                net_d_loaded_params = tl.files.load_npz(name=net_d_name)
                tl.files.assign_params(sess, net_d_loaded_params, net_d)
                print("[*] Loading checkpoints SUCCESS!")

            NUM_STYLE_LABELS = 45
            style_label_file = './style_names.txt'
            style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
            if NUM_STYLE_LABELS > 0:
                style_labels = style_labels[:NUM_STYLE_LABELS]

            if not os.path.exists(FLAGS.feature_dir + '%d'%(i + 1)):
                os.makedirs(FLAGS.feature_dir + '%d'%(i + 1))

            print('extract traning feature')

            # data_files = glob(os.path.join("./data_withoutagu", 'uc_train_256_data', "*.jpg"))
            data_files = glob(
                os.path.join("/home/admin1/dataset/NWPU-RESISC45/folder" + '%d' % (i + 1), FLAGS.dataset,
                             "*.jpg"))  # 21*80*4=6720
            shuffle(data_files)

            batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

            lens = batch_idxs * FLAGS.batch_size
            print('here')
            #print(batch_idxs)
            print(lens)

            y = np.zeros(lens, dtype=np.uint8)
            for m in range(lens):
                for j in range(len(style_labels)):
                    if style_labels[j] in data_files[m]:
                        y[m] = j
                        break

            feats = np.zeros((lens, 14336))

            for idx in range(batch_idxs):
                batch_files = data_files[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]
                # get real images
                batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size,
                                   is_grayscale=0) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                # update sample files based on shuffled data
                # img, errG = sess.run([net_g2.outputs, g_loss], feed_dict={z : sample_seed})
                feat = sess.run(features, feed_dict={real_images: batch_images})

                # print feat.shape

                begin = FLAGS.batch_size * idx
                end = FLAGS.batch_size + begin
                feats[begin:end, ...] = feat
                #print(idx)
            np.save('features%d/features%d_train.npy' % (i + 1, num), feats)
            np.save('features%d/label%d_train.npy' % (i + 1, num), y)

            print('extract testing feature')
            data_files = glob(
                os.path.join("/home/admin1/dataset/NWPU-RESISC45/folder" + '%d' % (i + 1), 'test20', "*.jpg"))
            shuffle(data_files)
            # data_files = data_files[0:5000]

            batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

            lens = batch_idxs * FLAGS.batch_size
            print(lens)

            y = np.zeros(lens, dtype=np.uint8)
            for m in range(lens):
                for j in range(len(style_labels)):
                    if style_labels[j] in data_files[m]:
                        y[m] = j
                        break

            feats = np.zeros((lens, 14336))

            for idx in range(batch_idxs):
                batch_files = data_files[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]

                batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size,
                                   is_grayscale=0) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                # update sample files based on shuffled data
                # img, errG = sess.run([net_g2.outputs, g_loss], feed_dict={z : sample_seed})
                feat = sess.run(features, feed_dict={real_images: batch_images})

                begin = FLAGS.batch_size * idx
                end = FLAGS.batch_size + begin
                feats[begin:end, ...] = feat

                #print(idx)

            np.save('features%d/features%d_test.npy' % (i + 1, num), feats)
            np.save('features%d/label%d_test.npy' % (i + 1, num), y)








if __name__ == '__main__':
    tf.app.run()
