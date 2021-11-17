## make training dataset
# split the original images into five folders
from glob import glob
from random import shuffle
from utils import *
import os
import shutil
import cv2

"""
This file is used to split training set and test set to support the five-fold cross-validation 
protocol. 

Usage : see README.md
"""

data_dir ="/home/admin1/cactus_test/test_NWPU-RESISC45/NWPU-RESISC45/"
for classname in os.listdir(r'/home/admin1/cactus_test/test_NWPU-RESISC45/NWPU-RESISC45/'):
    data_class_dir=os.path.join(data_dir,classname)
    data_files=glob(os.path.join(data_class_dir,'*.jpg'))
    shuffle(data_files)
    print(classname)
    print(data_class_dir)

    n = 5
    for i in range(n):
        train_folder = os.path.join('/home/admin1/data1/dataset/NWPU-RESISC45/', 'folder%d/' % (i + 1),
                                    'train80/')
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        test_folder = os.path.join('/home/admin1/data1/dataset/NWPU-RESISC45/', 'folder%d/' % (i + 1), 'test20/')
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        # preprocess the tif images


        for x in range(140):
            image = cv2.imread(data_files[x + i * 140])
            img_path= data_files[x + i * 140]
            size = image.shape
            if size[0] != 256 or size[1] != 256:
                image = cv2.resize(image, (256, 256), 0, 0)
            img_name=img_path.split('/')[7]
            cv2.imwrite(test_folder + img_name[0:-4] + '.jpg', image)  # save image as jpg


        for x in range(700):
            if x < i * 140 + 140 and x >= i * 140:
                continue
            image = cv2.imread(data_files[x])
            img_path = data_files[x]
            size = image.shape
            if size[0] != 256 or size[1] != 256:
                image = cv2.resize(image, (256, 256), 0, 0)
            img_name=img_path.split('/')[7]
            cv2.imwrite(train_folder + img_name[0:-4] + '.jpg', image)  # save image as jpg
            #shutil.copy(data_files[x], train_folder)




