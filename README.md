# SSRL-for-image-classification
Semi-supervised Representation Learning for Remote Sensing Image Classification Based on Generative Adversarial Networks

The installed packages are listed is conda.yaml.
 
The whole training and test procedure is as follows.
  1. creat a conda env and install all necessary packages.
  2. run data_split.py  to create five folders of dataset under five-folder cross verification protocol.
  3. run mixM2_Res.py to train model and save models.
  4. run extract_feature.py to extract features from images.
  5. train_svm.py to train svm and test.
  6. Utils.py implement the data augmentation. Network.py defines the network architecture  of Generator and Discriminator.

If you find this repository useful, please cite the following paper.

Yan Peiyao; He Feng*; Yang Yajie; Hu Fei; Semi-Supervised Representation Learning for Remote Sensing Image Classification Based on Generative Adversarial Networks, IEEE Access, 2020, 8: 54135-54144.
   
