#! /bin/bash

<<<<<<< HEAD
mkdir data
cd data
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
mkdir bin
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
=======
mkdir data checkpoints
wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P data/
wget http://images.cocodataset.org/zips/train2014.zip -P data/
unzip data/train2014.zip -d data/
>>>>>>> 14ffcac8e52a7c6febbb4d8b2e58d9fab8e20646
