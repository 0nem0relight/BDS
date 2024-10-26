# Enhancing Out-of-Distribution Detection with Bilateral Distribution Score

This is the official repository of  the paper Enhancing Out-of-Distribution Detection with Bilateral Distribution Score

# Setup

## Imagenet benchmark

[ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index): Please download and place the training data and validation data in
`./datasets/imagenet/train` and  `./datasets/imagenet/val`, respectively.

[Texture](https://www.robots.ox.ac.uk/~vgg/data/dtd/): Please download and place the dataset in `./datasets/ood_data`

For other OOD datasets 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf),please follow the instructions in the [link](https://github.com/deeplearning-wisc/knn-ood) to prepare.Then,put each dateset into `./datasets/ood_data` 

## CIFAR-10 benchmark

CIFAR-10 : The downloading process will start immediately upon running.

For OOD datasets we provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_data/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_data/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_data/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_data/LSUN`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_data/iSUN`.
* [LSUN_fix](https://drive.google.com/file/d/1KVWj9xpHfVwGcErH5huVujk9snhEGOxE/view?usp=sharing): download it and place it in the folder of `datasets/ood_data/LSUN_fix`.
* [ImageNet_fix](https://drive.google.com/file/d/1sO_-noq10mmziB1ECDyNhD5T4u5otyKA/view?usp=sharing): download it and place it in the folder of `datasets/ood_data/ImageNet_fix`.
* [ImageNet_resize](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz): download it and place it in the folder of `datasets/ood_data/Imagenet_resize`.
 
## Checkpont

Please download [Pre-trained models](https://drive.google.com/file/d/1wku0D4bfhPXMPGmMtO8NUpepWikHyvsY/view?usp=drive_link) and place in the `./checkpoints` folder.

# Run Experiments

## CIFAR-10 benchmark
run `run_cifar.py`

## Imagenet benchmark
run `run_imgnet.py`

# Acknowledgements
Parts of our codebase have been adopted from the repositories for [KNN-OOD](https://github.com/deeplearning-wisc/knn-ood)
