# Polarity Loss for Zero-shot Object Detection

This code is the testing side implementation of the following work:

Shafin Rahman, Salman Khan, and Nick Barnes. 
"Polarity Loss for Zero-shot Object Detection." 
arXiv preprint arXiv:1811.08982 (2018). ([Project Page](https://salman-h-khan.github.io/ProjectPages/ZSD_Arxiv19.html))

![OverviewFigure](https://salman-h-khan.github.io/images/Fig2_PL-ZSD.JPG)

## Requirements

* Download the trained model avaiabe on the link below and place it inside the "Model" directory ([Link to pre-trained model (h5 format)](https://www.dropbox.com/s/97gfrngizymricd/resnet50_polar_loss.h5?dl=0)). 
* Other requirements:
    - Python 2.7 (or 3.6)
    - Keras 2.1.4
    - OpenCV 3.3.0
    - Tensorflow 1.3.0
 * We have provided `.yaml` files in the `Config/` directory to seamlessly set-up conda enviornemnt with all the required dependencies. Simply run `conda env create --file=environment_keras_pyZZ.yaml` and replace `ZZ` with `27` or `36` depending on the python version. 

## Files

* `sample_input.txt`: a sample input file containing test image paths
* `gzsd.py`: to perform generalized zero-shot detection task using sample_input.txt
* `keras_retinanet`: directory containing the supporting code of the model. This directory is a modified version from original RetinaNet implementation [1] ([Link](https://github.com/fizyr/keras-retinanet))
* `Dataset`: directory containing sample input and output images.
* `Config`: directory containing configuration files to set up conda environment. 
* `Model`: directory containing a pre-trained model using the polarity loss.
* `MSCOCO`: This directory contains the source of data, proposed 65/15- seen/unseen split and experimental protocol used in experiments.
    - `cls_names_seen_coco.csv`: list of 65 MSCOCO seen classes. Each line contains a class name followed by an index.
    - `cls_names_test_coco.csv`: list of 80 MSCOCO object classes. Each line contains a class name followed by an index. Index 0 to 64 are from seen objects, and index 65 to 79 are from unseen.
    - `train_coco_seen_all.zip`: it is a zip version of csv file `train_coco_seen_all.csv` containing training image paths and annotations used in the paper. Each line contains a training image path, a bounding box co-ordinate and the ground-truth class name of that bounding box. For example, Filepath,x1,y1,x2,y2,class_name
    - `validation_coco_seen_all.csv`: test images with annotations for traditional object detection on only seen objects. File format, Filepath,x1,y1,x2,y2,class_name
    - `validation_coco_unseen_all.csv`: test images with annotations for zero-shot object detection on only unseen objects. File format, Filepath,x1,y1,x2,y2,class_name
    - `validation_coco_unseen_seen_all_gzsd.csv`: test images with annotations for generalized zero-shot object detection on both seen and unseen objects together. File format, Filepath,x1,y1,x2,y2,class_name
    - `word_w2v.txt`, `word_glo.txt`, and `word_ftx.txt`: word2vec, GloVe and FastText word vectors for 80 classes of MSCOCO.  The ith column represents the 300-dimensional word vectors of the class name of the ith row of `cls_names_test_coco.csv`
    - `vocabulary_list`: The list of vocabulary atoms from NUS-WIDE tag dataset [2] used in the paper.
    - `vocabulary_w2v.txt`, `vocabulary_glo.txt`, and `vocabulary_ftx.txt`: word2vec, GloVe and FastText word vectors of all vocabulary tags.  The ith column represents the 300-dimensional word vectors of the class name of the ith row of `vocabulary_list.txt`

## Running instruction
To run generalized zero-shot detection on sample input kept in `Dataset/Sampleinput`, simply run `gzsd.py` after installing all dependencies like Keras, Tensorflow, OpenCV and placing the pre-trained model in the `Model` directory. This code will generate the output files for each input image to `Dataset/Sampleoutput`.

## Notes on MSCOCO experiments
The resources required to reproduce results are kept in the directory `MSCOCO`. For training and testing, we used MSCOCO-2014 train images from `train2014.zip` and validation images from `val2014.zip`. These zipped archives are downloadable from MSCOCO website ([Link](http://cocodataset.org/#download)). Please find the exact list of images (with annotations) used for "training" in `MSCOCO/train_coco_seen_all.csv`. The lists of images used for "testing" different ZSL settings are:
* For traditional detection task: `MSCOCO/validation_coco_seen_all.csv`, 
* For zero-shot detection task: `MSCOCO/validation_coco_unseen_all.csv`, and 
* For generalized zero-shot detection task: `MSCOCO/validation_coco_unseen_seen_all_gzsd.csv`.

![ResultsSnapshot](https://salman-h-khan.github.io/images/Fig3_PL-ZSD.JPG) 
![Qualitative Results](https://salman-h-khan.github.io/images/Fig5_PL-ZSD.JPG) 
*Objects enclosed in red bounding boxes are the unseen objects.*

## Tests in the wild
We run the PL-ZSD model on two example videos from the [Youtube-8M](https://research.google.com/youtube8m/) dataset from Google AI. The demo videos contain several seen (e.g., pottend plant, person, hand-bag) and unseen classes (cat, train, suitcase). Note that we do not apply any pre/post processing procedure across temporal domain to smooth out the predictions. 

<!-- [![](http://img.youtube.com/vi/Qi5HfHatVXE/0.jpg)](http://www.youtube.com/watch?v=Qi5HfHatVXE "Demo Video (Cats)") 
[![](http://img.youtube.com/vi/UJFUqjEd3Rw/0.jpg)](http://www.youtube.com/watch?v=UJFUqjEd3Rw "Demo Video (Train station)") 
![Cat Gif](https://salman-h-khan.github.io/images/cat_demo.gif)
![Train Gif](https://salman-h-khan.github.io/images/train_demo.gif)
<img src="https://salman-h-khan.github.io/images/cat_demo.gif" width="350" />
-->

![Cat Gif](https://salman-h-khan.github.io/images/cat_demo.gif)

| [Link to Video 1](http://www.youtube.com/watch?v=Qi5HfHatVXE) | [Link to Video 2](http://www.youtube.com/watch?v=UJFUqjEd3Rw) |
|:-------------:|:-------------:|
| ![Cat Gif](https://salman-h-khan.github.io/images/cat_demo.gif) |  ![Train Gif](https://salman-h-khan.github.io/images/train_demo.gif)|


## Reference
[1] Lin, Tsung-Yi, Priyal Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. "Focal loss for dense object detection." IEEE transactions on pattern analysis and machine intelligence, 2018.

[2] Chua, Tat-Seng, et al. "NUS-WIDE: a real-world web image database from National University of Singapore." Proceedings of the ACM international conference on image and video retrieval. ACM, 2009.

## Citation
If you use this code, model and dataset splits for your research, please consider citing:
```
@article{rahman2018polarity,
title={Polarity Loss for Zero-shot Object Detection},
author={Rahman, Shafin and Khan, Salman and Barnes, Nick},
journal={arXiv preprint arXiv:1811.08982},
year={2018}}
```

## Acknowledgment
We thank the authors and contributors of original [RetinaNet implementation](https://github.com/fizyr/keras-retinanet). We also thank [Akshita Gupta](https://akshitac8.github.io) for testing the code.
