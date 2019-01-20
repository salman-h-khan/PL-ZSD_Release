Introduction
----------------
This code is the testing side implementation of the following work:
Shafin Rahman, Salman Khan, and Nick Barnes. "Polarity Loss for Zero-shot Object Detection." arXiv preprint arXiv:1811.08982 (2018).

Requirements
-------------------
Python 2.7
Keras 2.1.4
OpenCV 3.3.0
Tensorflow 1.3.0

Files
--------
sample_input.txt: a sample input file containing test image paths
gzsd.py: to perform generalized zero-shot detection task using sample_input.txt
keras_retinanet: directory containing the supporting code of the model. This directory is a modified version from original RetinaNet implementation[1] (https://github.com/fizyr/keras-retinanet)
Dataset: directory containing sample input and output images.
Config: directory containing configuration files to set up conda environment. 
Model: directory containing a pre-trained model using the polarity loss.
MSCOCO: This directory contains the source of data, proposed 65/15- seen/unseen split and experimental protocol used in experiments.
    cls_names_seen_coco.csv: list of 65 MSCOCO seen classes. Each line contains a class name followed by an index.
    cls_names_test_coco.csv: list of 80 MSCOCO object classes. Each line contains a class name followed by an index. Index 0 to 64 are from seen objects, and index 65 to 79 are from unseen.
    train_coco_seen_all.csv: training image paths and annotation used in the paper. Each line contains a training image path, a bounding box co-ordinate and the ground-truth class name of that bounding box. For example, Filepath,x1,y1,x2,y2,class_name
    validation_coco_seen_all.csv: test images with annotations for traditional object detection on only seen objects. File format, Filepath,x1,y1,x2,y2,class_name
    validation_coco_unseen_all.csv: test images with annotations for zero-shot object detection on only unseen objects. File format, Filepath,x1,y1,x2,y2,class_name
    validation_coco_unseen_seen_all_gzsd.csv: test images with annotations for generalized zero-shot object detection on both seen and unseen objects together. File format, Filepath,x1,y1,x2,y2,class_name
    word_w2v.txt, word_glo.txt, and word_ftx.txt: word2vec, GloVe and FastText word vectors of 80 classes of MSCOCO.  The ith column represents the 300-dimensional word vectors of the class name of the ith row of cls_names_test_coco.csv
    vocabulary_list: The list of vocabulary atoms from NUS-WIDE tag dataset[2] used in the paper.
    vocabulary_w2v.txt, vocabulary_glo.txt, and vocabulary_ftx.txt: word2vec, GloVe and FastText word vectors of all vocabulary tags.  The ith column represents the 300-dimensional word vectors of the class name of the ith row of vocabulary_list.txt

Running instruction
--------------------------
Download the trained model from the link: 
To run generalized zero-shot detection on sample input kept in Dataset/Sampleinput, simply run gzsd.py after installing all dependencies like Keras, Tensorflow, OpenCV. This code will generate the output files for each input image to Dataset/Sampleoutput.

Notes on MSCOCO experiments
-------------------------------------------
The resources required to reproduce results are kept in the directory MSCOCO. For training and testing, we used MSCOCO-2014 train images from train2014.zip and validation images from val2014.zip. Those zipped archives are downloadable from MSCOCO website (http://cocodataset.org/#download). Please find the exact list of images (with annotation) used during training from MSCOCO/train_coco_seen_all.csv, and during testing MSCOCO/validation_coco_seen_all.csv (for traditional detection task), MSCOCO/validation_coco_unseen_all.csv (for zero-shot detection task), and MSCOCO/validation_coco_unseen_seen_all_gzsd.csv (for generalized zero-shot detection task). 

# Can you mention the results table
# screenshots of the output (reference readme: https://github.com/jcjohnson/neural-style

Reference
---------------
[1] Lin, Tsung-Yi, Priyal Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. "Focal loss for dense object detection." IEEE transactions on pattern analysis and machine intelligence, 2018.

[2] Chua, Tat-Seng, et al. "NUS-WIDE: a real-world web image database from National University of Singapore." Proceedings of the ACM international conference on image and video retrieval. ACM, 2009.


Acknowledgment
------------------------
We thank the authors and contributors of original RetinaNet implementation (https://github.com/fizyr/keras-retinanet). We also thank Akshita Gupta (akshitac8.github.io).
