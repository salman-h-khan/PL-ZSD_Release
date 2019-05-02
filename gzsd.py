import keras
import cv2
from keras import backend as K
import numpy as np
import keras_resnet.models
from keras_retinanet.models import retinanet_vocab_w2v as retinanet # retinanet_vocab_glo
def format_img_size(img):
    """ formats the image size based on config """
    img_min_side = float(600)
    try:
        (height, width, _) = img.shape
    except AttributeError:
        pass
    #(height, width, _) = img.shape
    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img)
    img = format_img_channels(img)
    return img, ratio

num_seen = 65
word = np.loadtxt('MSCOCO/word_w2v.txt', dtype='float32', delimiter=',')
word_seen = word[:,:num_seen]
word_unseen = word[:,num_seen:]

wordname_lines = open('MSCOCO/cls_names_test_coco.csv').read().split("\n")
class_mapping = {}
for idx in range(int(len(wordname_lines)) - 1):
    class_mapping[idx] = wordname_lines[idx].split(',')[0]


inputs = keras.layers.Input(shape=(None, None, 3))
resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
model = retinanet.retinanet_bbox(inputs=inputs, num_classes=num_seen, backbone=resnet)
model.load_weights('Model/resnet50_polar_loss.h5')

lines = open('sample_input.txt').read().split("\n")
num_rois = 100
visualise = True # False True
detect_type = 'gzsd' # gzsd or zsd or seen_detection
seen_threshold = .4
unseen_threshold = .2

for idx in range(int(len(lines))-1):
    aline = lines[idx].split(" ")
    im_id = aline[1]
    filepath = aline[0]

    print('{}/{}'.format((idx+1), len(lines) - 1))

    img = cv2.imread(filepath)

    X, ratio = format_img(img)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # run network
    _, _, detections = model.predict_on_batch(X)

    # clip to image shape
    detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
    detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
    detections[:, :, 2] = np.minimum(X.shape[1], detections[:, :, 2])
    detections[:, :, 3] = np.minimum(X.shape[2], detections[:, :, 3])

    # correct boxes for image scale
    detections[0, :, :4] /= ratio

    # select scores from detections
    scores = detections[0, :, 4:]

    # select indices which have a score above the threshold
    indices_seen = np.where(scores > seen_threshold)


    T = 5
    mask = np.ones_like(scores,dtype='float32')
    mask[:, T:] = 0.0
    sorted_score = -np.sort(-scores, axis=1)
    sorted_score_arg = np.argsort(-scores, axis=1)
    sorted_score = np.multiply(sorted_score, mask)
    restroed_score = mask
    for i in range(scores.shape[0]):
        restroed_score[i, sorted_score_arg[i, :]] = sorted_score[i, :]

    unseen_pd = np.dot(restroed_score, np.transpose(word_seen))
    unseen_scores = np.dot(unseen_pd, word_unseen)
    val = np.max(unseen_scores, axis=1)
    val_arg = np.argmax(unseen_scores, axis=1)
    pos = np.where(val > unseen_threshold)

    indices_unseen = []
    indices_unseen.append(pos[0])
    indices_unseen.append(num_seen+val_arg[pos[0]])
    indices_unseen = tuple(indices_unseen)
    scores = np.concatenate((scores, unseen_scores), axis=1)

    # For Generalized Zero-shot Detection
    if detect_type == 'gzsd':
        indices = []
        indices.append(np.concatenate((indices_seen[0], indices_unseen[0])))
        indices.append(np.concatenate((indices_seen[1], indices_unseen[1])))
        indices = tuple(indices)

    # For ZSL only
    if detect_type == 'zsd':
        indices = indices_unseen

    # Only Traditional seen detection only
    if detect_type == 'seen_detection':
        indices = indices_seen


    # select those scores
    scores_ = scores[indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores_)[:num_rois]


    # select detections
    image_boxes = detections[0, indices[0][scores_sort], :4]
    image_scores = np.expand_dims(scores[indices[0][scores_sort], indices[1][scores_sort]], axis=1)
    image_detections = np.append(image_boxes, image_scores, axis=1)
    image_predicted_labels = indices[1][scores_sort]


    for i in range(0, image_predicted_labels.shape[0], 1):

        real_x1 = np.int(image_boxes[i, 0])
        real_y1 = np.int(image_boxes[i, 1])
        real_x2 = np.int(image_boxes[i, 2])
        real_y2 = np.int(image_boxes[i, 3])

        textLabel = '{}: {}'.format(class_mapping[image_predicted_labels[i]], int(100*image_scores[i]))

        if image_predicted_labels[i] < num_seen:
            print( '  seen--' + textLabel)
            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 255), 2)
        else:
            print( 'unseen--' + textLabel)
            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (255, 0, 255), 2)

        (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        textOrg = (real_x1, real_y1 - 0)

        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, .8, (0, 0, 0), 1)


    if visualise:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        
    cv2.imwrite('Dataset/Sampleoutput/{}'.format(filepath[20:]), img)