import yaml
import numpy as np
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
from gluoncv import data, utils
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data import VOCDetection
from collections import defaultdict
import copy
from sklearn.utils import shuffle

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)


def transform(data, label):
    if cfg['dataset'] == 'pascalvoc':
        data = mx.image.imresize(data, 416, 416)
    if cfg['dataset'] == 'cifar10' or cfg['dataset'] == 'pascalvoc':
        data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32) / 255

    return data, label

# Load Data
BATCH_SIZE = cfg['neural_network']['batch_size']
NUM_TRAINING_DATA = cfg['num_training_data']
num_training_data = cfg['num_training_data']
if cfg['dataset'] == 'cifar10':
    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('../data/cx2', train=True, transform=transform).take(num_training_data),
                                batch_size=NUM_TRAINING_DATA, shuffle=True, last_batch='discard')
    val_train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('../data/cx2', train=True, transform=transform).take(cfg['num_val_loss']),
                                batch_size=BATCH_SIZE, shuffle=False, last_batch='keep')
    val_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('../data/cx2', train=False, transform=transform),
                                batch_size=BATCH_SIZE, shuffle=False, last_batch='keep')
elif cfg['dataset'] == 'mnist':
    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=transform).take(num_training_data),
                                batch_size=NUM_TRAINING_DATA, shuffle=True, last_batch='discard')
    val_train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=transform).take(cfg['num_val_loss']),
                                batch_size=BATCH_SIZE, shuffle=False, last_batch='keep')
    val_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=False, transform=transform),
                                batch_size=BATCH_SIZE, shuffle=False, last_batch='keep')
elif cfg['dataset'] == 'pascalvoc':
    # typically we use 2007+2012 trainval splits for training data
    train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')],
                                 transform=transform)
    print("Length of training dataset: " + str(len(train_dataset)))
    val_train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')],
                                     transform=transform)
    # and use 2007 test as validation data
    val_test_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'test')], transform=transform)

    # behavior of batchify_fn: stack images, and pad labels
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    # Load PASCAL VOC datasets into dataloaders.
    # Note: See https://cv.gluon.ai/build/examples_detection/train_yolo_v3.html for explanation on batchify.
    train_data = mx.gluon.data.DataLoader(train_dataset.take(NUM_TRAINING_DATA), NUM_TRAINING_DATA, shuffle=False,
                            batchify_fn=batchify_fn, last_batch='discard')
    print("Batches of training dataloader: " + str(len(train_data)))
    val_train_data = mx.gluon.data.DataLoader(val_train_dataset.take(10), BATCH_SIZE, shuffle=False,
                                batchify_fn=batchify_fn, last_batch='keep')
    val_test_data = mx.gluon.data.DataLoader(val_test_dataset.take(10), BATCH_SIZE, shuffle=False,
                               batchify_fn=batchify_fn, last_batch='keep')

# There is too much data in the labels and images of pascalvoc to create a tensor in (N, data, label) format.
# In this case the dimensions are (1, 16551, 16551). I am going to split training data into 8 batches but still 
# split the batches into two halves.

if cfg['dataset'] == 'pascalvoc':
    X_first_half = []
    y_first_half = []
    X_second_half = []
    y_second_half = []

    # # TODO This doesn't really split data into close to equal halves
    # for counter, batch in enumerate(train_data):
    #     X, y = batch
    #     if counter < str(len(train_data) / 2:
    #         X_first_half.append(list(X.asnumpy()))
    #         y_first_half.append(list(y.asnumpy()))
    #     else:
    #         X_second_half.append(list(X.asnumpy()))
    #         y_second_half.append(list(y.asnumpy()))

    # Determine shape of training data and what is inside of it.
    

    # For partitioning small subset of data.
    for (X, y) in train_data:
        # Data and labels are initially mxnet nd arrays, but the data becomes a list of numpy nd arrays and the label
        # becomes a list of numpy nd arrays.
        X = list(X.asnumpy())
        y = list(y.asnumpy())
        # for random
        X_first_half = X[:int(len(X)/2)]
        y_first_half = y[:int(len(y)/2)]
        # for byclass
        X_second_half = X[int(len(X)/2):]
        y_second_half = y[int(len(y)/2):]
else:
    for (X, y) in train_data:
        # Data and labels are initially mxnet nd arrays, but the data becomes a list of numpy nd arrays and the label
        # becomes a list of numpy int32.
        X = list(X.asnumpy())
        y = list(y.asnumpy())
        # for random
        X_first_half = X[:int(len(X)/2)]
        y_first_half = y[:int(len(y)/2)]
        # for byclass
        X_second_half = X[int(len(X)/2):]
        y_second_half = y[int(len(y)/2):]

print("Length of X:", len(X))
print("Length of y:", len(y))
print("length of X_first_half", len(X_first_half))
print("length of y_first_half", len(y_first_half))
print("length of X_second_half", len(X_second_half))
print("length of y_second_half", len(y_second_half))

# Partition second_half image and label data into different classes.
if cfg['dataset'] == 'pascalvoc':
    train_data_byclass = defaultdict(lambda: ([], []))
    # For every image and label in the second half of the dataset, add them to lists in the dictionary keys that correspond with
    # each class index of each object in the datum. There will be duplicate images in the dictionary if an image has multiple objects.
    for j in range(len(X_second_half)):
        # Each pascal voc label is an n x 6 ndarray. There are n objects in each datum and the 5th column
        # has the class index for each object.
        for k in range(len(y_second_half[j])):
            # If there are no more valid objects go to the next image label.
            if y_second_half[j][k][4] == -1:
                break
            # Access the key of the dict correlating to a class index, and the corresponding value contains a tuple with two lists.
            train_data_byclass[y_second_half[j][k][4]][0].append(X_second_half[j])
            train_data_byclass[y_second_half[j][k][4]][1].append(y_second_half[j])
else:
    train_data_byclass = defaultdict(list)
    # For every image in the second half of the dataset, add it to a list in the dictionary key that corresponds with
    # its class index.
    for j in range(len(X_second_half)):
        train_data_byclass[y_second_half[j]].append(X_second_half[j])

data_per_class = {}
for object_class, (data, label) in sorted(train_data_byclass.items()):
    print(object_class, len(data), len(label), end=" | ")
print("Number of classes: ", len(train_data_byclass.values()))

def data_for_polygon(polygons):
    """
        Returns training data and labels for new epochs.
    """
    train_data_bypolygon = []
    train_label_bypolygon = []
    class_index = 0
    if cfg['dataset'] == 'pascalvoc':
        random_len = len(X_first_half) // len(polygons)
    else:
        random_len = len(X_first_half) // len(polygons) + 1
    print('random_len: ', random_len)

    # Create a list of list of images where each list of images corresponds to a class index. The list index
    # will correspond to the order in which the keys were added to the dictionary.
    # Ex 1: For MNIST, train_data_list[0][0] would access the first image (1x28x28 numpy.ndarray) that depicts whichever
    # key was added first into train_data_byclass.
    # Ex 2: For pascalvoc, train_data_list[0][0] would access the first image (3x416x416 numpy.ndarray) that depicts
    # whichever key was added first into train_databyclass.
    # train_data_list = list(train_data_byclass.values())
    train_data_list = [v for (k,v) in sorted(train_data_byclass.items())]

    for i in range(len(polygons)):
        # Take a 10th (if there are 10 polygons) of the non-partitioned randomly shuffled data and labels.
        X_ = X_first_half[i*random_len:(i+1)*random_len]
        y_ = y_first_half[i*random_len:(i+1)*random_len]
        X_new = copy.deepcopy(X_)
        y_new = copy.deepcopy(y_)

        print('X_ shape:', np.array(X_).shape)
        print('y_ shape:', np.array(y_).shape)

        temp_train_data_byclass = []
        temp_label_data_byclass = []

        # Add images and labels of a single class or multiple classes (if there are more classes than polygons, like in pascal voc)
        # into temporary lists.
        for j in range(len(train_data_list) // len(polygons)):
            temp_train_data_byclass.extend(train_data_list[class_index][0])
            print('class', i * 2 + j, 'added', np.array(train_data_list[class_index][0]).shape)
            temp_label_data_byclass.extend(train_data_list[class_index][1])
            class_index += 1

        # If the number of classes does not divide evenly among polygons, add the images and labels corresponding to the
        # remaining classes to the last polygon's training and label data.
        if i == len(polygons) - 1:
            while class_index < len(train_data_list) - 1:
                print("extra class protocol triggered:", 'i =', i, "len(polygons) - 1 = ", len(polygons) - 1, 'class_index =', class_index, 'len(train_data) - 1 =', len(train_data_list) - 1)
                temp_train_data_byclass.extend(train_data_list[class_index][0])
                temp_label_data_byclass.extend(train_data_list[class_index][1])
                class_index += 1

        print('class index:', class_index)

        # Add temporary list to a list that corresponds with data in a single polygon.
        X_new.extend(temp_train_data_byclass)
        # Add temporary list to a list that corresponds with labels in a single polygon
        y_new.extend(temp_label_data_byclass)

        print('X_new shape:', np.array(X_new).shape)
        print('y_new shape:', np.array(y_new).shape)
        print()

        # Change X_new and y_new into numpy arrays instead of simple lists and have their contents shuffled.
        X_new, y_new = shuffle(np.array(X_new), np.array(y_new))

        # Each index of train_data_bypolygon and train_data_bypolygon correspond to a polygon.
        train_data_bypolygon.append(X_new.tolist())
        train_label_bypolygon.append(y_new.tolist())

    print('train_data_bypolygon shape:', np.array(train_data_bypolygon, dtype=object).shape)
    print('train_label_bypolygon shape:', np.array(train_label_bypolygon, dtype=object).shape)
    return train_data_bypolygon, train_label_bypolygon
