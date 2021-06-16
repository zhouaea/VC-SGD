import csv
import os
import random

import psutil
import yaml
import time
import numpy as np
import mxnet as mx
from mxnet.image import imresize as data_resize, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
from gluoncv import data, utils
from gluoncv.data.transforms.bbox import resize as label_bbox_resize
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data import VOCDetection
from collections import defaultdict
import copy
import gc
from sklearn.utils import shuffle
from memory_profiler import profile

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)


def transform(data, label):
    if cfg['dataset'] == 'pascalvoc':
        h, w, _ = data.shape
        data = data_resize(data, cfg['neural_network']['height'], cfg['neural_network']['width'])
        label = label_bbox_resize(label, in_size=(w, h),
                                  out_size=(cfg['neural_network']['width'], cfg['neural_network']['height']))
        label = label.astype(np.float32)
        # Pad to 56 objects, better to do it here than to pad with an ndarray according to documentation.
        np.pad(label, (0, 56 - len(label)), 'constant', constant_values=(-1, -1))
    if cfg['dataset'] == 'cifar10' or cfg['dataset'] == 'pascalvoc':
        data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 255

    return data, label


start = time.time()
# Load Data
BATCH_SIZE = cfg['neural_network']['batch_size']
NUM_TRAINING_DATA = cfg['num_training_data']
num_training_data = cfg['num_training_data']
if cfg['dataset'] == 'cifar10':
    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.CIFAR10('../data/cx2', train=True, transform=transform).take(num_training_data),
        batch_size=NUM_TRAINING_DATA, shuffle=True, last_batch='discard')
    val_train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.CIFAR10('../data/cx2', train=True, transform=transform).take(cfg['num_val_train_data']),
        batch_size=BATCH_SIZE, shuffle=False, last_batch='keep')
    val_test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.CIFAR10('../data/cx2', train=False, transform=transform).take(cfg['num_test_data']),
        batch_size=BATCH_SIZE, shuffle=False, last_batch='keep')
elif cfg['dataset'] == 'mnist':
    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=transform).take(num_training_data),
        batch_size=NUM_TRAINING_DATA, shuffle=True, last_batch='discard')
    val_train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=transform).take(cfg['num_val_train_data']),
        batch_size=BATCH_SIZE, shuffle=False, last_batch='keep')
    val_test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST('../data/mnist', train=False, transform=transform).take(cfg['num_test_data']),
        batch_size=BATCH_SIZE, shuffle=False, last_batch='keep')
elif cfg['dataset'] == 'pascalvoc':
    # Call psutil before and after the code you want to analyze.
    if cfg['write_cpu_and_memory']:
        psutil.cpu_percent()

    # Typically we use 2007+2012 trainval splits for training data.
    # NOTES: 
    #   - Originally the training and label data are numpy ndarrays but are converted to mxnet ndarrays
    #   when they are passed into the dataloader.
    #   - The dataloader should have batches of 1/4 the full dataset.
    #   - X has a shape of (batch size, 3, 320, 320) and is an mxnet ndarray.
    #   - y has a shape of (batch size, objects in image, 6) and is an mxnet ndarray.
    print('loading training dataset...')
    train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')],
                                 transform=transform)

    val_train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')],
                                     transform=transform)

    # and use 2007 test as validation data
    val_test_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'test')], transform=transform)

    if cfg['write_cpu_and_memory']:
        with open(os.path.join('collected_results', 'computer_resource_percentages'),
                  mode='a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([psutil.cpu_percent(), psutil.virtual_memory().percent])

    # behavior of batchify_fn: stack images, and pad labels
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    # Load PASCAL VOC datasets into dataloaders.
    # Note: See https://cv.gluon.ai/build/examples_detection/train_yolo_v3.html for explanation on batchify.

    print('loading dataloader...')
    train_data = mx.gluon.data.DataLoader(train_dataset.take(NUM_TRAINING_DATA), NUM_TRAINING_DATA / 4 + (NUM_TRAINING_DATA % 4 > 0), # round up if there is a decimal
                                          shuffle=True,
                                          batchify_fn=batchify_fn, last_batch='keep')

    val_train_data = mx.gluon.data.DataLoader(val_train_dataset.take(cfg['num_val_train_data']),
                                              cfg['test_and_val_train_batch_size'],
                                              shuffle=False,
                                              batchify_fn=batchify_fn, last_batch='keep')
    val_test_data = mx.gluon.data.DataLoader(val_test_dataset.take(cfg['num_test_data']),
                                             cfg['test_and_val_train_batch_size'],
                                             shuffle=False,
                                             batchify_fn=batchify_fn, last_batch='keep')

    # Clean up unused datasets
    del train_dataset
    del val_train_dataset
    del val_test_dataset
    del batchify_fn
    gc.collect()

if cfg['write_cpu_and_memory']:
    with open(os.path.join('collected_results', 'computer_resource_percentages'),
              mode='a') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([psutil.cpu_percent(), psutil.virtual_memory().percent])

if cfg['even_distribution']:
    pass
else:
    if cfg['dataset'] == 'pascalvoc':
        pass
    else:
        for (X, y) in train_data:
            # Data and labels are initially mxnet nd arrays, but the data becomes a list of numpy nd arrays and the label
            # becomes a list of numpy int32.
            X = list(X.asnumpy())
            y = list(y.asnumpy())
            # for random
            X_first_half = X[:int(len(X) / 2)]
            y_first_half = y[:int(len(y) / 2)]
            # for byclass
            X_second_half = X[int(len(X) / 2):]
            y_second_half = y[int(len(y) / 2):]

    # Partition second_half image and label data into different classes.
    if cfg['dataset'] == 'pascalvoc':
        pass
    else:
        train_data_byclass = defaultdict(list)
        # For every image in the second half of the dataset, add it to a list in the dictionary key that corresponds with
        # its class index.
        for j in range(len(X_second_half)):
            train_data_byclass[y_second_half[j]].append(X_second_half[j])

@profile
def data_for_polygon(polygons):
    """
        Returns training data and labels for new epochs.
    """
    NUM_POLYGONS = len(polygons)
    start = time.time()
    image_data_bypolygon = []
    label_data_bypolygon = []

    if cfg['write_cpu_and_memory']:
        psutil.cpu_percent()

    # Do not organize by classes, just divide entire dataset into tenths.
    if cfg['even_distribution']:
        # Create 10 separate lists to contain data for each polygon.
        image_data_bypolygon = [[] for i in range(NUM_POLYGONS)]
        label_data_bypolygon = [[] for i in range(NUM_POLYGONS)]

        lists_in_polygon = 0
        current_polygon = 0

        # In each quarter of the image and label data, put a tenth into each polygon's list.
        for i, (X_quarter, y_quarter) in enumerate(train_data):
            one_tenth_index = int(len(X_quarter) / NUM_POLYGONS) + (len(X_quarter) % NUM_POLYGONS > 0) # round up if there is a decimal
            for j in range(NUM_POLYGONS):
                if lists_in_polygon == 4:
                    print("polygon", current_polygon, "partitioned")
                    lists_in_polygon = 0
                    current_polygon += 1
                image_data_bypolygon[current_polygon].append(X_quarter[j * one_tenth_index:(j + 1) * one_tenth_index])
                label_data_bypolygon[current_polygon].append(y_quarter[j * one_tenth_index:(j + 1) * one_tenth_index])
                lists_in_polygon += 1
        gc.collect()

    else:
        if cfg['dataset'] == 'pascalvoc':
            image_data_bypolygon = [[] for i in range(NUM_POLYGONS)]
            label_data_bypolygon = [[] for i in range(NUM_POLYGONS)]

            # Create lists to store data by pascalvoc classes.
            image_data_byclass = [[] for i in range(20)]
            label_data_byclass = [[] for i in range(20)]

            lists_in_polygon = 0
            current_polygon = 0

            for i, (X_quarter, y_quarter) in enumerate(train_data):
                # Assign half of data ("random data") to polygons (two lists per polygon)
                if i < 2:
                    one_tenth_index = int(len(X_quarter) / NUM_POLYGONS) + (
                                len(X_quarter) % NUM_POLYGONS > 0)  # round up if there is a decimal
                    # Divide the quarters into tenths, add two tenths to every polygon
                    for j in range(NUM_POLYGONS):
                        if lists_in_polygon == 2:
                            print("polygon", current_polygon, "random half partitioned")
                            lists_in_polygon = 0
                            current_polygon += 1
                        image_data_bypolygon[current_polygon].append(
                            X_quarter[j * one_tenth_index:(j + 1) * one_tenth_index])
                        label_data_bypolygon[current_polygon].append(
                            y_quarter[j * one_tenth_index:(j + 1) * one_tenth_index])
                        lists_in_polygon += 1
                # Sort half of data by class, to assign to polygons later.
                else:
                    # Measure performance.
                    if i == 2:
                        start = time.time()
                        if cfg['write_cpu_and_memory']:
                            psutil.cpu_percent()

                    # Iterate through each individual element of the batch.
                    for j in range(len(X_quarter)):
                        # Each pascal voc label is an n x 6 ndarray. There are n objects in each datum and the 5th column
                        # has the class index for each object.
                        potential_class_indices = []
                        # Store all class indices found in the training data.
                        for k in range(len(y_quarter[j])):
                            # If there are no more valid objects go to the next image label.
                            if y_quarter[j][k][4] == -1:
                                break
                            potential_class_indices.append(int(y_quarter[j][k][4].asscalar()))
                        # Access the list cell corresponding to a random class index found in the training data.
                        random_class_index = np.random.randint(len(potential_class_indices))

                        # Add the datum to a randomly chosen object class that it contains.
                        # Note: We must add an extra dimension to each datum so we concatenate by batch later.
                        image_data_byclass[potential_class_indices[random_class_index]].append(nd.expand_dims(X_quarter[j], axis=0))
                        label_data_byclass[potential_class_indices[random_class_index]].append(nd.expand_dims(y_quarter[j], axis=0))

            # Measure performance of partitioning data into classes
            end = time.time()
            if cfg['write_cpu_and_memory']:
                with open(os.path.join('collected_results', 'computer_resource_percentages'),
                          mode='a') as f:
                    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([psutil.cpu_percent(), psutil.virtual_memory().percent])
            print('Time to partition half of training data into classes:', end - start)

            # Print statistics on amount of data in each class.
            if cfg['dataset'] == 'pascalvoc':
                for i, class_data in enumerate(image_data_byclass):
                    print(i, len(class_data), end=" | ")
            print()

            # Add data from two random classes to each polygon (in the form of two lists).
            random_class_indices = [i for i in range(20)]
            random.shuffle(random_class_indices)

            lists_in_polygon = 0
            current_polygon = 0

            for i in random_class_indices:
                if lists_in_polygon == 2:
                    print("polygon", current_polygon, "class half partitioned")
                    lists_in_polygon = 0
                    current_polygon += 1
                # Note that it is important that the polygon lists are ndarrays. Thus, we
                # must convert our list of ndarrays into an ndarray of ndarrays.
                if len(image_data_byclass[i]) != 0:
                    temp_nd_image_data = nd.concat(*image_data_byclass[i], num_args=len(image_data_byclass[i]), dim=0)
                    temp_nd_label_data = nd.concat(*label_data_byclass[i], num_args=len(image_data_byclass[i]), dim=0)
                    print(temp_nd_image_data.shape)
                    print(temp_nd_label_data.shape)
                    image_data_bypolygon[current_polygon].append(temp_nd_image_data)
                    label_data_bypolygon[current_polygon].append(temp_nd_label_data)
                lists_in_polygon += 1

        else:
            random_len = len(X_first_half) // len(polygons) + 1

            for i in range(len(polygons)):
                X_ = X_first_half[i * random_len:(i + 1) * random_len]
                y_ = y_first_half[i * random_len:(i + 1) * random_len]
                X_new = copy.deepcopy(X_)
                y_new = copy.deepcopy(y_)
                train_data_list = list(train_data_byclass.values())
                X_new.extend(train_data_list[i])
                y_new.extend([list(train_data_byclass.keys())[i] for _ in range(len(train_data_list[i]))])
                X_new, y_new = shuffle(np.array(X_new), np.array(y_new))
                image_data_bypolygon.append(X_new.tolist())
                label_data_bypolygon.append(y_new.tolist())


    end = time.time()
    if cfg['write_cpu_and_memory']:
        with open(os.path.join('collected_results', 'computer_resource_percentages'),
                  mode='a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([psutil.cpu_percent(), psutil.virtual_memory().percent])
    print('Time to partition all training data into polygons:', end - start)

    print("polygons", len(image_data_bypolygon))
    print("lists per polygon", len(image_data_bypolygon[0]))
    print("image data per list", len(image_data_bypolygon[0][0]))
    print("dimension of image", len(image_data_bypolygon[0][0][0]))
    exit()
    return image_data_bypolygon, label_data_bypolygon
