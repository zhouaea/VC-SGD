import random

import yaml
import time
import numpy as np
import mxnet as mx
from mxnet.image import imresize as data_resize, nd
from gluoncv.data.transforms.bbox import resize as label_bbox_resize
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data import VOCDetection
from collections import defaultdict
import copy
import gc
from sklearn.utils import shuffle

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
FILTER_TO_ONE_CLASS = cfg['filter']['filter_to_one_class']
SELECTED_CLASS_INDEX = cfg['filter']['class_index']


def transform(data, label):
    if cfg['dataset'] == 'pascalvoc':
        h, w, _ = data.shape
        data = data_resize(data, cfg['pascalvoc_metrics']['height'], cfg['pascalvoc_metrics']['width'])
        label = label_bbox_resize(label, in_size=(w, h),
                                  out_size=(cfg['pascalvoc_metrics']['width'], cfg['pascalvoc_metrics']['height']))
        label = label.astype(np.float32)

        if FILTER_TO_ONE_CLASS:
            selected_class_object_rows = []

            # If an object in the label is of the desired class, store it in a list.
            for object_row in label:
                if object_row[4] == SELECTED_CLASS_INDEX:
                    selected_class_object_rows.append(object_row)

            # Replace label rows only with rows that are for a specific object class.
            labelRowIndex = 0;
            for object_row in selected_class_object_rows:
                label[labelRowIndex] = object_row
                labelRowIndex += 1

            # Delete all other rows.
            while labelRowIndex < len(label):
                mx.np.delete(label, labelRowIndex, 0)

        # Pad to 56 objects, better to do it here than to pad with an mxnet ndarray according to the documentation.
        label = np.pad(label, ((0, 56 - len(label)), (0, 0)), 'constant', constant_values=-1)
    if cfg['dataset'] == 'cifar10' or cfg['dataset'] == 'pascalvoc':
        data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 255

    return data, label

def filter_to_one_class(sample):
    """Given a dataset and a class number, return a new dataset that only has image/label pairs with objects of that class number"""
    # Iterate through each individual object in the label.
    _, sample_y = sample
    for n in sample_y:
        if n[4] == SELECTED_CLASS_INDEX:
            return True

        # No more objects in label.
        if n[4] == -1:
            break

    return False

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
    # Typically we use 2007+2012 trainval splits for training data.
    # NOTES: 
    #   - Originally the training data is an mxnet ndarray and the label data is a numpy ndarray but is converted to mxnet ndarrays
    #   when it is passed into the dataloader.
    #   - The dataloader should have batches of 1/4 the full dataset.
    #   - X has a shape of (batch size, 3, 320, 320) and is an mxnet ndarray.
    #   - y has a shape of (batch size, objects in image, 6) and is an mxnet ndarray.
    print('loading training and testing datasets...')
    if FILTER_TO_ONE_CLASS:
        train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')]).filter(filter_to_one_class).transform(transform)

        val_train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')]).filter(filter_to_one_class).transform(transform)

        # and use 2007 test as validation data
        val_test_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'test')]).filter(filter_to_one_class).transform(transform)

        print(type(train_dataset))
        print(len(train_dataset))
        print(type(val_train_dataset))
        print(len(val_train_dataset))
        print(type(val_test_dataset))
        print(len(val_test_dataset))
    else:
        train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')],
                                     transform=transform)

        val_train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')], transform=transform)

        # and use 2007 test as validation data
        val_test_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'test')], transform=transform)

        print(type(train_dataset))
        print(len(train_dataset))
        print(type(val_train_dataset))
        print(len(val_train_dataset))
        print(type(val_test_dataset))
        print(len(val_test_dataset))

    # behavior of batchify_fn: stack images, and pad labels
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    # Load PASCAL VOC datasets into dataloaders.
    # Note: See https://cv.gluon.ai/build/examples_detection/train_yolo_v3.html for explanation on batchify.

    print('loading dataloader...')
    start = time.time()
    train_data = mx.gluon.data.DataLoader(train_dataset.take(NUM_TRAINING_DATA), int(NUM_TRAINING_DATA / 4) + (NUM_TRAINING_DATA % 4 > 0), # round up if there is a decimal
                                          shuffle=True,
                                          batchify_fn=batchify_fn, last_batch='keep', num_workers=20)

    val_train_data = mx.gluon.data.DataLoader(val_train_dataset.take(cfg['num_val_train_data']),
                                              BATCH_SIZE,
                                              shuffle=False,
                                              batchify_fn=batchify_fn, last_batch='keep', num_workers=20)
    val_test_data = mx.gluon.data.DataLoader(val_test_dataset.take(cfg['num_test_data']),
                                             BATCH_SIZE,
                                             shuffle=False,
                                             batchify_fn=batchify_fn, last_batch='keep', num_workers=20)
    end = time.time()
    # After switching from pascalvoc_dataset objects to lazy_transform_dataset objects, data loading takes longer with less of a bottleneck for target generation.
    print("time to initialize dataloaders:", end - start)

    # Clean up unused datasets
    del train_dataset
    del val_test_dataset
    del batchify_fn
    gc.collect()

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
            train_data_byclass[
                y_second_half[j]].append(X_second_half[j])


def data_for_polygon(polygons):
    start = time.time()
    """
        Returns training data and labels for new epochs.
    """
    NUM_POLYGONS = len(polygons)
    image_data_bypolygon = []
    label_data_bypolygon = []

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
                    lists_in_polygon = 0
                    current_polygon += 1

                # We can't concatenate an empty array with an ndarray, so for the first list added to the polygon bucket, just assign it.
                if lists_in_polygon == 0:
                    image_data_bypolygon[current_polygon] = X_quarter[j * one_tenth_index:(j + 1) * one_tenth_index]
                    label_data_bypolygon[current_polygon] = y_quarter[j * one_tenth_index:(j + 1) * one_tenth_index]
                else:
                    image_data_bypolygon[current_polygon] = nd.concat(image_data_bypolygon[current_polygon], X_quarter[j * one_tenth_index:(j + 1) * one_tenth_index], num_args=2, dim=0)
                    label_data_bypolygon[current_polygon] = nd.concat(label_data_bypolygon[current_polygon], y_quarter[j * one_tenth_index:(j + 1) * one_tenth_index], num_args=2, dim=0)

                lists_in_polygon += 1

    else:
        if cfg['dataset'] == 'pascalvoc':
            image_data_bypolygon = [[] for i in range(NUM_POLYGONS)]
            label_data_bypolygon = [[] for i in range(NUM_POLYGONS)]

            # Create lists to store data by pascalvoc classes.
            image_data_byclass = [[] for i in range(20)]
            label_data_byclass = [[] for i in range(20)]

            lists_in_polygon = 0
            current_polygon = 0

            # Out of the 4 quarters of total training data, how many are sent directly to polygons and how many are filtered into classes?
            if cfg['dataset_random_distribution'] == 0.5:
                random_data_proportion_index = 2
            elif cfg['dataset_random_distribution'] == 0.25:
                random_data_proportion_index = 1
            else:
                print("ERROR: Please select either 0.5 or 0.25 for dataset_random_distribution in config.yml")
                exit()

            # Load the training data in quarters. (Each batch of training data is a quarter of the full dataset)
            for i, (X_quarter, y_quarter) in enumerate(train_data):
                # Assign one quarter or two quarters of total data ("random data, since the dataloader is shuffled") to polygons (two lists per polygon)
                if i < random_data_proportion_index:
                    one_tenth_index = int(len(X_quarter) / NUM_POLYGONS) + (
                                len(X_quarter) % NUM_POLYGONS > 0)  # round up if there is a decimal
                    # Divide the quarters into tenths, add two tenths to every polygon
                    for j in range(NUM_POLYGONS):
                        if lists_in_polygon == 2:
                            lists_in_polygon = 0
                            current_polygon += 1
                        # If the end slice point would be greater than the length of the batch,
                        # take slightly less than a tenth.
                        if (j+1) * one_tenth_index > len(X_quarter):
                            image_data_bypolygon[current_polygon].append(
                                X_quarter[j * one_tenth_index:])
                            label_data_bypolygon[current_polygon].append(
                                y_quarter[j * one_tenth_index:])
                        else:
                            image_data_bypolygon[current_polygon].append(
                                X_quarter[j * one_tenth_index:(j + 1) * one_tenth_index])
                            label_data_bypolygon[current_polygon].append(
                                y_quarter[j * one_tenth_index:(j + 1) * one_tenth_index])
                        lists_in_polygon += 1
                # Sort half of data by class, to assign to polygons later.
                else:
                    # Iterate through each individual element of the batch.
                    for j in range(len(X_quarter)):
                        # Each pascal voc label is an n x 6 ndarray. There are n objects in each datum and the 5th column
                        # has the class index for each object.
                        potential_class_indices = []
                        # Store all class indices found in the single training datum.
                        for k in range(len(y_quarter[j])):
                            # If there are no more valid objects go to the next image label.
                            if y_quarter[j][k][4] == -1:
                                break
                            potential_class_indices.append(int(y_quarter[j][k][4].asscalar()))

                        # Add the datum to a randomly chosen object class that it contains.
                        random_class_index = np.random.randint(len(potential_class_indices))
                        # Note: We must add an extra dimension to each datum so we can concatenate by batch later.
                        image_data_byclass[potential_class_indices[random_class_index]].append(nd.expand_dims(X_quarter[j], axis=0))
                        label_data_byclass[potential_class_indices[random_class_index]].append(nd.expand_dims(y_quarter[j], axis=0))

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
                    lists_in_polygon = 0
                    current_polygon += 1
                # Note that it is important that the polygon lists are ndarrays. Thus, we
                # must convert our list of ndarrays into an ndarray of ndarrays.
                if len(image_data_byclass[i]) != 0:
                    image_data_bypolygon[current_polygon].append(nd.concat(*image_data_byclass[i], num_args=len(image_data_byclass[i]), dim=0))
                    label_data_bypolygon[current_polygon].append(nd.concat(*label_data_byclass[i], num_args=len(label_data_byclass[i]), dim=0))
                lists_in_polygon += 1

            # Combine lists in each polygon into one giant list and shuffle each polygon list.
            for pi in range(len(polygons)):
                temp_image_data_forpolygon = nd.concat(*image_data_bypolygon[pi], num_args=len(image_data_bypolygon[pi]), dim=0)
                temp_label_data_forpolygon = nd.concat(*label_data_bypolygon[pi], num_args=len(label_data_bypolygon[pi]), dim=0)

                # We need to shuffle both image and label data in the same way, thus this complicated approach.
                random_data_indices = [i for i in range(len(temp_image_data_forpolygon))]
                random.shuffle(random_data_indices)

                image_data_bypolygon[pi] = temp_image_data_forpolygon[random_data_indices[0]: random_data_indices[0] + 1]
                label_data_bypolygon[pi] = temp_label_data_forpolygon[random_data_indices[0]: random_data_indices[0] + 1]
                for i, ri in enumerate(random_data_indices):
                    if i == 0:
                        continue
                    image_data_bypolygon[pi] = nd.concat(image_data_bypolygon[pi], temp_image_data_forpolygon[ri: ri + 1],
                                                             num_args=2, dim=0)
                    label_data_bypolygon[pi] = nd.concat(label_data_bypolygon[pi], temp_label_data_forpolygon[ri: ri + 1],
                                                             num_args=2, dim=0)

            # Clean up temporary variables
            del temp_image_data_forpolygon
            del temp_label_data_forpolygon
            del random_data_indices
            gc.collect()

        else:
            # Data partitioning for image classification.
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
    print("data partitioning took ", end - start, "seconds")
    return image_data_bypolygon, label_data_bypolygon