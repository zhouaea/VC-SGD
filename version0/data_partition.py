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
    if cfg['dataset'] == 'cifar10' or cfg['dataset'] == 'mnist':
        if cfg['dataset'] == 'cifar10':
            data = mx.nd.transpose(data, (2,0,1))
        data = data.astype(np.float32) / 255

    if cfg['dataset'] == 'pascalvoc':
        data = mx.image.resize_short_within(data, short=416, max_size=1024, mult_base=1)
        data = mx.nd.image.to_tensor(data)
        data = mx.nd.image.normalize(data, mean=0, std=1)
        data = data.astype(np.float32)
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
    val_train_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'trainval'), (2012, 'trainval')],
                                     transform=transform)
    # and use 2007 test as validation data
    val_test_dataset = VOCDetection(root='../data/pascalvoc', splits=[(2007, 'test')], transform=transform)

    # behavior of batchify_fn: stack images, and pad labels
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    train_data = mx.gluon.data.DataLoader(train_dataset.take(num_training_data), BATCH_SIZE, shuffle=True,
                            batchify_fn=batchify_fn, last_batch='discard')
    val_train_data = mx.gluon.data.DataLoader(val_train_dataset, BATCH_SIZE, shuffle=False,
                                batchify_fn=batchify_fn, last_batch='keep')
    val_test_data = mx.gluon.data.DataLoader(val_test_dataset, BATCH_SIZE, shuffle=False,
                               batchify_fn=batchify_fn, last_batch='keep')

for i in train_data:
    X, y = i
    X = list(X.asnumpy())
    y = list(y.asnumpy())
    # for random
    X_first_half = X[:int(len(X)/2)]
    y_first_half = y[:int(len(y)/2)]
    # for byclass
    X_second_half = X[int(len(X)/2):]
    y_second_half = y[int(len(y)/2):]

# Put second_half data into different classes
train_data_byclass = defaultdict(list)
for j in range(len(X_second_half)):
    train_data_byclass[y_second_half[j]].append(X_second_half[j])

# for i in train_data_byclass.values():
#     print(len(i))


def data_for_polygon(polygons):
    """
        Returns training data and labels for new epochs.
    """
    training_data_byclass = []
    training_label_byclass = []
    random_len = len(X_first_half) // len(polygons) + 1
    
    for i in range(len(polygons)):
        X_ = X_first_half[i*random_len:(i+1)*random_len]
        y_ = y_first_half[i*random_len:(i+1)*random_len]
        X_new = copy.deepcopy(X_)
        y_new = copy.deepcopy(y_)
        train_data_list = list(train_data_byclass.values())
        X_new.extend(train_data_list[i])
        y_new.extend([list(train_data_byclass.keys())[i] for _ in range(len(train_data_list[i]))])
        X_new, y_new = shuffle(np.array(X_new), np.array(y_new))
        training_data_byclass.append(X_new.tolist())
        training_label_byclass.append(y_new.tolist())

    return training_data_byclass, training_label_byclass