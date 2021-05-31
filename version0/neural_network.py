import math
import gluoncv as gcv
import heapq
import random
from collections import deque
import numpy as np
import yaml
import xml.etree.ElementTree as ET 
from mxnet import nd, autograd, gluon


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

# random.seed(cfg['seed'])
# np.random.seed(cfg['seed'])

class Neural_Network:
    """
    Neural network functions
    Attributes:
    - optimizer
    """
    def __init__(self):
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['neural_network']['learning_rate'])
        pass

    # The loss function
    def loss(self, output, label):
        if cfg['dataset'] == 'pascalvoc':
            # We don't use a standalone loss function for object detection since the yolo network can calculate loss
            # by itself in recording + training mode.
            pass
        else:
            loss_object = gluon.loss.SoftmaxCrossEntropyLoss()

        loss = loss_object(output, label)
        return loss