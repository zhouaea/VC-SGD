import csv
import math
import os
import time

import psutil

from neural_network import Neural_Network
import random
import numpy as np
import yaml
import heapq
from sklearn.utils import shuffle
from shapely.geometry import Point
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
from memory_profiler import profile

import mxnet as mx
from mxnet import nd, autograd, gluon
from memory_profiler import profile

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
BATCH_SIZE = cfg['neural_network']['batch_size']

# random.seed(cfg['seed'])
# np.random.seed(cfg['seed'])
counter = 0


class Vehicle:
    """
    Vehicle object for Car ML Simulator.
    Attributes:
    - car_id
    - x
    - y
    - speed
    - model
    - training_data_assigned
    - training_label_assigned
    - gradients
    """

    def __init__(self, car_id):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.net = None
        self.training_data_assigned = {}
        self.training_data = []
        self.training_label = []
        # self.training_label_assigned = []
        self.gradients = None
        # self.rsu_assigned = None

    def set_properties(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    @profile
    def download_model_from(self, central_server):
        self.net = central_server.net

    def handle_data(self):
        num_polygons = len(self.training_data_assigned)
        # Combine data from different polygons
        combined_data = []
        combined_label = []
        for data, label in self.training_data_assigned.values():
            combined_data.extend(data)
            combined_label.extend(label)
        # Shuffle
        combined_data, combined_label = shuffle(np.array(combined_data), np.array(combined_label))
        # Slice
        self.training_data = []
        self.training_label = []
        for i in range(num_polygons):
            self.training_data.append(nd.array(combined_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]))
            self.training_label.append(nd.array(combined_label[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]))

    def compute(self, simulation, closest_rsu, *args):
        psutil.cpu_percent()
        
        if cfg['dataset'] != 'pascalvoc':
            neural_net = Neural_Network()
        start = time.time()
        
        # Extra arguments are for pascalvoc dataset, to prevent intermediary lists from being used to save memory.
        if args is None:
            X = self.training_data.pop()
            y = self.training_label.pop()
        else:
            X = args[0]
            y = args[1]

        with autograd.record():
            if cfg['dataset'] == 'pascalvoc':
                print('gradient computation starting')
                # Passing input with * will calculate the loss instead of the model output.
                # Acquire all variables required to calculate loss.
                gt_bboxes = y[:, :, :4]
                gt_ids = y[:, :, 4:5]

                start_targets = time.time()

                objectness, center_targets, scale_targets, weights, class_targets = simulation.target_generator(
                    simulation.fake_x, simulation.feat_maps, simulation.anchors, simulation.offsets,
                    gt_bboxes, gt_ids, None)

                end_targets = time.time()
                print('targets calculated in ', end_targets - start_targets)

                # Calculate loss by using network in training mode and supplying extra target parameters.
                with autograd.train_mode():
                    obj_loss, center_loss, scale_loss, cls_loss = self.net(X, gt_bboxes, objectness,
                                                                           center_targets, scale_targets,
                                                                           weights, class_targets)
                loss = obj_loss + center_loss + scale_loss + cls_loss
                print('loss calculated')
            else:
                output = self.net(X)
                if cfg['attack'] == 'label' and len(closest_rsu.accumulative_gradients) < cfg['num_faulty_grads']:
                    loss = neural_net.loss(output, 9 - y)
                else:
                    loss = neural_net.loss(output, y)
        loss.backward()
        print('gradients computed')
        grad_collect = []
        for param in self.net.collect_params().values():
            if param.grad_req != 'null':
                grad_collect.append(param.grad().copy())

        self.gradients = grad_collect
        # print(self.gradients)
        # print(len(self.gradients))
        # for i in range(len(self.gradients)):
        #     print(len(self.gradients[i]))
        end = time.time()
        print('time to train on one batch:', end-start)
        print('CPU %:', psutil.cpu_percent())

        if cfg['write_runtime_statistics']:
            with open(os.path.join('collected_results', 'time_to_train_on_one_batch'), mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([end - start])

    def upload(self, simulation, closest_rsu):
        rsu = closest_rsu
        rsu.accumulative_gradients.append(self.gradients)
        # RSU checks if enough gradients collected
        if len(rsu.accumulative_gradients) >= cfg['simulation']['maximum_rsu_accumulative_gradients']:
            rsu.communicate_with_central_server(simulation.central_server)

    def compute_and_upload(self, simulation, closest_rsu):
        # We shuffle pascal voc data well enough that extra shuffling is not required for the dataset.
        if cfg['dataset'] == 'pascalvoc':
            # Avoid using intermediary lists to save memory.
            # Iterate through each individual batch
            for training_data, label_data in self.training_data_assigned.values():
                print('batch 1 start')
                self.compute(simulation, closest_rsu, training_data, label_data)
                self.upload(simulation, closest_rsu)
        else:
            # Shuffle the collected data
            self.handle_data()
            while self.training_data.values:
                self.compute(simulation, closest_rsu)
                self.upload(simulation, closest_rsu)

        self.training_data_assigned = {}

    # Return the RSU that is cloest to the vehicle
    def closest_rsu(self, rsu_list):
        shortest_distance = 99999999  # placeholder (a random large number)
        closest_rsu = None
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range and distance < shortest_distance:
                shortest_distance = distance
                closest_rsu = rsu
        return closest_rsu

    # Return a list of RSUs that is within the range of the vehicle
    # with each RSU being sorted from the closest to the furtherst
    def in_range_rsus(self, rsu_list):
        in_range_rsus = []
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                heapq.heappush(in_range_rsus, (distance, rsu))
        return [heapq.heappop(in_range_rsus)[1] for i in range(len(in_range_rsus))]

    # Return the index of the polygon the vehicle is currently in
    def in_polygon(self, polygons):
        for i, polygon in enumerate(polygons):
            if polygon.contains(Point(self.x, self.y)):
                return i
        raise Exception('Vehicle not in any polygon')