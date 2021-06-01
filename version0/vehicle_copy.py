import math
from neural_network import Neural_Network
import random
import numpy as np
import yaml
import heapq
from sklearn.utils import shuffle
from shapely.geometry import Point
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator

import mxnet as mx
from mxnet import nd, autograd, gluon


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
BATCH_SIZE = cfg['neural_network']['batch_size']

# random.seed(cfg['seed'])
# np.random.seed(cfg['seed'])


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
        if cfg['dataset'] == 'pascalvoc':
            self.target_generator = None
            self.fake_x = None
            self.anchors = None
            self.offsets = None
            self.feat_maps = None
        # self.rsu_assigned = None

    def set_properties(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def download_model_from(self, central_server):
        self.net = central_server.net
        if cfg['dataset'] == 'pascalvoc':
            self.target_generator = YOLOV3PrefetchTargetGenerator(
                num_class=len(self.net.classes))
            self.fake_x = mx.nd.zeros((cfg['neural_network']['batch_size'], 3, 416, 416))
            with autograd.train_mode():
                _, self.anchors, self.offsets, self.feat_maps, _, _, _, _ = self.net(self.fake_x)

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
            self.training_data.append(nd.array(combined_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]))
            self.training_label.append(nd.array(combined_label[i*BATCH_SIZE:(i+1)*BATCH_SIZE]))

    def compute(self, simulation, closest_rsu):
        neural_net = Neural_Network()
        X = self.training_data.pop()
        y = self.training_label.pop()
        print(X)
        print(y)

        with autograd.record():
            if cfg['dataset'] == 'pascalvoc':
                # Passing input with * will calculate the loss instead of the model output.
                # Acquire all variables required to calculate loss.
                gt_bboxes = mx.nd.array(y[:, :, :4]).astype(np.float32)
                gt_ids = mx.nd.array(y[np.newaxis, :, 4:5])

                objectness, center_targets, scale_targets, weights, class_targets = self.target_generator(
                    self.fake_x, self.feat_maps, self.anchors, self.offsets,
                    gt_bboxes, gt_ids, None)

                # Calculate loss by using network in training mode and supplying extra target parameters.
                with autograd.train_mode():
                    obj_loss, center_loss, scale_loss, cls_loss = self.net(X, gt_bboxes, objectness,
                                                                                          center_targets, scale_targets,
                                                                                          weights, class_targets)
                loss = obj_loss.mean().asscalar() + center_loss.mean().asscalar() + scale_loss.mean().asscalar() + cls_loss.mean().asscalar()
            else:
                output = self.net(X)
                if cfg['attack'] == 'label' and len(closest_rsu.accumulative_gradients) < cfg['num_faulty_grads']:
                    loss = neural_net.loss(output, 9 - y)
                else:
                    loss = neural_net.loss(output, y)
        loss.backward()

        grad_collect = []
        for param in self.net.collect_params().values():
            if param.grad_req != 'null':
                grad_collect.append(param.grad().copy())

        print('gradients collected:', grad_collect)
        self.gradients = grad_collect
        # print(self.gradients)
        # print(len(self.gradients))
        # for i in range(len(self.gradients)):
        #     print(len(self.gradients[i]))
        print("gradient computed")

    def upload(self, simulation, closest_rsu):
        rsu = closest_rsu
        rsu.accumulative_gradients.append(self.gradients)
        # RSU checks if enough gradients collected
        if len(rsu.accumulative_gradients) >= cfg['simulation']['maximum_rsu_accumulative_gradients']:
            rsu.communicate_with_central_server(simulation.central_server)

    def compute_and_upload(self, simulation, closest_rsu):
        self.handle_data()
        while self.training_data:
            self.compute(simulation, closest_rsu)
            self.upload(simulation, closest_rsu)
        self.training_data_assigned = {}


    
    # Return the RSU that is cloest to the vehicle
    def closest_rsu(self, rsu_list):
        shortest_distance = 99999999 # placeholder (a random large number)
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