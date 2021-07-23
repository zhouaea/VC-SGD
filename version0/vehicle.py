import csv
import math
import os
import time

import psutil
import gc

from neural_network import Neural_Network
import numpy as np
import yaml
import heapq
from sklearn.utils import shuffle
from shapely.geometry import Point

from mxnet import nd, autograd

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
        start = time.time()
        if cfg['dataset'] != 'pascalvoc':
            neural_net = Neural_Network()
        
        # Extra arguments are for pascalvoc dataset, to prevent intermediary lists from being used to save memory.
        if len(args) == 0:
            X = self.training_data.pop()
            y = self.training_label.pop()
        else:
            X = args[0]
            y = args[1]

        with autograd.record():
            if cfg['dataset'] == 'pascalvoc':
                gt_bboxes = y[:, :, :4]
                gt_ids = y[:, :, 4:5]

                start_targets = time.time()

                objectness, center_targets, scale_targets, weights, class_targets = simulation.target_generator(
                    simulation.fake_x, simulation.feat_maps, simulation.anchors, simulation.offsets,
                    gt_bboxes, gt_ids, None)

                end_targets = time.time()

                # Calculate loss by using network in training mode and supplying extra target parameters.
                with autograd.train_mode():
                    obj_loss, center_loss, scale_loss, cls_loss = self.net(X, gt_bboxes, objectness,
                                                                           center_targets, scale_targets,
                                                                           weights, class_targets)
                loss = obj_loss + center_loss + scale_loss + cls_loss
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

        self.gradients = grad_collect
        end = time.time()
        print('time to compute gradients', end - start)
        with open(os.path.join('collected_results', 'time_to_compute_gradients'), mode='a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([end - start])

    def upload(self, simulation, closest_rsu):
        # Send only the top k gradients in each layer of the network to save communication costs.
        if cfg['communication']['top_k_enabled'] or cfg['communication']['send_random_k_layers']:
            self.encode_gradients()
            self.print_gradient_size()

            # Upload data to RSU
            rsu = closest_rsu
            rsu.accumulative_gradients.append(self.gradients)

            # Save memory by deleting gradients from vehicle after upload.
            del self.gradients
            gc.collect()

            rsu.decode_gradients(simulation.central_server)
        # Use normal communication.
        else:
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
                self.compute(simulation, closest_rsu, training_data, label_data)
                self.upload(simulation, closest_rsu)
        else:
            # Shuffle the collected data
            self.handle_data()
            while self.training_data:
                self.compute(simulation, closest_rsu)
                self.upload(simulation, closest_rsu)

        self.training_data_assigned = {}

    def closest_rsu(self, rsu_list):
        """Return the RSU that is closest to the vehicle"""
        shortest_distance = 99999999  # placeholder (a random large number)
        closest_rsu = None
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range and distance < shortest_distance:
                shortest_distance = distance
                closest_rsu = rsu
        return closest_rsu

    def in_range_rsus(self, rsu_list):
        """Return a list of RSUs that is within the range of the vehicle
        with each RSU being sorted from the closest to the furthest"""
        in_range_rsus = []
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                heapq.heappush(in_range_rsus, (distance, rsu))
        return [heapq.heappop(in_range_rsus)[1] for i in range(len(in_range_rsus))]

    def in_polygon(self, polygons):
        """Return the index of the polygon the vehicle is currently in"""
        for i, polygon in enumerate(polygons):
            if polygon.contains(Point(self.x, self.y)):
                return i
        raise Exception('Vehicle not in any polygon')

    def encode_gradients(self):
        """Find the top-k gradients for each layer and encode them."""

        start = time.time()

        if cfg['communication']['top_k_enabled']:
            top_k_gradients = []
            for layer in self.gradients:
                # This function will return a 2d ndarray where row 0 has k values and row 1 has
                # k flattened indices. Ex: The last index of a 3 x 3 array would be 8 when flattened.
                top_k_gradients.append(nd.topk(layer, axis=None, k=cfg['communication']['k'], ret_typ='both'))
        elif cfg['communication']['send_random_k_layers']:
            pass

        # Overwrite other gradients.
        self.gradients = top_k_gradients

        end = time.time()

    def print_gradient_size(self):
        """Measure how much data is being transmitted from vehicle to gradient"""
        start = time.time()

        bytes_used = 0

        # Calculate number of bytes used in encoded gradients.
        if cfg['communication']['top_k_enabled']:
            for layer in self.gradients:
                # Each encoded layer is a list of 2 mxnet ndarrays. Thus, we count the elements in 
                # one of the ndarrays, multiply by the size in bytes of its datatype (float32), and
                # multiply by 2 since both mxnet ndarrays are of the same length and datatype
                bytes_used += layer[0].size * 32 * 2
        elif cfg['communication']['send_random_k_layers']:
            pass
        # Calculate number of bytes used in regular gradients.
        else:
            for layer in self.gradients:
                # Each layer is a one-dimensional to multidimensional ndarray.
                bytes_used += layer.size * 32

        if cfg['communication']['write_gradient_size']:
            with open(os.path.join('collected_results', 'gradient_sizes'), mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([bytes_used])

        end = time.time()

        if cfg['write_runtime_statistics']:
            with open(os.path.join('collected_results', 'time_to_print_gradients'), mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([end - start])
