import time

import psutil

from neural_network import Neural_Network
from vehicle import Vehicle
from data_partition import data_for_polygon
from data_partition import val_train_data, val_test_data
from gluoncv.model_zoo import get_model
import gluoncv as gcv
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
import numpy as np
import yaml
import mxnet as mx
from mxnet import gluon, nd, autograd
import csv
import os
from memory_profiler import profile

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
# np.random.seed(cfg['seed'])

class Central_Server:
    """
    Central Server object for Car ML Simulator.
    Attributes:
    - model
    - accumulative_gradients
    """

    
    def __init__(self, ctx):
        self.ctx = ctx
        self.net = gluon.nn.Sequential()
        if cfg['dataset'] == 'cifar10':
            with self.net.name_scope():
                #  First convolutional layer
                self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1, 1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1, 1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                self.net.add(gluon.nn.Dropout(rate=0.25))
                #  Second convolutional layer
                # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                # Third convolutional layer
                self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1, 1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1, 1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                self.net.add(gluon.nn.Dropout(rate=0.25))
                # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
                # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
                # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
                # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                # Flatten and apply fullly connected layers
                self.net.add(gluon.nn.Flatten())
                # net.add(gluon.nn.Dense(512, activation="relu"))
                # net.add(gluon.nn.Dense(512, activation="relu"))
                self.net.add(gluon.nn.Dense(128, activation="relu"))
                # net.add(gluon.nn.Dense(256, activation="relu"))
                self.net.add(gluon.nn.Dropout(rate=0.25))
                self.net.add(gluon.nn.Dense(10))  # classes = 10
        elif cfg['dataset'] == 'mnist':
            with self.net.name_scope():
                self.net.add(gluon.nn.Dense(128, activation='relu'))
                self.net.add(gluon.nn.Dense(64, activation='relu'))
                self.net.add(gluon.nn.Dense(10))
        elif cfg['dataset'] == 'pascalvoc':
            self.net = get_model('yolo3_mobilenet1.0_voc', pretrained=False)

        self.net.initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
        # OR do self.net.load_parameters('models/yolo_x', ctx=ctx)

        self.accumulative_gradients = []

    # Update the model with its accumulative gradients
    # Used for batch gradient descent
    
    def update_model(self):
        if cfg['write_cpu_and_memory']:
            psutil.cpu_percent()
        if len(self.accumulative_gradients) >= 10:
            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in self.accumulative_gradients]
            mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
            idx = 0
            for j, (param) in enumerate(self.net.collect_params().values()):
                if param.grad_req != 'null':
                    # mapping back to the collection of ndarray
                    # directly update model
                    lr = cfg['neural_network']['learning_rate']
                    param.set_data(
                        param.data() - lr * mean_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
                    idx += param.data().size
            self.accumulative_gradients = []

            if cfg['dataset'] == 'pascalvoc':
                # Update targets when updating model.
                self.fake_x = mx.nd.zeros((cfg['neural_network']['batch_size'], 3, cfg['pascalvoc_metrics']['height'], cfg['pascalvoc_metrics']['width']))
                with autograd.train_mode():
                    _, self.anchors, self.offsets, self.feat_maps, _, _, _, _ = self.net(self.fake_x)

            if cfg['write_cpu_and_memory']:
                with open(os.path.join('collected_results', 'computer_resource_percentages'),
                          mode='a') as f:
                    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([psutil.cpu_percent(), psutil.virtual_memory().percent])


class Simulation:
    """
    Simulation object for Car ML Simulator. Stores all the global variables.
    Attributes:
    - FCD_file
    - vehicle_dict
    - rsu_list
    - dataset
    """

    
    def __init__(self, FCD_file, vehicle_dict: dict, rsu_list: list, vc_list: list, polygons, central_server,
                 num_round):
        self.FCD_file = FCD_file
        self.vehicle_dict = vehicle_dict
        self.rsu_list = rsu_list
        self.vc_list = vc_list
        self.polygons = polygons
        self.central_server = central_server
        self.num_epoch = 0
        self.training_data = []
        # self.training_set = training_set
        self.val_train_data = val_train_data
        self.val_test_data = val_test_data
        self.current_batch_index_by_polygon = [0 for i in range(len(polygons))]
        self.image_data_bypolygon = []
        self.label_data_bypolygon = []
        self.num_round = num_round
        self.virtual_timestep = 0
        if cfg['dataset'] == 'pascalvoc':
            self.fake_x = mx.nd.zeros((cfg['neural_network']['batch_size'], 3, cfg['pascalvoc_metrics']['height'], cfg['pascalvoc_metrics']['width']))
            with autograd.train_mode():
                _, self.anchors, self.offsets, self.feat_maps, _, _, _, _ = self.central_server.net(self.fake_x)
            self.target_generator = YOLOV3PrefetchTargetGenerator(
                num_class=len(self.central_server.net.classes))
            self.epoch_loss = gcv.loss.YOLOV3Loss()
            self.epoch_accuracy = VOCMApMetric(iou_thresh=cfg['pascalvoc_metrics']['iou_threshold'], class_names=central_server.net.classes)
            self.sum_losses = []
            self.obj_losses = []
            self.center_losses = []
            self.scale_losses = []
            self.cls_losses = []
        else:
            self.epoch_loss = mx.metric.CrossEntropy()
            self.epoch_accuracy = mx.metric.Accuracy()

    
    def add_into_vehicle_dict(self, vehicle):
        self.vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'])

    
    def get_accu_loss(self):
        if cfg['write_cpu_and_memory']:
            psutil.cpu_percent()
        # accuracy on testing data
        start_for_all_data = time.time()
        for (data, label) in self.val_test_data:
            start = time.time()
            if cfg['dataset'] == 'pascalvoc':
                outputs = self.central_server.net(data)
                pred_object_class_indices = outputs[:][0]
                pred_object_probabilities = outputs[:][1]
                pred_bboxes = outputs[:][2]
                gt_bboxes = label[:, :, 0:4]
                gt_class_indices = label[:, :, 4:5]
                self.epoch_accuracy.update(pred_bboxes, pred_object_class_indices, pred_object_probabilities, gt_bboxes,
                                           gt_class_indices)
            else:
                outputs = self.central_server.net(data)
                # this following line takes EXTREMELY LONG to run
                self.epoch_accuracy.update(label, outputs)
            end = time.time()

            if cfg['write_runtime_statistics']:
                with open(os.path.join('collected_results', 'time_to_calculate_accuracy_on_one_datum'), mode='a') as f:
                    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([end - start])
        print('time to calculate accuracy for', cfg['num_test_data'], 'test data:', end-start_for_all_data)


        start_for_all_data = time.time()
        # cross entropy on training data
        for data, label in self.val_train_data:
            start = time.time()
            if cfg['dataset'] == 'pascalvoc':
                # Acquire all variables required to calculate loss.
                gt_bboxes = mx.nd.array(label[:, :, :4]).astype(np.float32)
                gt_ids = label[:, :, 4:5]

                objectness, center_targets, scale_targets, weights, class_targets = self.target_generator(
                    self.fake_x, self.feat_maps, self.anchors, self.offsets,
                    gt_bboxes, gt_ids, None)

                # Calculate loss by using network in recording AND training mode and supplying extra target parameters.
                with autograd.record():
                    with autograd.train_mode():
                        # Returns four nd arrays, each with BATCH_SIZE number of losses.
                        obj_loss, center_loss, scale_loss, cls_loss = self.central_server.net(data, gt_bboxes, objectness, center_targets, scale_targets, weights, class_targets)
                sum_loss = obj_loss + center_loss + scale_loss + cls_loss

                # Store the average of the losses for each data batch.
                self.obj_losses.append(obj_loss.mean().asscalar())
                self.center_losses.append(center_loss.mean().asscalar())
                self.scale_losses.append(scale_loss.mean().asscalar())
                self.cls_losses.append(cls_loss.mean().asscalar())
                self.sum_losses.append(sum_loss.mean().asscalar())
            else:
                outputs = self.central_server.net(data)
                self.epoch_loss.update(label, nd.softmax(outputs))
            end = time.time()
            if cfg['write_runtime_statistics']:
                with open(os.path.join('collected_results', 'time_to_calculate_loss_on_one_datum'), mode='a') as f:
                    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([end - start])

        if cfg['write_cpu_and_memory']:
            with open(os.path.join('collected_results', 'computer_resource_percentages'),
                      mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([psutil.cpu_percent(), psutil.virtual_memory().percent])
        print('time it takes to calculate loss for', cfg['num_val_train_data'], 'validation data', end-start_for_all_data)

    
    def print_accuracy(self, epoch_runtime, virtual_time_step):
        self.epoch_accuracy.reset()
        if cfg['dataset'] == 'pascalvoc':
            self.obj_losses = []
            self.center_losses = []
            self.scale_losses = []
            self.cls_losses = []
            self.sum_losses = []
        else:
            self.epoch_loss.reset()
        print("finding accu and loss ...")

        # Calculate accuracy and loss
        self.get_accu_loss()

        # Retrieve acuracy and loss and then save them into a csv.
        _, accu = self.epoch_accuracy.get()

        if cfg['dataset'] == 'pascalvoc':
            obj_loss = np.array(self.obj_losses).mean()
            center_loss = np.array(self.center_losses).mean()
            scale_loss = np.array(self.scale_losses).mean()
            cls_loss = np.array(self.cls_losses).mean()
            sum_loss = np.array(self.sum_losses).mean()

            self.save_data(accu, sum_loss, epoch_runtime, virtual_time_step, obj_loss, center_loss, scale_loss, cls_loss)
            print(
                "Epoch {:03d}: Loss: {:03f}, Accuracy: {}, Epoch Runtime: {}, Virtual Time Step: {}, Object Loss: {:03f}, Center Loss: {:03f}, Scale Loss: {:03f}, CLS Loss: {:03f}\n".format(
                    self.num_epoch,
                    sum_loss, accu, epoch_runtime, virtual_time_step,
                    obj_loss, center_loss, scale_loss, cls_loss))
        else:
            _, loss = self.epoch_loss.get()
            self.save_data(accu, loss, virtual_time_step)
            print("Epoch {:03d}: Loss: {:03f}, Accuracy: {:03f}\n".format(self.num_epoch,
                                                                          loss,
                                                                          accu))

    
    def save_data(self, accu, loss, epoch_runtime, virtual_time_step, *losses):
        if not os.path.exists('collected_results'):
            os.makedirs('collected_results')
        dir_name = cfg['dataset'] + '-' + 'VC' + str(cfg['simulation']['num_vc']) + '-' + 'round' + str(
            self.num_round) + '.csv'
        p = os.path.join('collected_results', dir_name)

        with open(p, mode='a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if cfg['dataset'] == 'pascalvoc':
                if self.num_epoch == 1:
                    writer.writerow(['Epoch number', 'Epoch Runtime', 'Virtual Time Step', 'Accuracy (By class), IOU Thresh is ' + str(cfg['pascalvoc_metrics']['iou_threshold']), 'Loss', 'Object Loss', 'Center Loss', 'Scale Loss', 'CLS Loss', 'Aggregation Method', 'Attack Type'])
                writer.writerow([self.num_epoch, epoch_runtime, virtual_time_step, accu, loss, losses[0], losses[1], losses[2], losses[3], cfg['aggregation_method'], cfg['attack']])
            else:
                writer.writerow([self.num_epoch, epoch_runtime, virtual_time_step, accu, loss, cfg['aggregation_method'], cfg['attack']])


    def new_epoch(self, *args):
        if len(args) != 0:
            epoch_runtime = args[0]
            virtual_time_step = args[1]

        if self.num_epoch != 0:
            print('Epoch', self.num_epoch, 'runtime:', epoch_runtime)

        # Calculate accuracy and loss every 5 epochs.
        if self.num_epoch != 0 and self.num_epoch % 5 == 0:
            self.print_accuracy(epoch_runtime, virtual_time_step)

        if self.num_epoch != 0 and self.num_epoch % 10 == 0:
            if not os.path.exists('models'):
                os.makedirs('models')
            filename = 'models/yolo_' + str(self.num_epoch)
            self.central_server.net.save_parameters(filename)

        self.num_epoch += 1
        print("partitioning data...")
        self.image_data_bypolygon, self.label_data_bypolygon = data_for_polygon(self.polygons)
        self.current_batch_index_by_polygon = [0 for i in range(len(self.polygons))]
