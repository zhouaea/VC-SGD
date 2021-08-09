import time

from vehicle import Vehicle
from data_partition import data_for_polygon
from data_partition import val_train_data, val_test_data
import gluoncv as gcv
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
import numpy as np
import yaml
import mxnet as mx
from mxnet import gluon, nd, autograd
import csv
import os

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
batch_size = cfg['neural_network']['batch_size']

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
            self.fake_x = mx.nd.zeros(
                (batch_size, 3, cfg['pascalvoc_metrics']['height'], cfg['pascalvoc_metrics']['width']), ctx=self.central_server.ctx)
            with autograd.train_mode():
                _, self.anchors, self.offsets, self.feat_maps, _, _, _, _ = self.central_server.net(self.fake_x)
            self.target_generator = YOLOV3PrefetchTargetGenerator(
                num_class=len(self.central_server.net.classes))
            self.epoch_loss = gcv.loss.YOLOV3Loss()
            self.epoch_accuracy = VOCMApMetric(iou_thresh=cfg['pascalvoc_metrics']['iou_threshold'],
                                               class_names=central_server.net.classes)
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
        # accuracy on testing data
        start_for_all_data = time.time()
        for (data, label) in self.val_test_data:
            if cfg['dataset'] == 'pascalvoc':
                # Load neural network inputs into gpu memory, if using a gpu.
                data = nd.array(data, ctx=self.central_server.ctx)
                gt_bboxes = nd.array(label[:, :, 4:5], ctx=self.central_server.ctx)
                gt_class_indices = nd.array(label[:, :, 4:5], ctx=self.central_server.ctx)

                outputs = self.central_server.net(data)
                pred_object_class_indices = outputs[:][0]
                pred_object_probabilities = outputs[:][1]
                pred_bboxes = outputs[:][2]
                self.epoch_accuracy.update(pred_bboxes, pred_object_class_indices, pred_object_probabilities, gt_bboxes,
                                           gt_class_indices)
            else:
                outputs = self.central_server.net(data)
                # this following line takes EXTREMELY LONG to run
                self.epoch_accuracy.update(label, outputs)
        end = time.time()
        print('time it takes to calculate accuracy for', cfg['num_test_data'], 'test data',
              end - start_for_all_data)

        start_for_all_data = time.time()
        # cross entropy on training data
        for data, label in self.val_train_data:
            if cfg['dataset'] == 'pascalvoc':
                # Load neural network inputs into gpu memory, if using a gpu.
                data = nd.array(data, ctx=self.central_server.ctx)
                gt_bboxes = mx.nd.array(label[:, :, :4], ctx=self.central_server.ctx)
                gt_ids = mx.nd.array(label[:, :, 4:5], ctx=self.central_server.ctx)

                objectness, center_targets, scale_targets, weights, class_targets = self.target_generator(
                    self.fake_x, self.feat_maps, self.anchors, self.offsets,
                    gt_bboxes, gt_ids, None)

                # Calculate loss by using network in recording AND training mode and supplying extra target parameters.
                with autograd.record():
                    with autograd.train_mode():
                        # Returns four nd arrays, each with BATCH_SIZE number of losses.
                        obj_loss, center_loss, scale_loss, cls_loss = self.central_server.net(data, gt_bboxes,
                                                                                              objectness,
                                                                                              center_targets,
                                                                                              scale_targets, weights,
                                                                                              class_targets)
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

        print('time it takes to calculate loss for', cfg['num_val_train_data'], 'validation data',
              end - start_for_all_data)

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

            self.save_data(accu, sum_loss, epoch_runtime, virtual_time_step, obj_loss, center_loss, scale_loss,
                           cls_loss)
        else:
            _, loss = self.epoch_loss.get()
            self.save_data(accu, loss, epoch_runtime, virtual_time_step)

    def save_data(self, accu, loss, epoch_runtime, virtual_time_step, *losses):
        if not os.path.exists('collected_results'):
            os.makedirs('collected_results')
        dir_name = cfg['dataset'] + '-' + 'VC' + str(cfg['simulation']['num_vc']) + '-' + 'round' + str(
            self.num_round) + '.csv'
        p = os.path.join('collected_results', dir_name)

        with open(p, mode='a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if cfg['dataset'] == 'pascalvoc':
                if self.central_server.num_epoch == 5:
                    writer.writerow(['Epoch number', 'Epoch Runtime', 'Virtual Time Step',
                                     'Accuracy (By class), IOU Thresh is ' + str(
                                         cfg['pascalvoc_metrics']['iou_threshold']), 'Loss', 'Object Loss',
                                     'Center Loss', 'Scale Loss', 'CLS Loss', 'Aggregation Method', 'Attack Type'])
                writer.writerow(
                    [self.central_server.num_epoch, epoch_runtime, virtual_time_step, accu, loss, losses[0], losses[1],
                     losses[2], losses[3], cfg['aggregation_method'], cfg['attack']])
            else:
                if self.central_server.num_epoch == 5:
                    writer.writerow(
                        ['Epoch number', 'Epoch Runtime', 'Virtual Time Step', 'Accuracy', 'Loss', 'Aggregation Method',
                         'Attack Type'])
                writer.writerow([self.central_server.num_epoch, epoch_runtime, virtual_time_step, accu, loss,
                                 cfg['aggregation_method'], cfg['attack']])

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')

        if cfg['dataset'] == 'pascalvoc':
            filename = 'models/yolo_' + str(self.central_server.num_epoch)
        elif cfg['dataset'] == 'mnist':
            filename = 'models/6_layer_sequential_' + str(self.central_server.num_epoch)
        else:
            filename = 'models/some_model_' + str(self.central_server.num_epoch)

        self.central_server.net.save_parameters(filename)

    def new_epoch(self, epoch_runtime, virtual_time_step):
        # Write the runtime of each epoch.
        if self.central_server.num_epoch != 0:
            print('Epoch', self.central_server.num_epoch, 'runtime:', epoch_runtime)
            with open(os.path.join('collected_results', 'epoch_runtime'), mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch_runtime])

        # Calculate accuracy and loss every 5 epochs.
        if self.central_server.num_epoch != 0 and self.central_server.num_epoch % 5 == 0:
            self.print_accuracy(epoch_runtime, virtual_time_step)

        # Save model every 10 epochs
        if self.central_server.num_epoch != 0 and self.central_server.num_epoch % 10 == 0:
            self.save_model()

        self.central_server.num_epoch += 1

        # Decay learning rate when 80% of epochs are completed and 90% of epochs are completed.
        if self.central_server.num_epoch == 0.8 * cfg['neural_network'][
            'epoch'] - 1 or self.central_server.num_epoch == 0.9 * cfg['neural_network']['epoch'] - 1:
            self.central_server.lr *= 0.1

        # Exit after all epochs are completed, so data does not have to be repartitioned before the simulation exits.
        if self.central_server.num_epoch > cfg['neural_network']['epoch']:
            exit()

        print("partitioning data...")
        self.image_data_bypolygon, self.label_data_bypolygon = data_for_polygon(self.polygons)
        self.current_batch_index_by_polygon = [0 for i in range(len(self.polygons))]
