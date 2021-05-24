from neural_network import Neural_Network
from vehicle import Vehicle
from data_partition import data_for_polygon
from data_partition import val_train_data, val_test_data
from gluoncv import model_zoo
import gluoncv as gcv
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
import numpy as np
import yaml
import mxnet as mx
from mxnet import gluon, nd
import csv
import os


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
        self.net = gluon.nn.Sequential()
        if cfg['dataset'] == 'cifar10':
            with self.net.name_scope():
                #  First convolutional layer
                self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                self.net.add(gluon.nn.Dropout(rate=0.25))
                #  Second convolutional layer
                # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                # Third convolutional layer
                self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
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
                self.net.add(gluon.nn.Dense(10)) # classes = 10
        elif cfg['dataset'] == 'mnist':
            with self.net.name_scope():
                self.net.add(gluon.nn.Dense(128, activation='relu'))
                self.net.add(gluon.nn.Dense(64, activation='relu'))
                self.net.add(gluon.nn.Dense(10))
        elif cfg['dataset'] == 'pascalvoc':
            net = model_zoo.get_model('yolo3_darknet53_voc', pretrained_base=False)
        self.net.initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)

        self.accumulative_gradients = []

    # Update the model with its accumulative gradients
    # Used for batch gradient descent
    def update_model(self):
        if len(self.accumulative_gradients) >= 10:
            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in self.accumulative_gradients]
            mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
            idx = 0
            for j, (param) in enumerate(self.net.collect_params().values()):
                if param.grad_req != 'null':
                    # mapping back to the collection of ndarray
                    # directly update model
                    lr = cfg['neural_network']['learning_rate']
                    param.set_data(param.data() - lr * mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
                    idx += param.data().size
            self.accumulative_gradients = []


class Simulation:
    """
    Simulation object for Car ML Simulator. Stores all the global variables.
    Attributes:
    - FCD_file
    - vehicle_dict
    - rsu_list
    - dataset
    """
    def __init__(self, FCD_file, vehicle_dict: dict, rsu_list: list, vc_list: list, polygons, central_server, num_round):
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
        self.training_data_byclass = []
        self.training_label_byclass = []
        self.num_round = num_round
        self.running_time = 0
        if cfg['dataset'] == 'pascalvoc':
            self.epoch_loss = gcv.loss.YOLOV3Loss()
            self.epoch_accuracy = VOCMApMetric()
        else:
            self.epoch_loss = mx.metric.CrossEntropy()
            self.epoch_accuracy = mx.metric.Accuracy()
       
    def add_into_vehicle_dict(self, vehicle):
        self.vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'])

    def get_accu_loss(self):
        # accuracy on testing data
        for i, (data, label) in enumerate(self.val_test_data):
            outputs = self.central_server.net(data)
            # this following line takes EXTREMELY LONG to run
            self.epoch_accuracy.update(label, outputs)
        # cross entropy on training data
        for i, (data, label) in enumerate(self.val_train_data):
            outputs = self.central_server.net(data)
            self.epoch_loss.update(label, nd.softmax(outputs))


    def print_accuracy(self, time):
        self.epoch_accuracy.reset()
        self.epoch_loss.reset()
        print("finding accu and loss ...")

        # Calculate accuracy and loss
        self.get_accu_loss()

        _, accu = self.epoch_accuracy.get()
        _, loss = self.epoch_loss.get()

        # Save accuracy and loss to csv
        self.save_data(accu, loss, time)

        print("Epoch {:03d}: Loss: {:03f}, Accuracy: {:03f}\n".format(self.num_epoch,
                                                                    loss,
                                                                    accu))

    def save_data(self, accu, loss, time):
        if not os.path.exists('collected_results'):
            os.makedirs('collected_results')
        dir_name = cfg['dataset'] + '-' + 'VC' + str(cfg['simulation']['num_vc']) + '-' + 'round' + str(self.num_round) + '.csv'
        p = os.path.join('collected_results', dir_name)
        with open(p, mode='a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.num_epoch, time, accu, loss, cfg['aggregation_method'], cfg['attack']])
            

    def new_epoch(self):
        self.num_epoch += 1
        # for i, (data, label) in enumerate(self.training_set):
        #     self.training_data.append((data, label))
        self.training_data_byclass, self.training_label_byclass = data_for_polygon(self.polygons)

