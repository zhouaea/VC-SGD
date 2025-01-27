from sumo import SUMO_Dataset
from central_server import Central_Server, Simulation
from vehicle import Vehicle

import yaml
from locationPicker_v3 import output_junctions
# from map_partition import polygons
import xml.etree.ElementTree as ET 

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
import numpy as np
import time, random, argparse, itertools
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--num-round', type=int, default=0,
                        help='number of round.')
    opt = parser.parse_args()
    return opt

import sys
print(' '.join(sys.argv))

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
BATCH_SIZE = cfg['neural_network']['batch_size']

def simulate(simulation):
    tree = ET.parse(simulation.FCD_file)
    root = tree.getroot()
    simulation.new_epoch()
    
    # Maximum training epochs
    while simulation.num_epoch <= cfg['neural_network']['epoch']:

        # Clear the vehicle dict after each loop of sumo file
        simulation.vehicle_dict = {}

        # For each time step (sec) in the FCD file 
        for timestep in root:

            if simulation.num_epoch > cfg['neural_network']['epoch']:
                break

            # if simulation.num_epoch > cfg['neural_network']['epoch']:
            #     break
            if float(timestep.attrib['time']) % 200 == 0:
                print(timestep.attrib['time'])

            vc_vehi_count = [0 for vc in simulation.vc_list]
            # For each vehicle on the map at the timestep (Find available vehicular clouds)
            for vehicle in timestep.findall('vehicle'):

                # If vehicle not yet stored in vehicle_dict
                if vehicle.attrib['id'] not in simulation.vehicle_dict:
                    simulation.add_into_vehicle_dict(vehicle)
                # Get the vehicle object from vehicle_dict
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]  
                # Set location and speed
                vehi.set_properties(float(vehicle.attrib['x']),
                                    float(vehicle.attrib['y']),
                                    float(vehicle.attrib['speed']))

                # Find car count for each vehicular cloud
                for i, vc in enumerate(simulation.vc_list):
                    if (vc.rsu_x - vehi.x) ** 2 + (vc.rsu_y - vehi.y) ** 2 <= cfg['comm_range']['v2rsu'] ** 2:
                        vc_vehi_count[i] += 1
            # The list of vehicular clouds that have enough cars
            vc_list = [vc for i, vc in enumerate(simulation.vc_list) if vc_vehi_count[i] >= cfg['simulation']['vc_min_vehi']]
            # The combined list of physical rsus and vcs
            rsu_list = simulation.rsu_list + vc_list  

            # For each vehicle on the map at the timestep (Training)
            for vehicle in timestep.findall('vehicle'):
                
                # Find the polygon the vehi is currently in and assign polygon-specific data to it
                polygon_index = vehi.in_polygon(simulation.polygons)
                # If the vehi goes into a new polygon
                if polygon_index not in vehi.training_data_assigned:
                    # There is still data in this epoch
                    # print([len(i) for i in simulation.training_data_byclass])
                    if any(len(x) >= BATCH_SIZE for x in simulation.training_data_byclass):
                        # There is still data in this polygon
                        if len(simulation.training_data_byclass[polygon_index]) >= BATCH_SIZE:
                            training_data_assigned = simulation.training_data_byclass[polygon_index][:BATCH_SIZE]
                            del simulation.training_data_byclass[polygon_index][:BATCH_SIZE]
                            training_label_assigned = simulation.training_label_byclass[polygon_index][:BATCH_SIZE]
                            del simulation.training_label_byclass[polygon_index][:BATCH_SIZE]
                            vehi.training_data_assigned[polygon_index] = (training_data_assigned, training_label_assigned)
                    else:
                        simulation.print_accuracy(simulation.running_time + float(timestep.attrib['time']))
                        simulation.new_epoch()
                        if len(simulation.training_data_byclass[polygon_index]) >= BATCH_SIZE:
                            training_data_assigned = simulation.training_data_byclass[polygon_index][:BATCH_SIZE]
                            del simulation.training_data_byclass[polygon_index][:BATCH_SIZE]
                            training_label_assigned = simulation.training_label_byclass[polygon_index][:BATCH_SIZE]
                            del simulation.training_label_byclass[polygon_index][:BATCH_SIZE]
                            vehi.training_data_assigned[polygon_index] = (training_data_assigned, training_label_assigned)
                
                closest_rsu = vehi.closest_rsu(rsu_list)
                if closest_rsu is not None:
                    # Download Model
                    vehi.download_model_from(simulation.central_server)

                    vehi.compute_and_upload(simulation, closest_rsu)

        simulation.running_time += float(timestep.attrib['time'])   
                
    return simulation.central_server.net


def main():
    opt = parse_args()

    num_gpus = opt.num_gpus
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    num_round = opt.num_round

    ROU_FILE = cfg['simulation']['ROU_FILE']
    NET_FILE = cfg['simulation']['NET_FILE']
    FCD_FILE = cfg['simulation']['FCD_FILE']
    
    RSU_RANGE = cfg['comm_range']['v2rsu']           # range of RSU
    NUM_RSU = cfg['simulation']['num_rsu']           # number of RSU
    NUM_VC = cfg['simulation']['num_vc']

    sumo_data = SUMO_Dataset(ROU_FILE, NET_FILE)
    vehicle_dict = {}
    # location_list = sumo_data.rsuList_random(RSU_RANGE, NUM_RSU+NUM_VC) # uncomment this for random locations
    location_list = sumo_data.rsuList(RSU_RANGE, NUM_RSU+NUM_VC, output_junctions)

    # Polygons class of zones partitioned
    with open("map_partition.data", "rb") as filehandle:
        polygons = pickle.load(filehandle)

    # Priotize RSU locations
    rsu_list = location_list[:NUM_RSU]
    vc_list = location_list[NUM_RSU:]

    # Proitize VC locations
    # rsu_list = location_list[NUM_VC:]
    # vc_list = location_list[:NUM_VC]

    central_server = Central_Server(context)


    # def transform(data, label):
    #     if cfg['dataset'] == 'cifar10':
    #         data = mx.nd.transpose(data, (2,0,1))
    #     data = data.astype(np.float32) / 255
    #     return data, label

    # # Load Data
    # batch_size = cfg['neural_network']['batch_size']
    # num_training_data = cfg['num_training_data']
    # if cfg['dataset'] == 'cifar10':
    #     train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('../data/cx2', train=True, transform=transform).take(num_training_data),
    #                             batch_size, shuffle=True, last_batch='discard')
    #     val_train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('../data/cx2', train=True, transform=transform).take(cfg['num_val_loss']),
    #                                 batch_size, shuffle=False, last_batch='keep')
    #     val_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('../data/cx2', train=False, transform=transform),
    #                                 batch_size, shuffle=False, last_batch='keep')
    # elif cfg['dataset'] == 'mnist':
    #     train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=transform).take(num_training_data),
    #                             batch_size, shuffle=True, last_batch='discard')
    #     val_train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=transform).take(cfg['num_val_loss']),
    #                                 batch_size, shuffle=False, last_batch='keep')
    #     val_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=False, transform=transform),
    #                                 batch_size, shuffle=False, last_batch='keep')


    # simulation = Simulation(FCD_FILE, vehicle_dict, rsu_list, vc_list, polygons, central_server, train_data, val_train_data, val_test_data, num_round)
    simulation = Simulation(FCD_FILE, vehicle_dict, rsu_list, vc_list, polygons, central_server, num_round)
    model = simulate(simulation)

    # # Test the accuracy of the computed model
    # test_accuracy = tf.keras.metrics.Accuracy()     
    # for (x, y) in test_dataset:
    #     logits = model(x, training=False) 
    #     prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    #     test_accuracy(prediction, y)
    # print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


if __name__ == '__main__':
    main()
