import csv
import os

import psutil

from sumo import SUMO_Dataset
from central_server import Central_Server
from simulation import Simulation
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
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--num-round', type=int, default=0,
                        help='number of round.')
    opt = parser.parse_args()
    return opt


print(' '.join(sys.argv))

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
BATCH_SIZE = cfg['neural_network']['batch_size']


def extract_batch_from_polygon(simulation, polygon_index):
    if cfg['dataset'] == 'pascalvoc':
        # We do not delete elements in the list because mxnet ndarrays do not have delete operations.
        # Storing slice references of the training data will suffice.
        training_data_assigned = simulation.image_data_bypolygon[polygon_index][
                                 simulation.current_batch_index_by_polygon[polygon_index] * BATCH_SIZE:(
                                                                                                               simulation.current_batch_index_by_polygon[
                                                                                                                   polygon_index] + 1) * BATCH_SIZE]
        training_label_assigned = simulation.label_data_bypolygon[polygon_index][
                                  simulation.current_batch_index_by_polygon[polygon_index] * BATCH_SIZE:(
                                                                                                                simulation.current_batch_index_by_polygon[
                                                                                                                    polygon_index] + 1) * BATCH_SIZE]
        simulation.current_batch_index_by_polygon[polygon_index] += 1
    else:
        training_data_assigned = simulation.image_data_bypolygon[polygon_index][:BATCH_SIZE]
        del simulation.image_data_bypolygon[polygon_index][:BATCH_SIZE]
        training_label_assigned = simulation.label_data_bypolygon[polygon_index][:BATCH_SIZE]
        del simulation.label_data_bypolygon[polygon_index][:BATCH_SIZE]

    return training_data_assigned, training_label_assigned


def simulate(simulation):
    tree = ET.parse(simulation.FCD_file)
    root = tree.getroot()
    simulation.new_epoch(0, 0)
    data_last_found = None
    epoch_runtime_start = time.time()

    # Maximum training epochs
    while simulation.central_server.num_epoch <= cfg['neural_network']['epoch']:

        # Clear the vehicle dict after each loop of sumo file
        simulation.vehicle_dict = {}

        # For each time step (sec) in the FCD file 
        for timestep in root:

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
            vc_list = [vc for i, vc in enumerate(simulation.vc_list) if
                       vc_vehi_count[i] >= cfg['simulation']['vc_min_vehi']]
            # The combined list of physical rsus and vcs
            rsu_list = simulation.rsu_list + vc_list

            # For each vehicle on the map at the timestep (Training)
            for vehicle in timestep.findall('vehicle'):
                # Find the polygon the vehi is currently in.
                polygon_index = vehi.in_polygon(simulation.polygons)
                # If the vehi goes into a new polygon
                if polygon_index not in vehi.training_data_assigned:
                    # There is still data in this epoch. If each polygon has data but less than the batch
                    # size, discard them.
                    if ((cfg['dataset'] != 'pascalvoc') and (
                    any(len(x) >= BATCH_SIZE for x in simulation.image_data_bypolygon))) or ((any(
                            (current_batch_index + 1) * BATCH_SIZE <= len(simulation.image_data_bypolygon[i]) for
                            i, current_batch_index in enumerate(simulation.current_batch_index_by_polygon))) and (
                                                                                                     cfg[
                                                                                                         'dataset'] == 'pascalvoc')):
                        # There is still enough data in this polygon.
                        if (cfg['dataset'] != 'pascalvoc' and len(
                                simulation.image_data_bypolygon[polygon_index]) >= BATCH_SIZE) or (
                                cfg['dataset'] == 'pascalvoc' and (
                                (simulation.current_batch_index_by_polygon[polygon_index] + 1) * BATCH_SIZE <= len(
                            simulation.image_data_bypolygon[polygon_index]))):

                            training_data_assigned, training_label_assigned = extract_batch_from_polygon(simulation,
                                                                                                         polygon_index)

                            if cfg['dataset'] == 'pascalvoc':
                                print([len(image_data_in_polygon) - simulation.current_batch_index_by_polygon[
                                    i] * BATCH_SIZE for i, image_data_in_polygon in
                                       enumerate(simulation.image_data_bypolygon)])
                            else:
                                print([len(x) for x in simulation.image_data_bypolygon])

                            vehi.training_data_assigned[polygon_index] = (
                                training_data_assigned, training_label_assigned)
                    else:
                        epoch_runtime_end = time.time()
                        simulation.new_epoch(epoch_runtime_end - epoch_runtime_start,
                                             simulation.virtual_timestep + float(timestep.attrib['time']))
                        epoch_runtime_start = time.time()
                        if len(simulation.image_data_bypolygon[polygon_index]) >= BATCH_SIZE:
                            training_data_assigned, training_label_assigned = extract_batch_from_polygon(simulation,
                                                                                                         polygon_index)
                            vehi.training_data_assigned[polygon_index] = (
                                training_data_assigned, training_label_assigned)

                closest_rsu = vehi.closest_rsu(rsu_list)
                if closest_rsu is not None:
                    # Download Model
                    vehi.download_model_from(simulation.central_server)

                    vehi.compute_and_upload(simulation, closest_rsu)

        simulation.virtual_timestep += float(timestep.attrib['time'])

    return simulation.central_server.net


def main():
    print('initializing simulation...')

    opt = parse_args()

    num_gpus = opt.num_gpus
    context = mx.gpu(0)

    num_round = opt.num_round

    ROU_FILE = cfg['simulation']['ROU_FILE']
    NET_FILE = cfg['simulation']['NET_FILE']
    FCD_FILE = cfg['simulation']['FCD_FILE']

    RSU_RANGE = cfg['comm_range']['v2rsu']  # range of RSU
    NUM_RSU = cfg['simulation']['num_rsu']  # number of RSU
    NUM_VC = cfg['simulation']['num_vc']

    sumo_data = SUMO_Dataset(ROU_FILE, NET_FILE)
    vehicle_dict = {}
    # location_list = sumo_data.rsuList_random(RSU_RANGE, NUM_RSU+NUM_VC) # uncomment this for random locations
    location_list = sumo_data.rsuList(RSU_RANGE, NUM_RSU + NUM_VC, output_junctions)

    # Polygons class of zones partitioned
    with open("map_partition.data", "rb") as filehandle:
        polygons = pickle.load(filehandle)

    # Priotize RSU locations
    rsu_list = location_list[:NUM_RSU]
    vc_list = location_list[NUM_RSU:]

    central_server = Central_Server(context)

    # simulation = Simulation(FCD_FILE, vehicle_dict, rsu_list, vc_list, polygons, central_server, train_data, val_train_data, val_test_data, num_round)
    simulation = Simulation(FCD_FILE, vehicle_dict, rsu_list, vc_list, polygons, central_server, num_round)
    model = simulate(simulation)


if __name__ == '__main__':
    main()
