import csv
import numpy as np
import yaml

file = open('../config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

runtime_writing_enabled = cfg['write_runtime_statistics']

runtimes = []

runtime_filenames = ['time_to_upload_and_receive_gradients', 'gradient_sizes', 'time_to_decode_gradients', 'time_to_print_gradients', 'time_to_encode_gradients', ]

if runtime_writing_enabled is True:
    for filename in runtime_filenames:
        # Store the runtime in each line as a float in an array to be calculated on.
        with open(filename) as log:
            line = log.readline()
            while line != '':
                runtimes.append(float(line))
                line = log.readline()

        print("Median and average runtime in seconds for " + filename + " :")
        print(str(np.percentile(runtimes, 50)) + ", " + str(np.average(runtimes)))
