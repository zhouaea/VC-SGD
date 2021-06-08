import numpy as np
import yaml

file = open('../config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

runtime_writing_enabled = cfg['write_runtime_statistics']
cpu_and_memory_writing_enabled = cfg['write_cpu_and_memory']

runtimes = []

runtime_filenames = ['time_to_train_on_one_batch', 'time_to_calculate_accuracy_on_one_datum', 'time_to_calculate_loss_on_one_datum', 'time_for_vehicle_to_enter_zone_with_data']
cpu_and_memory_filenames = ['computer_resource_percentages']

if runtime_writing_enabled is True:
    for filename in runtime_filenames:
        # Store the runtime in each line as a float in an array to be calculated on.
        with open(filename) as log:
            line = log.readline()
            while line != '':
                runtimes.append(float(line))
                line = log.readline()

        print("Median and average in milliseconds for " + filename + " :")
        print(str(np.percentile(runtimes, 50)) + ", " + str(np.average(runtimes)))

cpu_percentages = []
ram_percentages = []

if cpu_and_memory_writing_enabled is True:
    for filename in cpu_and_memory_filenames:
        # Store the runtime in each line as a float in an array to be calculated on.
        with open(filename) as log:
            line = log.readline()
            while line != '':
                cpu_percentages.append(float(line[0]))
                ram_percentages.append(float(line[1]))
                line = log.readline()

    print("Median and average in milliseconds for " + 'CPU' + " :")
    print(str(np.percentile(cpu_percentages, 50)) + ", " + str(np.average(cpu_percentages)))

    print("Median and average in milliseconds for " + 'RAM' + " :")
    print(str(np.percentile(cpu_percentages, 50)) + ", " + str(np.average(cpu_percentages)))