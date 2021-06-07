import numpy as np

runtimes = []

filenames = ['time_to_train_on_one_batch', 'time_to_calculate_accuracy_on_one_datum', 'time_to_calculate_loss_on_one_datum', 'time_for_vehicle_to_enter_zone_with_data']

for filename in filenames:
    # Store the runtime in each line as a float in an array to be calculated on.
    with open(filename) as log:
        line = log.readline()
        while line != '':
            runtimes.append(float(line))
            line = log.readline()

    print("Median and average in milliseconds for" + filename + ":")
    print(str(np.percentile(runtimes, 50)) + ", " + str(np.average(runtimes)))