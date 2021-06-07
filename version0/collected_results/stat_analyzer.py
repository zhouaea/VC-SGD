import numpy as np

runtimes = []

filename = 'time_to_train_on_one_batch'

# Store the runtime in each line as a float in an array to be calculated on.
with open(filename) as log:
    line = log.readline()
    while line != '':
        runtimes.append(float(line))
        line = log.readline()

print("Median and average in milliseconds:")
print(str(np.percentile(runtimes, 50)) + ", " + str(np.average(runtimes)))