import yaml

file = open('../config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

runtime_writing_enabled = cfg['write_runtime_statistics']
cpu_and_memory_writing_enabled = cfg['write_cpu_and_memory']

runtime_filenames = ['time_to_train_on_one_batch', 'time_to_calculate_accuracy_on_one_datum', 'time_to_calculate_loss_on_one_datum', 'time_for_vehicle_to_enter_zone_with_data']
cpu_and_memory_filenames = ['computer_resource_percentages']

if runtime_writing_enabled is True:
    for filename in runtime_filenames:
        f = open(filename, "w")
        f.truncate()
        f.close()

if cpu_and_memory_writing_enabled is True:
    for filename in cpu_and_memory_filenames:
        f = open(filename, "w")
        f.truncate()
        f.close()
