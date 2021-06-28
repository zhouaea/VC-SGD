import yaml

file = open('../config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

runtime_writing_enabled = cfg['write_runtime_statistics']

runtime_filenames = ['time_to_upload_and_receive_gradients', 'gradient_sizes']

if runtime_writing_enabled is True:
    for filename in runtime_filenames:
        f = open(filename, "w")
        f.truncate()
        f.close()
