from neural_network import Neural_Network 
import byz
import nd_aggregation
import numpy as np
import yaml
import random
from mxnet import nd

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

# random.seed(cfg['seed'])
# np.random.seed(cfg['seed'])

# Load structure of gradients of the specific model being used to decode uploaded vehicle data.
if cfg['communication']['top_k_enabled']:
    if cfg['dataset'] == 'mnist':
        with open('gradient_format/sequential_gradient_format.yml', 'r') as infile:
            gradient_structure = yaml.load(infile, Loader=yaml.FullLoader)
    elif cfg['dataset'] == 'pascalvoc':
        with open('gradient_format/yolov3_gradient_format.yml', 'r') as infile:
            gradient_structure = yaml.load(infile, Loader=yaml.FullLoader)

class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - rsu_x
    - rsu_y
    - rsu_range
    - accumulative_gradients
    """
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range, traffic_proportion):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.accumulative_gradients = []

    def aggregate(self, net, grad_list, byz=byz.no_byz):
        f = cfg['num_faulty_grads']
        aggre_method = cfg['aggregation_method']
        if aggre_method == 'cgc':
            return nd_aggregation.cgc_filter(grad_list, net, f, byz)
        elif aggre_method == 'simplemean':
            return nd_aggregation.simple_mean_filter(grad_list, net, f, byz)

    # The RSU updates the model in the central server with its accumulative gradients and downloads the 
    # latest model from the central server
    def communicate_with_central_server(self, central_server):
        # Different methods of attacking
        if cfg['attack'] == 'signflip':
            byz.signflip_attack(self)
            aggre_gradients = self.aggregate(central_server.net, self.accumulative_gradients)
        elif cfg['attack'] == 'gaussian':
            aggre_gradients = self.aggregate(central_server.net, self.accumulative_gradients, byz.gaussian_attack)
        elif cfg['attack'] == 'bitflip':
            aggre_gradients = self.aggregate(central_server.net, self.accumulative_gradients, byz.bitflip_attack)
        else:
            # NO attack
            aggre_gradients = self.aggregate(central_server.net, self.accumulative_gradients)
            
        self.accumulative_gradients = []
        central_server.accumulative_gradients.append(aggre_gradients)
        # if enough gradients accumulated in cloud, then update model
        if len(central_server.accumulative_gradients) >= cfg['simulation']['maximum_rsu_accumulative_gradients']:
            central_server.update_model()

    def decode_gradients(self):
        """Decode data sent from vehicle"""
        received_gradient_index = len(self.accumulative_gradients) - 1
        encoded_data = self.accumulative_gradients[received_gradient_index]
        decoded_data = []

        # Create a list of 2D ndarrays with the structure of the model gradients, initialized to 0.
        for layer_shape in gradient_structure:
            decoded_data.append(nd.zeros(layer_shape))

        # Use the values and indices in the encoded list to change the values of the newly
        # initialized gradient list.
        for layer, (top_k_values_in_layer, top_k_flattened_indices_in_layer) in enumerate(encoded_data):
            # Go through each top-k element for the layer.
            for j in range(len(top_k_values_in_layer)):
                layer_shape = decoded_data[layer].shape

                # Reshape each flattened index according to the dimensions of the layer.
                if len(layer_shape) == 1:
                    # No need to reshape flattened index since the layer is already one dimensional.
                    reshaped_index = int(top_k_flattened_indices_in_layer[j].asscalar())
                elif len(layer_shape) == 2:
                    # Divide index by the number of columns of the layer to get two dimensional indices.
                    x, y = divmod(int(top_k_flattened_indices_in_layer[j].asscalar()),
                                  layer_shape[1])
                    reshaped_index = (x, y)
                # Layer is three dimensional.
                elif len(layer_shape) == 3:
                    # Divide index by len(row) * len(col) to get depth.
                    x, x_remainder = divmod(int(top_k_flattened_indices_in_layer[j].asscalar()),
                                            (layer_shape[1] * layer_shape[2]))
                    # Divide remainder of that by number of columns to get row and column number.
                    y, z = divmod(x_remainder, layer_shape[2])
                    reshaped_index = (x, y, z)
                else:
                    print('VCSGD error, dimension of gradient layer is not supported:',
                          len(layer_shape))
                    exit()

                # Change specific element in layer to reflect result obtained from vehicle.
                decoded_data[layer][reshaped_index] = top_k_values_in_layer[j]

        self.accumulative_gradients[received_gradient_index] = decoded_data
