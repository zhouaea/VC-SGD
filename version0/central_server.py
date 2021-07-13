from gluoncv.model_zoo import get_model
import yaml
import mxnet as mx
from mxnet import gluon, nd, autograd

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
batch_size = cfg['neural_network']['batch_size']
# np.random.seed(cfg['seed'])

class Central_Server:
    """
    Central Server object for Car ML Simulator.
    Attributes:
    - model
    - accumulative_gradients
    - decoded_data_template (if using communication efficient mode)
    """

    
    def __init__(self, ctx):
        self.ctx = ctx
        self.num_epoch = 0
        self.lr = cfg['neural_network']['learning_rate']
        self.net = gluon.nn.Sequential()
        if cfg['dataset'] == 'cifar10':
            with self.net.name_scope():
                #  First convolutional layer
                self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1, 1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1, 1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                self.net.add(gluon.nn.Dropout(rate=0.25))
                #  Second convolutional layer
                # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                # Third convolutional layer
                self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1, 1), activation='relu'))
                self.net.add(gluon.nn.BatchNorm())
                self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1, 1), activation='relu'))
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
                self.net.add(gluon.nn.Dense(10))  # classes = 10
        elif cfg['dataset'] == 'mnist':
            with self.net.name_scope():
                self.net.add(gluon.nn.Dense(128, activation='relu'))
                self.net.add(gluon.nn.Dense(64, activation='relu'))
                self.net.add(gluon.nn.Dense(10))
        elif cfg['dataset'] == 'pascalvoc':
            self.net = get_model('yolo3_mobilenet1.0_voc', pretrained=False)

        self.net.initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
        # OR do self.net.load_parameters('models/model_x', ctx=ctx)

        self.accumulative_gradients = []

        # Load structure of gradients of the specific model being used so RSUs can decode uploaded vehicle data.
        if cfg['communication']['top_k_enabled']:
            if cfg['dataset'] == 'mnist':
                with open('gradient_format/sequential_gradient_format.yml', 'r') as infile:
                    gradient_structure = yaml.load(infile, Loader=yaml.FullLoader)
            elif cfg['dataset'] == 'pascalvoc':
                with open('gradient_format/yolov3_gradient_format.yml', 'r') as infile:
                    gradient_structure = yaml.load(infile, Loader=yaml.FullLoader)

            self.decoded_data_template = []

            # Create a list of 2D ndarrays with the structure of the model gradients, initialized to 0.
            for layer_shape in gradient_structure:
                self.decoded_data_template.append(nd.zeros(layer_shape))


    def update_model(self):
        """Update the model with its accumulative gradients. Used for batch gradient descent"""
        if len(self.accumulative_gradients) >= cfg['simulation']['maximum_rsu_accumulative_gradients']:
            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in self.accumulative_gradients]
            mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
            idx = 0
            print('lr:', self.lr)
            for j, (param) in enumerate(self.net.collect_params().values()):
                if param.grad_req != 'null':
                    # mapping back to the collection of ndarray
                    # directly update model
                    param.set_data(
                        param.data() - self.lr * mean_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
                    idx += param.data().size
            self.accumulative_gradients = []

            if cfg['dataset'] == 'pascalvoc':
                # Update targets when updating model.
                self.fake_x = mx.nd.zeros((batch_size, 3, cfg['pascalvoc_metrics']['height'], cfg['pascalvoc_metrics']['width']))
                with autograd.train_mode():
                    _, self.anchors, self.offsets, self.feat_maps, _, _, _, _ = self.net(self.fake_x)