from collections import OrderedDict
import numpy as np
import torch
import warnings
from torch import nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import ModuleDict
from mlutils.constraints import positive
from mlutils.layers.cores import DepthSeparableConv2d, Core2d, Stacked2dCore
from ..utility.nn_helpers import get_io_dims, get_module_output, set_random_seed, get_dims_for_loader_dict
from mlutils import regularizers
from mlutils.layers.readouts import PointPooled2d
from .pretrained_models import TransferLearningCore


class DepthSeparableCore(Core2d, nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=3,
        gamma_input=0.0,
        skip=0,
        final_nonlinearity=True,
        bias=False,
        momentum=0.1,
        pad_input=True,
        batch_norm=True,
        hidden_dilation=1,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        stack=None,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see mlutils.regularizers)
            skip:           Adds a skip connection
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            bias:           Adds a bias layer. Note: bias and batch_norm can not both be true
            momentum:       BN momentum
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            normalize_laplace_regularizer: Boolean, if set to True, will use the LaplaceL2norm function from
                mlutils.regularizers, which returns the regularizer as |laplace(filters)| / |filters|
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            stack = -1 will only select the last layer as the readout layer
                            stack = 0  will only readout from the first layer
        """

        super().__init__()

        assert not bias or not batch_norm, "bias and batch_norm should not both be true"

        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)

        self.layers = layers
        self.gamma_input = gamma_input
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.features = nn.Sequential()
        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [range(self.layers)[stack]] if isinstance(stack, int) else stack

        # --- first layer
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            input_channels, hidden_channels, input_kern, padding=input_kern // 2 if pad_input else 0, bias=bias
        )
        if batch_norm:
            layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if layers > 1 or final_nonlinearity:
            layer["nonlin"] = nn.ELU(inplace=True)
        self.features.add_module("layer0", nn.Sequential(layer))

        # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):

        # --- other layers
        h_pad = ((hidden_kern - 1) * hidden_dilation + 1) // 2
        for l in range(1, self.layers):
            layer = OrderedDict()
            layer["ds_conv"] = DepthSeparableConv2d(hidden_channels,hidden_channels, hidden_kern, padding=h_pad,bias=False)
            if batch_norm:
                layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            if final_nonlinearity or l < self.layers - 1:
                layer["nonlin"] = nn.ELU(inplace=True)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1))
            if l in self.stack:
                ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def regularizer(self):
        return self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels


class PointPooled2dReadout(nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, pool_steps, pool_kern, bias, init_range, gamma_readout):
        # super init to get the _module attribute
        super(PointPooled2dReadout, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, PointPooled2d(
                            in_shape,
                            n_neurons,
                            pool_steps=pool_steps,
                            pool_kern=pool_kern,
                            bias=bias,
                            init_range=init_range)
                            )

        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)


    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultiGaussReadout(nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_mu_range, init_sigma_range, bias, gamma_readout):
        # super init to get the _module attribute
        super(MultiGaussReadout, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, Gaussian2d(
                            in_shape=in_shape,
                            outdims=n_neurons,
                            init_mu_range=init_mu_range,
                            init_sigma_range=init_sigma_range,
                            bias=True)
                            )

        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


def ds_core_gauss_readout(dataloaders, seed, hidden_channels=32, input_kern=13,          # core args
                                 hidden_kern=3, layers=3,  gamma_input=0.1,
                                 skip=0, final_nonlinearity=True, momentum=0.9,
                                 pad_input=False, batch_norm=True, hidden_dilation=1,
                                 laplace_padding=None, input_regularizer='LaplaceL2norm',
                                 init_mu_range=0.2, init_sigma_range=0.5, readout_bias=True,  # readout args,
                                 gamma_readout=4,  elu_offset=0, stack=None,
                                 ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v['targets'][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v['inputs'] for k, v in session_shape_dict.items()}
    input_channels = [v['inputs'][1] for _, v in session_shape_dict.items()]
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    # get a stacked2D core from mlutils
    core = DepthSeparableCore(input_channels=input_channels[0],
                         hidden_channels=hidden_channels,
                         input_kern=input_kern,
                         hidden_kern=hidden_kern,
                         layers=layers,
                         gamma_input=gamma_input,
                         skip=skip,
                         final_nonlinearity=final_nonlinearity,
                         bias=False,
                         momentum=momentum,
                         pad_input=pad_input,
                         batch_norm=batch_norm,
                         hidden_dilation=hidden_dilation,
                         laplace_padding=laplace_padding,
                         input_regularizer=input_regularizer,
                         stack=stack)

    readout = MultiGaussReadout(core, in_shape_dict=in_shapes_dict,
                                   n_neurons_dict=n_neurons_dict,
                                   init_mu_range=init_mu_range,
                                   bias=readout_bias,
                                   init_sigma_range=init_sigma_range,
                                   gamma_readout=gamma_readout)

    # initializing readout bias to mean response
    for k in dataloaders:
        readout[k].bias.data = dataloaders[k].dataset[:].targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def ds_core_point_readout(dataloaders, seed, hidden_channels=32, input_kern=13,          # core args
                                 hidden_kern=3, layers=3, gamma_input=0.1,
                                 skip=0, final_nonlinearity=True, core_bias=False, momentum=0.9,
                                 pad_input=False, batch_norm=True, hidden_dilation=1,
                                 laplace_padding=None, input_regularizer='LaplaceL2norm',
                                 pool_steps=2, pool_kern=3, readout_bias=True,  # readout args,
                                 init_range=0.2, gamma_readout=0.1,  elu_offset=0, stack=None,
                                 ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v['targets'][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v['inputs'] for k, v in session_shape_dict.items()}
    input_channels = [v['inputs'][1] for _, v in session_shape_dict.items()]
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    # get a stacked2D core from mlutils
    core = DepthSeparableCore(input_channels=input_channels[0],
                         hidden_channels=hidden_channels,
                         input_kern=input_kern,
                         hidden_kern=hidden_kern,
                         layers=layers,
                         gamma_input=gamma_input,
                         skip=skip,
                         final_nonlinearity=final_nonlinearity,
                         bias=core_bias,
                         momentum=momentum,
                         pad_input=pad_input,
                         batch_norm=batch_norm,
                         hidden_dilation=hidden_dilation,
                         laplace_padding=laplace_padding,
                         input_regularizer=input_regularizer,
                         stack=stack)

    readout = PointPooled2dReadout(core, in_shape_dict=in_shapes_dict,
                                   n_neurons_dict=n_neurons_dict,
                                   pool_steps=pool_steps,
                                   pool_kern=pool_kern,
                                   bias=readout_bias,
                                   gamma_readout=gamma_readout,
                                   init_range=init_range)

    # initializing readout bias to mean response
    for k in dataloaders:
        readout[k].bias.data = dataloaders[k].dataset[:].targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def stacked2d_core_gaussian_readout(dataloaders, seed, hidden_channels=32, input_kern=13,          # core args
                                 hidden_kern=3, layers=3, gamma_hidden=0, gamma_input=0.1,
                                 skip=0, final_nonlinearity=True, core_bias=False, momentum=0.9,
                                 pad_input=False, batch_norm=True, hidden_dilation=1,
                                 laplace_padding=None, input_regularizer='LaplaceL2norm',
                                 readout_bias=True, init_mu_range=0.2, init_sigma_range=0.5,  # readout args,
                                 gamma_readout=0.1,  elu_offset=0, stack=None,
                                 ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v['targets'][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v['inputs'] for k, v in session_shape_dict.items()}
    input_channels = [v['inputs'][1] for _, v in session_shape_dict.items()]
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    # get a stacked2D core from mlutils
    core = Stacked2dCore(input_channels=input_channels[0],
                         hidden_channels=hidden_channels,
                         input_kern=input_kern,
                         hidden_kern=hidden_kern,
                         layers=layers,
                         gamma_hidden=gamma_hidden,
                         gamma_input=gamma_input,
                         skip=skip,
                         final_nonlinearity=final_nonlinearity,
                         bias=core_bias,
                         momentum=momentum,
                         pad_input=pad_input,
                         batch_norm=batch_norm,
                         hidden_dilation=hidden_dilation,
                         laplace_padding=laplace_padding,
                         input_regularizer=input_regularizer,
                         stack=stack)

    readout = MultiGaussReadout(core, in_shape_dict=in_shapes_dict,
                                   n_neurons_dict=n_neurons_dict,
                                   init_mu_range=init_mu_range,
                                   init_sigma_range=init_sigma_range,
                                   bias=readout_bias,
                                   gamma_readout=gamma_readout)

    # initializing readout bias to mean response
    for k in dataloaders:
        readout[k].bias.data = dataloaders[k].dataset[:].targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model



def vgg_core_gauss_readout(dataloaders, seed,
                           input_channels=1, tr_model_fn='vgg16', # begin of core args
                           model_layer=11, momentum=0.1, final_batchnorm=True,
                           final_nonlinearity=True, bias=False,
                           init_mu_range=0.4, init_sigma_range=0.6, readout_bias=True, # begin or readout args
                           gamma_readout=0.002, elu_offset=-1):
    """
    A Model class of a predefined core (using models from torchvision.models). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v['targets'][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v['inputs'] for k, v in session_shape_dict.items()}
    input_channels = [v['inputs'][1] for _, v in session_shape_dict.items()]
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    class Encoder(nn.Module):
        """
        helper nn class that combines the core and readout into the final model
        """
        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.readout.regularizer(data_key=data_key) + self.core.regularizer()

    set_random_seed(seed)

    core = TransferLearningCore(input_channels=input_channels[0],
                                tr_model_fn=tr_model_fn,
                                model_layer=model_layer,
                                momentum=momentum,
                                final_batchnorm=final_batchnorm,
                                final_nonlinearity=final_nonlinearity,
                                bias=bias)

    readout = MultiGaussReadout(core, in_shape_dict=in_shapes_dict,
                                n_neurons_dict=n_neurons_dict,
                                init_mu_range=init_mu_range,
                                bias=readout_bias,
                                init_sigma_range=init_sigma_range,
                                gamma_readout=gamma_readout)

    # initializing readout bias to mean response
    for k in dataloaders:
        readout[k].bias.data = dataloaders[k].dataset[:].targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


class Gaussian2d(nn.Module):
    """
    Instantiates an object that can used to learn a point in the core feature space for each neuron,
    sampled from a Gaussian distribution with some mean and variance at train but set to mean at test time, that best predicts its response.
    The readout receives the shape of the core as 'in_shape', the number of units/neurons being predicted as 'outdims', 'bias' specifying whether
    or not bias term is to be used and 'init_range' range for initialising the mean and variance of the gaussian distribution from which we sample to
    uniform distribution, U(-init_range,init_range) and  uniform distribution, U(0.0, 3*init_range) respectively.
    The grid parameter contains the normalized locations (x, y coordinates in the core feature space) and is clipped to [-1.1] as it a
    requirement of the torch.grid_sample function. The feature parameter learns the best linear mapping between the feature
    map from a given location, sample from Gaussian at train time but set to mean at eval time, and the unit's response with or without an additional elu non-linearity.
    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]
        init_sigma_range (float): initialises sigma with Uniform([0.0, init_sigma_range])
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_mu_range,
        init_sigma_range,
        batch_sample=True,
        **kwargs
    ):

        super().__init__()
        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma_range <= 0.0:
            raise ValueError(
                "either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive"
            )
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.batch_sample = batch_sample
        self.grid_shape = (1, outdims, 1, 2)
        self.mu = Parameter(
            torch.Tensor(*self.grid_shape)
        )  # mean location of gaussian for each neuron
        self.sigma = Parameter(
            torch.Tensor(*self.grid_shape)
        )  # standard deviation for gaussian for each neuron
        self.features = Parameter(
            torch.Tensor(1, c, 1, outdims)
        )  # feature weights for each channel of the core

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.init_sigma_range = init_sigma_range
        self.initialize()

    def initialize(self):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        self.mu.data.uniform_(-self.init_mu_range, self.init_mu_range)
        self.sigma.data.uniform_(0, self.init_sigma_range)
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def sample_grid(self, batch_size, sample=True):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        Args:
            batch_size (int): size of the batch
            sample (bool): sample determines whether to draw a sample or use the mean of the Gaussian distribution per neuron.
        """
        with torch.no_grad():
            self.mu.clamp_(
                min=-1, max=1
            )  # at eval time, only self.mu is used so it must belong to [-1,1]
            self.sigma.clamp_(min=0)  # sigma/variance is always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]

        if self.training and sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()

        return torch.clamp(norm * self.sigma + self.mu, min=-1,
                           max=1)  # grid locations in feature space sampled randomly around the mean self.mu

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    def feature_l1(self, average=True):
        """
        returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def forward(self, x, sample=True, shift=None, out_idx=None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample: sample determines whether to draw a sample or use the mean of the Gaussian distribution per neuron.
            shift: shifts the location of the grid (from eye-tracking data)
            out_idx: index of neurons to be predicted
        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError(
                "the specified feature map dimension is not the readout's expected input dimension"
            )
        feat = self.features.view(1, c, self.outdims)
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(
                batch_size=N, sample=sample
            )  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(
                N, outdims, 1, 2
            )

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = (
            self.__class__.__name__
            + " ("
            + "{} x {} x {}".format(c, w, h)
            + " -> "
            + str(self.outdims)
            + ")"
        )
        if self.bias is not None:
            r += " with bias"
        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r