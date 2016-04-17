#!/usr/bin/env python
"""Convolutional network example.

Run the training for 50 epochs with
```
python __init__.py --num-epochs 50
```
It is going to reach around 0.8% error rate on the test set.

"""
from __future__ import print_function

import sys
import logging
import numpy
import os
import subprocess
from argparse import ArgumentParser

import theano
from theano import tensor

from blocks.algorithms import (
    GradientDescent, RMSProp, Adam,
    Restrict, CompositeRule, VariableClipping)
from blocks.bricks.base import application
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.simple import Linear
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.serialization import load_parameters
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.roles import OUTPUT, WEIGHT
from blocks.theano_expressions import l2_norm
from blocks.initialization import NdarrayInitialization
from fuel.datasets import DogsVsCats
from fuel.schemes import ShuffledScheme, SequentialExampleScheme
from fuel.streams import DataStream, ServerDataStream

from server import add_transformers

logger = logging.getLogger(__name__)


class Glorot(NdarrayInitialization):
    def generate(self, rng, shape):
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
        half_width = (6. / (fan_in + fan_out)) ** 0.5
        m = rng.uniform(-half_width, half_width, size=shape)
        return m.astype(theano.config.floatX)


class LeNet(FeedforwardSequence, Initializable):
    """LeNet-like convolutional network.

    The class implements LeNet, which is a convolutional sequence with
    an MLP on top (several fully-connected layers). For details see
    [LeCun95]_.

    .. [LeCun95] LeCun, Yann, et al.
       *Comparison of learning algorithms for handwritten digit
       recognition.*,
       International conference on artificial neural networks. Vol. 60.

    Parameters
    ----------
    conv_activations : list of :class:`.Brick`
        Activations for convolutional network.
    num_channels : int
        Number of channels in the input image.
    image_shape : tuple
        Input image shape.
    filter_sizes : list of tuples
        Filter sizes of :class:`.blocks.conv.ConvolutionalLayer`.
    feature_maps : list
        Number of filters for each of convolutions.
    pooling_sizes : list of tuples
        Sizes of max pooling for each convolutional layer.
    repeat_times : list of int
        How many times to repeat each convolutional layer.
    top_mlp_activations : list of :class:`.blocks.bricks.Activation`
        List of activations for the top MLP.
    top_mlp_dims : list
        Numbers of hidden units and the output dimension of the top MLP.
    stride : int
        Step of convolution for the first layer, 1 will be used
        for all other layers.
    border_mode : str
        Border mode of convolution (similar for all layers).

    """
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes, repeat_times,
                 top_mlp_activations, top_mlp_dims,
                 stride, border_mode='valid', **kwargs):
        self.stride = stride
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        # Construct convolutional layers with corresponding parameters
        self.layers = []
        for i, activation in enumerate(conv_activations):
            for j in range(repeat_times[i]):
                self.layers.append(
                    Convolutional(
                        filter_size=filter_sizes[i], num_filters=feature_maps[i],
                        step=(1, 1) if i > 0 or j > 0 else (self.stride, self.stride),
                        border_mode=self.border_mode,
                        name='conv_{}_{}'.format(i, j)))
                self.layers.append(activation)
            self.layers.append(MaxPooling(pooling_sizes[i], name='pool_{}'.format(i)))

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims

    @application(inputs=['image'])
    def apply_5windows(self, image):
        width, height = self.image_shape
        # the dimension 0 stands for the
        hor_offset = (image.shape[1] - width) / 2
        ver_offset = (image.shape[2] - height) / 2
        x_views = tensor.concatenate(
            [image[None, :, :width, :height],
             image[None, :, :width, -height:],
             image[None, :, -width:, :height],
             image[None, :, -width:, -height:],
             image[None, :,
                   hor_offset:hor_offset + width, ver_offset:ver_offset + height]],
             axis=0)
        return self.apply(x_views).mean(axis=0)[None, :]


def main(mode, save_to, num_epochs, load_params=None,
         feature_maps=None, mlp_hiddens=None,
         conv_sizes=None, pool_sizes=None, stride=None, repeat_times=None,
         batch_size=None, num_batches=None, algo=None,
         test_set=None, valid_examples=None,
         dropout=None, max_norm=None, weight_decay=None):
    if feature_maps is None:
        feature_maps = [20, 50, 50]
    if mlp_hiddens is None:
        mlp_hiddens = [500]
    if conv_sizes is None:
        conv_sizes = [5, 5, 5]
    if pool_sizes is None:
        pool_sizes = [2, 2, 2]
    if repeat_times is None:
        repeat_times = [1, 1, 1]
    if batch_size is None:
        batch_size = 500
    if valid_examples is None:
        valid_examples = 2500
    if stride is None:
        stride = 1
    if test_set is None:
        test_set = 'test'
    if algo is None:
        algo = 'rmsprop'

    image_size = (128, 128)
    output_size = 2

    if (len(feature_maps) != len(conv_sizes) or
        len(feature_maps) != len(pool_sizes) or
        len(feature_maps) != len(repeat_times)):
        raise ValueError("OMG, inconsistent arguments")

    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 3, image_size,
                    stride=stride,
                    filter_sizes=zip(conv_sizes, conv_sizes),
                    feature_maps=feature_maps,
                    pooling_sizes=zip(pool_sizes, pool_sizes),
                    repeat_times=repeat_times,
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='full',
                    weights_init=Glorot(),
                    biases_init=Constant(0))
    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.initialize()
    logging.info("Input dim: {} {} {}".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))


    single_x = tensor.tensor3('image_features')
    x = tensor.tensor4('image_features')
    single_y = tensor.lvector('targets')
    y = tensor.lmatrix('targets')

    # Training
    probs = convnet.apply(x)
    cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
            .copy(name='cost'))
    error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                  .copy(name='error_rate'))

    cg = ComputationGraph([cost, error_rate])
    if dropout:
        relu_outputs = VariableFilter(bricks=[Rectifier], roles=[OUTPUT])(cg)
        cg = apply_dropout(cg, relu_outputs, dropout)
        cost, error_rate = cg.outputs
    if weight_decay:
        logger.debug("Apply weight decay {}".format(weight_decay))
        cost += weight_decay * l2_norm(cg.parameters)
        cost.name = 'cost'

    # Validation
    valid_probs = convnet.apply_5windows(single_x)
    valid_cost = (CategoricalCrossEntropy().apply(single_y, valid_probs)
            .copy(name='cost'))
    valid_error_rate = (MisclassificationRate().apply(
        single_y, valid_probs).copy(name='error_rate'))

    model = Model([cost, error_rate])
    if load_params:
        logger.info("Loaded params from {}".format(load_params))
        with open(load_params, 'r') as src:
            model.set_parameter_values(load_parameters(src))

    # Training stream with random cropping
    train = DogsVsCats(("train",), subset=slice(None, 25000 - valid_examples, None))
    train_str =  DataStream(
        train, iteration_scheme=ShuffledScheme(train.num_examples, batch_size))
    train_str = add_transformers(train_str, random_crop=True)

    # Validation stream without cropping
    valid = DogsVsCats(("train",), subset=slice(25000 - valid_examples, None, None))
    valid_str = DataStream(
        valid, iteration_scheme=SequentialExampleScheme(valid.num_examples))
    valid_str = add_transformers(valid_str)

    if mode == 'train':
        directory, _ = os.path.split(sys.argv[0])
        env = dict(os.environ)
        env['THEANO_FLAGS'] = 'floatX=float32'
        port = numpy.random.randint(1025, 10000)
        server = subprocess.Popen(
            [directory + '/server.py',
             str(25000 - valid_examples), str(batch_size), str(port)],
            env=env, stderr=subprocess.STDOUT)
        train_str = ServerDataStream(
            ('image_features', 'targets'), produces_examples=False,
            port=port)

        save_to_base, save_to_extension = os.path.splitext(save_to)

        # Train with simple SGD
        if algo == 'rmsprop':
            step_rule = RMSProp(decay_rate=0.999, learning_rate=0.0003)
        elif algo == 'adam':
            step_rule = Adam()
        else:
            assert False
        if max_norm:
            conv_params = VariableFilter(bricks=[Convolutional], roles=[WEIGHT])(cg)
            linear_params = VariableFilter(bricks=[Linear], roles=[WEIGHT])(cg)
            step_rule = CompositeRule(
                [step_rule,
                 Restrict(VariableClipping(max_norm, axis=0), linear_params),
                 Restrict(VariableClipping(max_norm, axis=(1, 2, 3)), conv_params)])

        algorithm = GradientDescent(
            cost=cost, parameters=model.parameters,
            step_rule=step_rule)
        # `Timing` extension reports time for reading data, aggregating a batch
        # and monitoring;
        # `ProgressBar` displays a nice progress bar during training.
        extensions = [Timing(every_n_batches=100),
                    FinishAfter(after_n_epochs=num_epochs,
                                after_n_batches=num_batches),
                    DataStreamMonitoring(
                        [valid_cost, valid_error_rate],
                        valid_str,
                        prefix="valid"),
                    TrainingDataMonitoring(
                        [cost, error_rate,
                        aggregation.mean(algorithm.total_gradient_norm)],
                        prefix="train",
                        after_epoch=True),
                    TrackTheBest("valid_error_rate"),
                    Checkpoint(save_to, save_separately=['log'],
                                before_training=True, after_epoch=True)
                        .add_condition(
                            ['after_epoch'],
                            OnLogRecord("valid_error_rate_best_so_far"),
                            (save_to_base + '_best' + save_to_extension,)),
                    Printing(every_n_batches=100)]

        model = Model(cost)

        main_loop = MainLoop(
            algorithm,
            train_str,
            model=model,
            extensions=extensions)
        try:
            main_loop.run()
        finally:
            server.terminate()
    elif mode == 'test':
        classify = theano.function([single_x], valid_probs.argmax())
        test = DogsVsCats((test_set,))
        test_str = DataStream(
            test, iteration_scheme=SequentialExampleScheme(test.num_examples))
        test_str = add_transformers(test_str)
        correct = 0
        with open(save_to, 'w') as dst:
            print("id", "label", sep=',', file=dst)
            for index, example in enumerate(test_str.get_epoch_iterator()):
                image = example[0]
                prediction = classify(image)
                print(index + 1, classify(image), sep=',', file=dst)
                if len(example) > 1 and prediction == example[1]:
                    correct += 1
        print(correct / float(test.num_examples))
    else:
        assert False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = ArgumentParser("An example of training a convolutional network "
                            "on the MNIST dataset.")
    parser.add_argument("mode",
                        help="What to do", choices=['train', 'test'])
    parser.add_argument("save_to",
                        help="Destination to save the state of the training "
                             "process.")

    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("--load-params", help="Path to load parameters from")
    parser.add_argument("--batch-size", type=int,
                        help="Batch size.")
    parser.add_argument("--algo", choices=['rmsprop', 'adam'],
                        help="The algorithm to use.")
    parser.add_argument("--dropout", type=float,
                        help="Dropout coefficient")
    parser.add_argument("--max-norm", type=float,
                        help="Dropout coefficient")
    parser.add_argument("--weight-decay", type=float,
                        help="Weight decay coefficient")

    parser.add_argument("--test-set", type=str)
    parser.add_argument("--valid-examples", type=int)

    parser.add_argument("--stride", type=int,
                        help="Stride for the first layer")
    parser.add_argument("--feature-maps", type=int, nargs='+',
                        help="List of feature maps numbers.")
    parser.add_argument("--mlp-hiddens", type=int, nargs='+',
                        help="List of numbers of hidden units for the MLP.")
    parser.add_argument("--conv-sizes", type=int, nargs='+',
                        help="Convolutional kernels sizes. The kernels are "
                        "always square.")
    parser.add_argument("--pool-sizes", type=int, nargs='+',
                        help="Pooling sizes. The pooling windows are always "
                             "square. Should be the same length as "
                             "--conv-sizes.")
    parser.add_argument("--repeat-times", type=int, nargs='+',
                        help="Number of times to repeat each conv. layer")

    args = parser.parse_args()
    main(**vars(args))
