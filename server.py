#!/usr/bin/env python

import sys
import numpy
import math

from fuel.datasets import DogsVsCats
from fuel.schemes import ShuffledScheme
from fuel.server import start_server
from fuel.streams import DataStream
from fuel.transformers import ScaleAndShift, ForceFloatX
from fuel.transformers.image import (
    RandomFixedSizeCrop,
    SourcewiseTransformer, ExpectsAxisLabels)

from PIL import Image

class ForceMinimumDimension(SourcewiseTransformer, ExpectsAxisLabels):
    def __init__(self, data_stream, min_dim, resample='nearest',
                 **kwargs):
        self.min_dim = min_dim
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(ForceMinimumDimension, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        original_min_dim = min(example.shape[-2:])
        multiplier = self.min_dim / float(original_min_dim)
        dt = example.dtype
        # If we're dealing with a colour image, swap around the axes
        # to be in the format that PIL needs.
        if example.ndim == 3:
            im = example.transpose(1, 2, 0)
        else:
            im = example
        im = Image.fromarray(im)
        width, height = im.size
        width = int(math.ceil(width * multiplier))
        height = int(math.ceil(height * multiplier))
        im = numpy.array(im.resize((width, height))).astype(dt)
        # If necessary, undo the axis swap from earlier.
        if im.ndim == 3:
            example = im.transpose(2, 0, 1)
        else:
            example = im
        return example


def add_transformers(stream, random_crop=False):
    stream = ForceMinimumDimension(stream, 128,
                                   which_sources=['image_features'])
    if random_crop:
        stream = RandomFixedSizeCrop(stream, (128, 128),
                                    which_sources=['image_features'])
    stream = ScaleAndShift(stream, 1 / 255.0, 0,
                           which_sources=['image_features'])
    stream = ForceFloatX(stream)
    return stream

if __name__ == '__main__':
    train = DogsVsCats(("train",), subset=slice(None, int(sys.argv[1]), None))
    train_str =  DataStream(
        train, iteration_scheme=ShuffledScheme(train.num_examples, int(sys.argv[2])))
    train_str = add_transformers(train_str, random_crop=True)
    start_server(train_str)
