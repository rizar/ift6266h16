---
layout: post
title:  "First result: 71.5% accuracy!"
date:   2016-03-04 21:08:37
categories: jekyll update
---

I have the first convnet experiment in my life! Hooray!

As a developer of Blocks&Fuel I could not stand the temptation to use the most out of 
them. I started my project from the [standard Blocks convnet example](https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/__init__.py).

The network in this example is very similar to the famous LeNet. It consists of:

- a convolutional layer with 20 filters, each filter is 5x5
- max pooling with step 2
- a convolutional layer with 50 filters, each filter is 5x5
- max pooling with step 2
- a fully connected layer with 500 units
- an softmax output layer with 2 units

Rectifier units are used throughout the network. In this regard it is quite different
from the original LeNet, since the latter used saturating units.

As suggested in one of the previous posts, I used random 128x128 crops during
training time. For validation I used the technique that I know from 
[AlexNet paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). Namely, I average the predictions of the network
over 5 windows, 4 from corners and 1 one from the center of the image.
[Here](https://github.com/rizar/ift6266h16/blob/master/main.py#L197) is the implementation.

If I were to choose one word to describe this experiment, the would be _underfitting_.
The training set performance never went significantly above 70%. Using 5 windows for 
prediction for the validation gave me 4-5% additional percent of accuracy. This is an expected an quite significant improvement.

TODO: post the training curves






