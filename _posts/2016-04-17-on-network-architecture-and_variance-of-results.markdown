---
layout: post
title:  "On network architecture and variance of results - 88.9%"
date:   2016-04-17 12:46:00
categories: jekyll update
---

How many convolutional and pooling layers should my network have?

Famous convnets from the literature are all built in such a way that the 
last convolutional layer produces an output with rather small width and height:

- 5x5 for LeNet
- 11x11 for AlexNet
- 7x7 for [VGG](http://arxiv.org/pdf/1409.1556.pdf)

My LeNet version is very similar to the original LeNet (the difference is that I have 
zero padding), but it has to process images of 128x128 instead of 32 x 32. This means 
that my current architecture does not have enough translation invariance and probably
also has way too many parameters. I should have much more subsampling, 
and possibly, more convolutional layers.

To check this hypothesis, I did something very simple: tried adding 1 and 2 more layers,
together with pooling and subsampling. The results are below. It is interesting, 
that adding the 3rd layer does not help as much as adding the 4th.

![deeper]({{site.baseurl}}/downloads/2layers_vs_3layers.png)

