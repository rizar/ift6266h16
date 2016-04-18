---
layout: post
title:  "Regularization and summary - 94.9%"
date:   2016-04-18 11:12:00
categories: jekyll update
---

I have three models that I can not decide from: 4-layer LeNet, and 6- and 8-layer VGG. All of them overfit.
Let the regularization decide who is the fittest!

In my preliminary experiments I tried using dropout in conjuction with weight decay or max norm clipping, 
like used in [the original dropout tech-report](http://arxiv.org/abs/1207.0580). The results were often 
quite good, but in general very hard to predict. Sometimes dropout did not help at all. Dropout with 
a too high max norm constraint simply did not want to train! 

In the end I figured that for the models I care about I will try the following:

- pure dropout 
- dropout with column norm constraint of 1 

Quite expectedly, the training with dropout took much longer. Here are the results:

- 8 layer VGG performance improved from 91.9% to 94.9% with pure dropout
- 6 layer VGG performance improved from XXX to YYY with pure dropout
- 4 layer LeNet performance improved from XXX to YYY with pure dropout

Currently, 8 layer VGG is my champion with a very small margin above less regularized 6-layer VGG.
Its performance is 94.5% on the test and 96.4% on the full training set. This rather small margin
suggests that I could try to increase the capacity of the network layers. Or maybe use
advanced optimization techniques (momentum, annealing), to fine-tune the network.
