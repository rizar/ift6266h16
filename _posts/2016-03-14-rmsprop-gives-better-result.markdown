---
layout: post
title:  "RMSProp improves my result to 77%!"
date:   2016-03-13 21:08:37
categories: jekyll update
---

Hooray! I averaged the validation error rate over the
last epochs, because it was pretty unstable in the end of training.

I used the learning rate 0.0003, using 0.001 led to divergence.
But, still I underfit quite a bit.
![error_rate]({{site.baseurl}}/downloads/rmsprop_error_rate.png)

Interestingly, gradient norm was only growing in the course of the training:
![gradient_norm]({{site.baseurl}}/downloads/rmsprop_gradient_norm.png)
