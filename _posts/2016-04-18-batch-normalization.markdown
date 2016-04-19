---
layout: post
title:  "Batch normalization - 95.9%"
date:   2016-04-18 22:00:00
categories: jekyll update
---

A big issue for my latest training experiments was the training speed. When an
8-layer VGG was trained with dropout, even 8 epochs did not really seems
sufficient. This is exactly the context in which [Batch
normalization](http://arxiv.org/abs/1502.03167) was invented. Batch
normalization is a technique that allows to speed up the training a lot 
and often drop regularization. It works as follows:

- for each minibatch, the inputs of each unit are shifted and rescaled to look
  like if they are coming from a Gaussian $$\mathcal{N}(0, 1)$$

- the transformed inputs are shifted and scaled again, but this time with
  trainable scaling and shift parameters $$\gamma$$ and $$\beta$$ respectively.
  This make the next layer "feel" that its inputs are coming from a distribution
  $$\mathcal{N}(\beta, \gamma)$$ that it prefers.

- running averages of per-batch statistics are kept to become a part of the
  final trained network 

I tried two variants of batch norm: mean-only version, that only shifts the
data, and the full one. The results are... overwhelming, let's say:

![batch_norm1]({{site.baseurl}}/downloads/batch_norm1.png)

BOOM, applying batch normalization to my 8-layer VGG network brings me almost 
as far as additional data augmentation combined with dropout do. It is
interesting however, that in the very beginning of the training the batch
normalized network performs worse. Here is my understanding what's going on:
with my rather small batch size of 100 a lot of noise is introduced into the training
procedure. This is a great regularizer, but it also hurts the normalization
machinery a bit.

Finally, I tried to combine batch normalization with my data augmenation and
dropout:

![batch_norm1]({{site.baseurl}}/downloads/batch_norm2.png)

The mean-only batch norm feels much better in the presence of random cropping
and provides me with the new champion (95.9% on the validation set, 95.2% on
the test set). Adding dropout slows down training a lot for the mean-only
version and drives the full batch norm crazy.
