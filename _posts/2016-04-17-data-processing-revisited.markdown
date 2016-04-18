---
layout: post
title:  "Data processing revisited - 83.6%"
date:   2016-04-17 12:18:00
categories: jekyll update
---

I have realized that my current data processing method is non-sense. Let's take again
a quick look at the histogram of the minimal dimension:

![min_dims]({{site.baseurl}}/downloads/shapes.png)

For most images *both* dimensions are more than 300. This means that by choosing
a 128x128 window I focus only on a small part of the image.

I tried a new preprocessing, which resizes the images to have a minimal dimension
of strictly 128. The results for my LeNet are shown on the picture below (10c is with the old prerocessing, 17 is with the new one). Now I finally observe the overfitting one is supposed
to suffer on a dataset of only 25k examples! However, the old preprocessing had
a strong data augmentation effect, which I think I lack with the new one. I think
I should keep margins along both dimensions, much like they do it in AlexNet paper:
their network expects 224x224 image, but they crop it from a 256x256 one.

![new_preproc]({{site.baseurl}}/downloads/new_preprocessing.png)
