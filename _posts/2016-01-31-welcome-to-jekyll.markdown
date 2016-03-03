---
layout: post
title:  "Let's get it started!"
date:   2016-01-31 21:08:37
categories: jekyll update
---

As a requirement for the course IFT6266h16, this blog will document my struggles with the course project.
The task is to classify images of cats (meaw-meaw!) against images of dogs (woof-woof!). I will start with 
a quick analysis of the dataset.

I have downloaded the zip files and converted them to HDF5 using Fuel. The sad
news is that the size of `dogs_vs_cats.hdf5` is 17 gigabytes, whereas the zip archives
for the training and testing were together less than a gig. This suggests that HDF5 contains uncompressed bitmaps. A quick look into the [code](https://github.com/mila-udem/fuel/blob/master/fuel/converters/dogs_vs_cats.py#L87) confirms this hypothesis. 

All right, let's look inside

{% highlight ipython %}
In [6]: dvc = DogsVsCats(['train'])
In [7]: dvc.num_examples
Out[7]: 25000
# This is 2 times less than MNIST, which let's me hope
# that training won't take that long.
In [12]: shapes = [data[0].shape for data in dvc.get_example_stream().get_epoch_iterator()]
In [13]: min_dimension = [min(x, y) for d, x, y in shapes]
In [17]: pyplot.hist(min_dimension)
# Let's see what are the sizes of the images
{% endhighlight %}
The result is below: 
![min_dims]({{site.baseurl}}/downloads/shapes.png)
<!-- ![min_dims](https://upload.wikimedia.org/wikipedia/en/e/ec/Lisa_Simpson.png) -->

[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help
