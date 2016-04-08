---
layout: post
title:  "Fixed bugs and increased the batch size"
date:   2016-04-08 10:35:00
categories: jekyll update
---

I know you were missing new posts for me, but now I am back! :)

Here are the main updates for the last week:

- Turns out my choice of the validation set was not good, it consisted of cats only!
  I switched to the one prescribed by Bart.

- I learnt how to use cluster, and now I can do a lot of hyperparameter search. 

- I dropped the fancy initialization inherited from Blocks MNIST example. The simpler the better.

- I realized, that I was using a way too small batch size. Increasing it from 10 to 100 made 
  training much faster, and I trained my LeNet two times longer.

My current results are: 

- The test error is 19.9%.

- The validation error is lower, it oscillates between 15% and 20%.

- The training error is about 20% when with random cropping and 18.2% without it. This means
  I do not overfit at all! I find this rather strange.

TO DO:

- More layers! Two convolutional layers is far from enough. 

- Train longer, the validation error keep improving.

- Save the best model (for now I am just using the last one).

And here is training curve:

![error_rate]({{site.baseurl}}/downloads/lenet_bs100.png)
