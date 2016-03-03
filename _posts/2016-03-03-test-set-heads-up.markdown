---
layout: post
title:  "Don't use test set for Cats&Dogs in Fuel!"
date:   2016-01-31 21:08:37
categories: jekyll update
---

What a surprise - it turns out the labels for the test are not provided!
There was an error in Fuel converter, due to which all images from the 
test set were tagged as dog: see 
[the issue](https://github.com/mila-udem/fuel/issues/322) 
and the PRs that fix it: 
[one](https://github.com/mila-udem/fuel/pull/323)
and [two](https://github.com/mila-udem/fuel/pull/324). 

Hopefully I can still evaluate my test set prediction using Kaggle. For now, 
I will just use last 5000 examples as a validation set.
