---
title:  "Computer Vision/Machine Learning coding interviews Part-2"
mathjax: true
layout: post
categories: media
---

Over the past 7 years, I have had the chance to give multiple interviews for the computer vision and machine learning roles. This post is to delve down into the common questions asked in these interviews. These are in general 45-60 minute interviews asked with a real world problem in mind. This is part-2 of the interview coding questions.


# Question - Given an input layer of (NCHW) and a conv kernel of size of (FCkHkW). Write a conv2d layer in python using only Numpy



{% highlight python %}

"""
    Perform 2D convolution on an input tensor (N, C, H, W)
    
    Args:
        weight (np.ndarray): Filter weights of shape (F, C, KH, KW)
"""

import numpy as np


# the output will be of size:
# outputH = (H + 2*padding - kH)//stride + 1
# outputW = (W + 2*padding - kW)//stride + 1
# output layer is (F, C, outputH, outputW)

# to be clear F is the number of filters learnt of size CxkHxkW

This could be one of the easiest questions to get asked or your worst nightmare. Key is to get the fundamentals right. When I wanted to try to code this up in a non-interview scenario I was fumbling a lot on the dimensions and defintely not able to code it up in 30 minutes. Key here is to keep in mind the output dimensions and then iterate through. Also bias is (F,) shaped, meaning 1 per channel output for each kernel.

def conv2d(input_tensor, weight, padding=0, stride=1, bias=True):

    F, C, kH, kW = weight.shape

    N, C, H, W = input_tensor.shape

    oH = (H + 2*padding - kH)//stride + 1
    oW = (W + 2*padding - kW)//stride + 1

    #output is (batches, no. of filters with each in the channel, oH, oW)
    output = np.zeros((N, F,  oH, oW))

    # for each batch
    for n in range(N):
        #for each of the filters
        for f in range(F):
            #for each row
            for r in range(oH):
                # for each col
                for c in range(oW):
                    #skip for the stride
                    r_start = r*stride
                    c_start = c*stride
                    
                    # for image filtering like center pixel
                    #input = input_tensor[n, :, r_start-kH:r_start+kH, c_start-kW:c_start+kW]
                    # pytorch like implmenetations of covolutions have anchor on top left
                    input = input_tensor[n, :, r_start:r_start+kH, c_start:c_start+kW]
                    output[n, f, r, c] =  np.sum(weight[f, :, :, :]*input)

                    #if bias, then add bias term
            
            if bias:
                output[n, f, :, :] += bias[f]

    
    return output

{% endhighlight %}

