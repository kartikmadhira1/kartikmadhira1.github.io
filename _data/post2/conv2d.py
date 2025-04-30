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