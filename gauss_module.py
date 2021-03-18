# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2d


"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    # Generate a vector x of values on which the Gaussian filter is defined: integer values on the interval [-3*sigma, 3*sigma]
    low = int(-3*sigma)
    high = int((3*sigma))
    range_x = [i for i in range(low, high+1)]
    Gx = []
    for x in range_x:
        G = (np.exp((-x ** 2) / (2 * (sigma ** 2))))* (1 / (math.sqrt(2 * math.pi)* sigma))  # Formula for gaussian
        Gx.append(G)

    # Gx is a list in which we have values calculated by gaussian formula
    return np.array(Gx), range_x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):

    # fill the kernel with the values on which the G filter is defined (range [-3*sigma,3*sigma])
    # extact the first row of the kernel
    Gx = gauss(sigma)[0]

    Gx = Gx.reshape(1, Gx.size)   # This was done because input to the conv2d need 2d


    # # computing the first convolution
    tmp_img = conv2d(img, Gx, mode='full', boundary='fill', fillvalue=0,)
    # now tmp_img (the output of the first convolution) is an array



    Gy = np.transpose(Gx)

    # computing the second convolution (on the output of the first one)
    smooth_img = conv2d(tmp_img, Gy, mode='full', boundary='fill', fillvalue=0)
    # # SOURCE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

    return smooth_img

"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    low = int(-3*sigma)
    high = int((3*sigma))
    range_x = [i for i in range(low, high+1)]
    Dx = []
    for x in range_x:
        # Just taken derivative of gaussian formula formula
        G = -x*(np.exp((-x ** 2) / (2 * (sigma ** 2))))* (1 / (math.sqrt(2 * math.pi)* (sigma**3)))
        Dx.append(G)
    return np.array(Dx), range_x
#
def gaussderiv(img, sigma):

    smooth_img = gaussianfilter(img, sigma)   # Call gaussian filter function to get smoothen image

    Dx =  np.array( [[-1,0,1],   # Kernal which we will use to take derivative in x direction
             [-1,0,1],
             [-1,0,1]])
    Dx = Dx/3

    Dy = np.transpose(Dx)        # Kernal which we will use to take derivative in y direction

    imgDx = conv2d(smooth_img, Dx, mode='full', boundary='fill', fillvalue=0)  # convolving with smoothen image in X direction

    imgDy = conv2d(smooth_img, Dy, mode='full', boundary='fill', fillvalue=0)  # convolving with smoothen image in Y direction

    return imgDx, imgDy
