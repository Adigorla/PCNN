from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PST_func import PST
from numpy_pst import ground_PST
import mahotas as mh
import tensorflow as tf
import numpy as np
from tensorflow.python import roll as _roll
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export

def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.
    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
    Parameters
    ----------
    x : array_like, Tensor
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.
    Returns
    -------
    y : Tensor.
    """
    x = ops.convert_to_tensor_v2(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    return _roll(x, shift, axes)

if __name__ == "__main__":
    LPF = 0.21
    Phase_strength = 0.48 
    Warp_strength= 12.14
# Thresholding parameters (for post processing after the edge is computed)
    Threshold_min = -1
    Threshold_max = 0.0019
    x = mh.imread('./lena.jpg')
   #print('{}'.format(x.shape))
    test=x[8:11,8:11,1]
    #print('{}'.format(test))
    out = ground_PST(test, LPF, Phase_strength, Warp_strength, Threshold_min, Threshold_max, 0)
    print('After ground: {}'.format(out))
    img2 = PST(test, LPF, Phase_strength, Warp_strength, Threshold_min, Threshold_max)
    print('After tensor {}'.format(img2)) 
    #mh.imsave('./shift.jpg', img)