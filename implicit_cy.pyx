import numpy as np
cimport numpy as np
from scipy.special import lambertw
from libc.math cimport log, exp
import cython

cdef float a(float u_temp, float multiplier):
    """Returns solution to a as defined in the paper"""
    cdef float diff = multiplier - u_temp
    cdef float return_val
    if diff < -15:
        return_val = 0.0
    elif diff > 15:
        return_val = diff - log(diff)
    else:
        return_val = np.real(lambertw(exp(diff)))
    return return_val

@cython.cdivision(True)
cpdef float f1(float u_temp, float multiplier, float x_norm, float learning_rate, float u_old):
    """First derivative of u equation"""
    cdef float a_temp = a(u_temp, multiplier)
    cdef float return_value = (2 * learning_rate
            - 2 * learning_rate * exp(-u_temp)
            - a_temp / (1 + a_temp) / x_norm
            + 2 * (u_temp - u_old)
            - a_temp ** 2 / (1 + a_temp) / x_norm
            )
    return return_value