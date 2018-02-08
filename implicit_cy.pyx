import numpy as np
cimport numpy as np
from scipy.special import lambertw
from libc.math cimport log, exp
import cython

@cython.cdivision(True)
cpdef float f1(float u_temp, float multiplier, float x_norm, float learning_rate, float u_old):
    """First derivative of u equation"""
    cdef float diff = multiplier - u_temp
    cdef float a_temp = diff
    cdef float L1 = diff
    cdef float L2 = log(diff)

    if diff > 15:
        # Asymptotic expansion from equations (15-18) of
        # http://mathworld.wolfram.com/LambertW-Function.html
        a_temp = (L1
                  - L2
                  + L2 / L1
                  + L2 * (-2 + L2) / (2 * L1 ** 2)
                  + L2 * (6 - 9 * L2 + 2 * L2 ** 2) / (6 * L1 ** 3)
                  + L2 * (-12 + 36 * L2 - 22 * L2 ** 2 + 3 * L2 ** 3) / (12 * L1 ** 4)
                  )
    else:
        a_temp = lambertw(np.exp(diff)).real

    cdef float return_value = (2 * learning_rate
            - 2 * learning_rate * exp(-u_temp)
            - a_temp / (1 + a_temp) / x_norm
            + 2 * (u_temp - u_old)
            - a_temp ** 2 / (1 + a_temp) / x_norm
            )
    return return_value