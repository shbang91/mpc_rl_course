import numpy as np
from casadi import SX, Function

from math import isnan, sqrt, ceil, inf


def is_inside_box(x, y, x1, x2, y1, y2):
    return x1 <= x and x <= x2 and y1 <= y and y <= y2
    # return False

def integrate_RK4(x_expr, u_expr, xdot_expr, dt, N_steps=1):

    h = dt/N_steps

    x_end = x_expr

    xdot_fun = Function('xdot', [x_expr, u_expr], [xdot_expr])

    for _ in range(N_steps):

        k_1 = xdot_fun(x_end, u_expr)
        k_2 = xdot_fun(x_end + 0.5 * h * k_1, u_expr) 
        k_3 = xdot_fun(x_end + 0.5 * h * k_2, u_expr)
        k_4 = xdot_fun(x_end + k_3 * h, u_expr)

        x_end = x_end + (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
    
    F_expr = x_end

    return F_expr


def interpolate_linear(x, x1, v1, x2, v2):
    '''Interpolate the value at x from values associated with two points.
        The two points are a list of triplets:  (x, value).
    '''
          
    return ((x2 - x)*v1 + (x - x1)*v2)/(x2 - x1)

def interpolate_bilinear(x, y, x1, y1, x2, y2, q11, q12, q21, q22):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.
    The function covers also the case of degenerate rectangles.
    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    # print(x, y)
    # print(x1, y1)
    # print(x2, y2)
    # print(q11, q12, q21, q22)

    if not (x1 <= x <= x2 and y1 <= y <= y2):
        import pdb; pdb.set_trace()
    if x1 == x2 and y1 == y2:
        return q11
    elif x1 == x2:
        return interpolate_linear(y, y1, q11, y2, q22)  
    elif y1 == y2:
        return interpolate_linear(x, x1, q11, x2, q22)    
    else:
        val = (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1))
        return val 

def project_indices(x, values):
    #  takes scalar value in and projects in on grid values
    #  return the corresponding index of values (which might lie outside of
    #  values if x is outside values)
    #  returns two equal indices if x lies exactly on the grid

    values = list(values)
    smaller_vals = list(filter(lambda v: v < x, values))

    if not smaller_vals:
        idx1 = -1
    else:
        idx1 = values.index(smaller_vals[-1])
    
    idx2 = idx1 + 1

    if idx2 < len(values) and x == values[idx2]:   # value lies exactly on the grid
        idx1 = idx2

    if idx1 != -1 and not (values[idx1] <= x):
        import pdb; pdb.set_trace()
    if idx2 != len(values) and not (x <= values[idx2]):
        import pdb; pdb.set_trace()

    return (idx1, idx2)
