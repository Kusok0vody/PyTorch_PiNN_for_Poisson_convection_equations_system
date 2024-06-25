from numpy import vstack, concatenate, sin, cos, sinh, cosh, power, pi
from torch import device as d, cuda, FloatTensor
device = d("cuda:0" if cuda.is_available() else "cpu")
import numpy as np

def form_condiition(ranges:list, arr):
    """
    Forms an array with a condition at given coordinates
    """

    boundary = vstack(concatenate((arr.reshape(1,-1), ranges))).T
    return boundary


def form_boundaries(ranges, cond, ones, zeros):
    """
    Generates an array of boundary conditions

    Border order – top, bottom, left, right
    """
    if len(ranges)==3:
        u_top = form_condiition([ranges[0], ones, ranges[2]], cond[0])
        u_bottom = form_condiition([ranges[0], zeros, ranges[2]], cond[1])
        u_left = form_condiition([zeros, ranges[1], ranges[2]], cond[2])
        u_right = form_condiition([ones, ranges[1], ranges[2]], cond[3])
    if len(ranges)==2:
        u_top = form_condiition([ranges[0], ones], cond[0])
        u_bottom = form_condiition([ranges[0], zeros], cond[1])
        u_left = form_condiition([zeros, ranges[1]], cond[2])
        u_right = form_condiition([ones, ranges[1]], cond[3])   
    return vstack([u_top, u_bottom, u_left, u_right])


def form_condition_arrays(BC_arr):
    """
    Forms separate arrays with coordinates and condition
    """

    f = FloatTensor(BC_arr[:, 0]).to(device)
    X = []
    for i in range(1,len(BC_arr[0])):
        X.append(FloatTensor(BC_arr[:,i]).to(device))
    
    return f, X


def Poisson_analytical(mu, Q0, w, x, y, chi, H, L, K):
    """
    Returns an analytical solution to the Poisson equation at zero concentration\
    
    Works wrong...
    """

    pp = x / H
    for k in range(1,K):
        pp += 2 / (pi * k * chi) * power(-1, k%2) * sin(pi * k * chi / H) * cos( 2 * pi * k * y / H) * sinh( 2 * pi * k * (x - L / 2) / H) / (2 * pi * k / H * cosh(2 * pi * k * L / 2 / H))

    return 12 * mu * Q0 / power(w,3) * pp
