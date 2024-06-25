from numpy import vstack, sin, cos, sinh, cosh, power, pi
from torch import device, cuda, FloatTensor
devicee = device("cuda:0" if cuda.is_available() else "cpu")


def form_condiition(x_range, y_range, t_range, arr):
    """
    Forms an array with a condition at given coordinates
    """

    boundary = vstack([x_range, y_range, t_range, arr]).T
    return boundary


def form_boundaries(x_range, y_range, t_range, cond, ones, zeros):
    """
    Generates an array of boundary conditions

    Border order – top, bottom, left, right
    """

    u_top = form_condiition(x_range, ones, t_range, cond[0])
    u_bottom = form_condiition(x_range, zeros, t_range, cond[1])
    u_left = form_condiition(zeros, y_range, t_range, cond[2])
    u_right = form_condiition(ones, y_range, t_range, cond[3])

    return vstack([u_top, u_bottom, u_left, u_right])


def form_condition_arrays(BC_arr):
    """
    Forms separate arrays with coordinates and condition
    """

    x = BC_arr[:, 0]
    y = BC_arr[:, 1]
    t = BC_arr[:, 2]
    f = BC_arr[:, 3]
    
    x = FloatTensor(x).to(devicee)
    y = FloatTensor(y).to(devicee)
    t = FloatTensor(t).to(devicee)
    f = FloatTensor(f).to(devicee)

    return f, x, y, t


def Poisson_analytical(mu, Q0, w, x, y, chi, H, L, K):
    """
    Returns an analytical solution to the Poisson equation at zero concentration\
    
    Works wrong...
    """

    pp = x / H
    for k in range(1,K):
        pp += 2 / (pi * k * chi) * power(-1, k%2) * sin(pi * k * chi / H) * cos( 2 * pi * k * y / H) * sinh( 2 * pi * k * (x - L / 2) / H) / (2 * pi * k / H * cosh(2 * pi * k * L / 2 / H))

    return 12 * mu * Q0 / power(w,3) * pp
