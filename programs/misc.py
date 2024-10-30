from numpy import sin, cos, sinh, cosh, power, pi, where
import torch


def Poisson_analytical(mu, Q0, w, x, y, chi, H, L, K):
    """
    Returns an analytical solution to the Poisson equation at zero concentration
    """
    pp = x / H
    for k in range(1,K):
        pp += 2 / (pi * k * chi) * power(-1, k%2) * sin(pi * k * chi / H) * cos( 2 * pi * k * y / H) * sinh( 2 * pi * k * (x - L / 2) / H) / (2 * pi * k / H * cosh(2 * pi * k * L / 2 / H))

    return 12 * mu * Q0 / power(w,3) * pp


def psi(y, chi):
    return torch.where(abs(y - max(y) / 2).round(decimals=10) <= chi / 2, 1., 0.)


def heaviside(x):
    return where(x>=0, 1, 0)


def delta(x):
    return torch.where(x==0, 1, 0)


def th(x, k=1, c=0, b=0):
    return torch.tanh(k * (x - c)).round(decimals=10) + b


def psi_th(x, c1=0, c2=1):
    return th(x=x, k=500, c=c1, b=1) * th(x=x, k=-500, c=c2, b=1) / 4


def heaviside_th(x):
    return 1 / (1 + 1 / torch.exp(500*x))


def delta_th(y, c=0):
    return th(-500, y, c, 1) * th(500, y, c, 1)


def viscosity(mu_0, c, c_max, beta=-2.5):
    return mu_0 * (1 - c / c_max) ** (beta)


def compare_viscosity(c1, c2, cmax, beta=-2.5):
    return viscosity(1,c1,cmax,beta) / viscosity(1,c2,cmax,beta)