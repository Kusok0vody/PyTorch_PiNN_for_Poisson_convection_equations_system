from numpy import sin, cos, sinh, cosh, power, pi


def Poisson_analytical(mu, Q0, w, x, y, chi, H, L, K):
    """
    Returns an analytical solution to the Poisson equation at zero concentration\
    """
    pp = x / H
    for k in range(1,K):
        pp += 2 / (pi * k * chi) * power(-1, k%2) * sin(pi * k * chi / H) * cos( 2 * pi * k * y / H) * sinh( 2 * pi * k * (x - L / 2) / H) / (2 * pi * k / H * cosh(2 * pi * k * L / 2 / H))

    return 12 * mu * Q0 / power(w,3) * pp