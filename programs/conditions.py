from numpy import array, vstack, isin, sin, cos, sinh, cosh, power, inf, pi
from torch import device, cuda, FloatTensor
devicee = device("cuda:0" if cuda.is_available() else "cpu")

def form_condiition(x_range, y_range, t_range, arr):
    """
    Формирует массив с условием на заданных координатах
    """

    boundary = vstack([x_range, y_range, t_range, arr]).T
    return boundary


def form_boundaries(x_range, y_range, t_range, cond, ones, zeros,
                    top=False, bottom=False, left=False, right=False):
    """
    Формирует массив граничных условий

    Порядок границ – верхняя, нижняя, левая, правая
    """

    i = 0
    # Создание массивов с inf 
    u_top = u_bottom = u_left = u_right = array([[inf,inf,inf,inf]])
    
    # Заполнение массивов значениями, если они есть
    if top==True:
        u_top = form_condiition(x_range, ones, t_range, cond[i])
        i += 1

    if bottom==True:
        u_bottom = form_condiition(x_range, zeros, t_range, cond[i])
        i += 1

    if left==True:
        u_left = form_condiition(zeros, y_range, t_range, cond[i])
        i += 1

    if right==True:
        u_right = form_condiition(ones, y_range, t_range, cond[i])
        i += 1
    
    BC = vstack([u_top, u_bottom, u_left, u_right])

    # очистка массива от незаданных границ
    remove = inf
    BC = BC.reshape(-1)

    filtered = BC[~isin(BC, remove)]
    filtered = filtered.reshape((int(filtered.shape[0]/4),4))

    return filtered


def form_condition_arrays(BC_arr):
    """
    Формирует отдельные массивы с координатами и условием
    """

    x = BC_arr[:, 0]
    y = BC_arr[:, 1]
    t = BC_arr[:, 2]
    f = BC_arr[:, 3]
    
    x = FloatTensor(x).to(devicee)
    y = FloatTensor(y).to(devicee)
    t = FloatTensor(t).to(devicee)
    f = FloatTensor(f).to(devicee)

    return x, y, t, f

def Poisson_analytical(mu, Q0, w, x, y, chi, H, L, K):
    """
    Возвращает аналитическое решение уравнения Пуассона при нулевой концентрации
    """
    pp = x / H
    for k in range(1,K):
        pp += 2 / (pi * k * chi) * power(-1, k%2) * sin(pi * k * chi / H) * cos( 2 * pi * k * y / H) * sinh( 2 * pi * k * (x - L / 2) / H) / (2 * pi * k / H * cosh(2 * pi * k * L / 2 / H))
    return 12 * mu * Q0 / power(w,3) * pp
