import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def form_condiition(ranges:list, arr):
    """
    Forms an array with a condition at given coordinates
    """
    boundary = arr
    mesh = torch.stack(torch.meshgrid(ranges)).reshape(len(ranges), -1)
    boundary = torch.vstack([boundary, mesh])
    return boundary.T


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
    return torch.vstack([u_top, u_bottom, u_left, u_right])


def form_condition_arrays(BC_arr):
    """
    Forms separate arrays with coordinates and condition
    """
    f = BC_arr[:, 0]
    X = []
    for i in range(1,len(BC_arr[0])):
        X.append(BC_arr[:,i])
    
    return f, X


def formBC_coords(raw_x, raw_y, raw_t):
    top = torch.stack(torch.meshgrid((raw_x, torch.ones_like(raw_y), raw_t))).reshape(3, -1).T
    bottom = torch.stack(torch.meshgrid((raw_x, torch.zeros_like(raw_y), raw_t))).reshape(3, -1).T
    left = torch.stack(torch.meshgrid((torch.zeros_like(raw_y), raw_y, raw_t))).reshape(3, -1).T
    right = torch.stack(torch.meshgrid((torch.ones_like(raw_y), raw_y, raw_t))).reshape(3, -1).T
    all = torch.vstack([top, bottom, left, right])
    x = all[:,0]
    y = all[:,1]
    t = all[:,2]
    return x, y, t