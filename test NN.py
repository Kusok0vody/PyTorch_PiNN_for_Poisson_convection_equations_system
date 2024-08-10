import numpy as np 
import matplotlib.pyplot as plt

import torch

import programs.NN as NN
import programs.visualize as vis
import programs.conditions as cnd

torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

Nx = 100
Ny = 100
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x,y)
cond = -12 * 0.1 * -1 / 0.51**2 *(1 - 0.5)**(-2.5)
print (cond)
net = NN.Poisson_Convection(w=1., mu0=0.1, cmax=1, v_in=-1, c_in=0.5, chi=0.2,
                            c_cond=[[0, 0, 0.5, 0], [1, 1, 0, 1], [False, False, True, False]], p_cond=[[0, 0, cond, cond], [1, 1, 1, 1], [False, False, True, True]],
                            size=[0,1,0,1,0,10], collocation=32, cond_points=32)



# net.full_load("data\CL_w,[1, 0.694, 0.51, 0.39, 0.308, 0.25, 0.207, 0.173, 0.148, 0.128, 0.111]/0.25",
            #   "data\CL_w,[1, 0.694, 0.51, 0.39, 0.308, 0.25, 0.207, 0.173, 0.148, 0.128, 0.111]/0.25_data.npy")
# net.full_load("data\CL_w,[0.0318, 0.025, 0.02, 0.015, 0.01]/0.0318",
            #   "data\CL_w,[0.0318, 0.025, 0.02, 0.015, 0.01]/0.0318_data.npy")
net.full_load("tty","ttyd.npy")
net.model.eval()
outputs = vis.plot_results(0, 1, 0, 1, 2, net, Nx, Ny)#,clims=[0.4,0.7,0,1])

vis.plot_BC(outputs[2], x, y, 1/(Nx-1), 1/(Ny-1), [1,1,1,1])



N = 60
mesh_x, mesh_y = np.meshgrid(x, y)
x = np.ravel(mesh_x).reshape(-1, 1)
y = np.ravel(mesh_y).reshape(-1, 1)

pt_x = torch.autograd.Variable(torch.from_numpy(x).float(), requires_grad=True).cuda()
pt_y = torch.autograd.Variable(torch.from_numpy(y).float(), requires_grad=True).cuda()

period = np.linspace(0, 2, N)
all_u = []



for i in range(N):
    t = torch.full(x.shape, period[i])
    pt_t = torch.autograd.Variable(t, requires_grad=True).cuda()
    pt_u = net.model([pt_x, pt_y, pt_t])[:,0]
    u = pt_u.data.cpu().numpy()
    mesh_u = u.reshape(mesh_x.shape)
    all_u.append(mesh_u)
all_u = np.array(all_u)
print (all_u.shape)
# plt.show()

vis.anim_result(all_u,
                2/N,
                # clims=[0,1],
                colour='turbo',
                path='gifs_NN',
                savetogif=True,
                showMe=True,
                )