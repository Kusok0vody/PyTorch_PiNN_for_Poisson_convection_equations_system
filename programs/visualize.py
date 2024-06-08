import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch


def plot_results(i, x_min, x_max, y_min, y_max, t, net):
  x = np.arange(x_min, x_max, 0.001)
  y = np.arange(y_min, y_max, 0.001)
  
  mesh_x, mesh_y = np.meshgrid(x, y)
  x = np.ravel(mesh_x).reshape(-1, 1)
  y = np.ravel(mesh_y).reshape(-1, 1)

  pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).cuda()
  pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).cuda()

  period = np.linspace(0, t, 2)

  fig, axes = plt.subplots(1, 2, dpi=150)
  y = []
  for index, axis in enumerate(axes.ravel()):
    t = torch.full(x.shape, period[index])
    pt_t = Variable(t, requires_grad=True).cuda()
    pt_u = net(pt_x, pt_y, pt_t)[:,i]
    u = pt_u.data.cpu().numpy()
    mesh_u = u.reshape(mesh_x.shape)
    # cm = axis.pcolormesh(mesh_x, mesh_y, mesh_u, cmap='jet')#, vmin=-1, vmax=1
    y.append(mesh_u)
    cm = axis.imshow(mesh_u, cmap='jet', extent=[x_min,x_max,y_min,y_max])#, vmin=-1, vmax=1
    cm.set_clim(np.min(mesh_u), np.max(mesh_u))
    axis.set_xlim([x_min, x_max])
    axis.set_title(f't={pt_t.data.cpu().numpy()[0]}')
    axis.set_xticks(np.arange(x_min,x_max+0.005,0.1))
    axis.set_yticks(np.arange(y_min,y_max+0.005,0.1))
    axis.set_ylim([y_min, y_max])
    print (np.min(mesh_u), np.max(mesh_u))
  a = fig.colorbar(cm, ax=axis)
  fig.tight_layout()
  plt.show()
  return y