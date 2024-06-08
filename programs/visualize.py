import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch


def plot_results(x_min, x_max, y_min, y_max, t, net):
  dx = dy = 0.01
  x = np.arange(x_min, x_max+dx/2, dx)
  y = np.arange(y_min, y_max+dy/2, dy)
  
  mesh_x, mesh_y = np.meshgrid(x, y)
  x = np.ravel(mesh_x).reshape(-1, 1)
  y = np.ravel(mesh_y).reshape(-1, 1)

  pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).cuda()
  pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).cuda()

  period = np.linspace(0, t, 2)

  data = []
  images = []
  f = [[0,0],[1,0],[0,1],[1,1]]

  for i in range(len(f)):
      ti = torch.full(x.shape, np.float64(period[f[i][0]]))
      pt_t = Variable(ti, requires_grad=True).cuda()
      pt_u = net(pt_x, pt_y, pt_t)[:,f[i][1]]
      u = pt_u.data.cpu().numpy()
      mesh_u = u.reshape(mesh_x.shape)
      data.append(mesh_u)
  
  fig, axes = plt.subplots(nrows=2,ncols=3,dpi=150, 
                  gridspec_kw={"width_ratios":[1,1,0.05]})
  fig.subplots_adjust(wspace=0.6)

  k = 0
  for i in range(2):
      for j in range(2):
        im = axes[i,j].pcolormesh(mesh_x, mesh_y, data[k], cmap='magma')
        axes[i,j].set_xticks(np.arange(x_min, x_max+dx/2, 0.2))
        axes[i,j].set_yticks(np.arange(y_min, y_max+dy/2, 0.2))
        axes[i,j].set_title(f"t={period[j]}")
        axes[i,j].set_aspect('equal', 'box')
        images.append(im)
        k += 1
      fig.colorbar(im, cax=axes[i,j+1])
      images[k-2].set_clim(np.min((data[k-1], data[k-2])), np.max((data[k-1], data[k-2])))
      images[k-1].set_clim(np.min((data[k-1], data[k-2])), np.max((data[k-1], data[k-2])))
  fig.tight_layout()
  plt.show()

  return data

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print (device)
# net = NN.Net(device)
# net = net.to(device)
# net.load("norm1.1")
# plot_results(0, 1, 0, 1, 2, net)
