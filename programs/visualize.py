import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch


def plot_results(x_min:np.float64,
                 x_max:np.float64,
                 y_min:np.float64,
                 y_max:np.float64,
                 t:np.float64,
                 net,
                 Nx:int,
                 Ny:int)->list:
  """
  Plots data from Neural Network

  Parameters:
  -----------
  x_min : np.float64
    Minimum of x
  x_max : np.float64
    Maximum of x
  y_min : np.float64
    Minimum of y
  y_max : np.float64
    Maximum of y
  t : np.float64
    End time
  net
    Neural Network
  Nx : int
    Number of points on the x axis
  Ny : int
    Number of points on the x axis

  Returns
  -------
  data : list
    Array of the first and the last frames
  """
  dx = (x_max - x_min) / (Nx - 1)
  dy = (y_max - y_min) / (Ny - 1)
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
      pt_u = net([pt_x, pt_y, pt_t])[:,f[i][1]]
      u = pt_u.data.cpu().numpy()
      mesh_u = u.reshape(mesh_x.shape)
      data.append(mesh_u)
  
  fig, axes = plt.subplots(nrows=2,ncols=3,dpi=150, 
                  gridspec_kw={"width_ratios":[1,1,0.05]})
  fig.subplots_adjust(wspace=0.6)

  k = 0
  for i in range(2):
      for j in range(2):
        im = axes[i,j].pcolormesh(mesh_x, mesh_y, data[k], cmap='jet')
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


def plot(data:np.ndarray, limits:list[np.float64], title:str=None, clim:list=None):
  """
  Plots data

  Parameters:
  -----------
  data : np.ndarray
  limits : list[np.float64]
    Limits of plot
  title : str
    Title for a plot
  """
  plt.imshow(data, cmap='jet', extent=[limits[0], limits[1], limits[2], limits[3]])
  plt.colorbar()
  if title!=None:
    plt.title(title)
  if clim!=None:
    plt.clim(clim[0],clim[1])


def plot_BC(data:np.ndarray, x:np.ndarray, y:np.ndarray, dx:np.float64, dy:np.float64, BC_types:list):
  """
  Plots boundaries with specified types (Neumann is 1 and Dirichlet is any other value) 
  
  Parameters:
  -----------
  data : np.ndarray
  x : np.ndarray
    Array of points of x axis
  x : np.ndarray
    Array of points of y axis
  dx : np.float64
    Step between x points
  dy : np.float64
    Step between y points
  BC_types : list
    Array of conditions
  """
  plt.figure(figsize=(7, 7))

  plt.subplot(221)
  if BC_types[0]==1:
    plt.plot(x, (data[0] - data[1]) / dy, c='black')
  else:
    plt.plot(x, data[0], c='black')
  plt.xticks(np.arange(np.min(x), np.max(x)+dx/2, 10*dx))
  plt.grid()
  plt.title('Top')

  plt.subplot(222)
  if BC_types[1]==1:
    plt.plot(x, (data[-1] - data[-2])/ dy, c='black')
  else:
    plt.plot(x, data[-1], c='black')
  plt.xticks(np.arange(np.min(x), np.max(x)+dx/2, 10*dx))
  plt.grid()
  plt.title('Bottom')

  plt.subplot(223)
  if BC_types[2]==1:
    plt.plot(y, (data[:,0] - data[:,1]) / dx, c='black')
  else:
    plt.plot(y, data[:,0], c='black')
  plt.xticks(np.arange(np.min(x), np.max(x)+dx/2, 10*dx))
  plt.grid()
  plt.title('Left')

  plt.subplot(224)
  if BC_types[0]==1:
    plt.plot(y, (data[:,-1] - data[:,-2]) / dx, c='black')
  else:
    plt.plot(y, data[:,-1], c='black')
  plt.xticks(np.arange(np.min(x), np.max(x)+dx/2, 10*dx))
  plt.grid()
  plt.title('Right')
