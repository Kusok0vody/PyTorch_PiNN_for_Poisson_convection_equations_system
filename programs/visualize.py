import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import programs.misc as misc



def plot_results(data,
                 limits:list,
                 t:list,
                 path='',
                 cmaps:list=['turbo']*6,
                 title:list=None,
                 lims:list=None
                 ):
    """
    Plots data from Neural Network
    
    Parameters:
    -----------
    data : array of arrays
    Arrays of two arrays
    limits : list[np.float64]
    Limits of plot
    t : np.float64
    End time
    net
    Neural Network
    Nx : int
    Number of points on the x axis
    Ny : int
    Number of points on the x axis
    clims : list
    Array of limits for colourbars
    
    Returns
    -------
    data : list
    Array of the first and the last frames
    """
    x_min, x_max, y_min, y_max = limits

    if cmaps is None:
        cmaps = ['turbo'] * 6  # По умолчанию одна цветовая карта для всех

    if title is None:
        title = [f'$f_{i}$ at $t={t[j // 3]}$' for j, i in enumerate(range(1, 7))]

    if lims is None:
        lims = [np.min(data[i // 2]) if i % 2 == 0 else np.max(data[i // 2]) for i in range(6)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 10))  # Создаём 2x3 сетку графиков

    for i, ax in enumerate(axes.flat):  # Перебираем оси на графике
        col = i % 3  # Определяем столбец (0 - первый, 1 - второй, 2 - третий)
        im = ax.imshow(data[i], cmap=cmaps[i], vmin=lims[2 * col], vmax=lims[2 * col + 1], extent=[x_min, x_max, y_max, y_min])
        ax.set_title(title[i])
        ax.invert_yaxis()
        ax.set_xlabel('x, m')
        ax.set_ylabel('y, m')

        # Добавляем colorbar только для верхнего ряда (первых 3 графиков)
        if i < 3:
            fig.colorbar(im, ax=ax, location='bottom')

    fig.tight_layout()
    plt.show()
    if path!='':
      fig.savefig(path)


def plot(data:np.ndarray, limits:list=False, title:str=None, clim:list=None):
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
  plt.gca().invert_yaxis()
  if limits!=False:
    plt.imshow(data, cmap='turbo', extent=[limits[0], limits[1], limits[2], limits[3]])
  else:
     plt.imshow(data, cmap='turbo')
  plt.colorbar()
  if title!=None:
    plt.title(title)
  if clim!=None:
    plt.clim(clim[0],clim[1])


def plot_BC(data_tb:np.ndarray, data_lr:np.ndarray, x:np.ndarray, y:np.ndarray):
    """
    Plots boundaries with specified types (Neumann is 1 and Dirichlet is any other value) 
    
    Parameters:
    -----------
    data_tb : np.ndarray
    data_lr : np.ndarray
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
    plt.plot(x, data_tb[0], c='black')
    plt.grid()
    plt.title('Top')
    
    plt.subplot(222)
    plt.plot(x, data_tb[-1], c='black')
    plt.grid()
    plt.title('Bottom')
    
    plt.subplot(223)
    plt.plot(y, data_lr[:,0], c='black')
    plt.grid()
    plt.title('Left')
    
    plt.subplot(224)
    plt.plot(y, data_lr[:,-1], c='black')
    plt.grid()
    plt.title('Right')


def check_PDE_points(t, net, Nx, Ny, eps, size):

    x = torch.linspace(net.size[0], net.size[1], Nx)
    y = torch.linspace(net.size[2], net.size[3], Ny)
    mesh_XYT = torch.stack(torch.meshgrid(x, y, torch.Tensor([t]).to(net.device), indexing='ij')).reshape(3, -1).T
    
    X = torch.autograd.Variable(mesh_XYT[:,0], requires_grad=True)
    Y = torch.autograd.Variable(mesh_XYT[:,1], requires_grad=True)
    T = torch.autograd.Variable(mesh_XYT[:,2], requires_grad=True)

    conv  = net.get_conv(X,Y,T).reshape(Nx, Ny).transpose(1,0) / (net.w_func(X,Y).max()**3).data.cpu().numpy()
    div   = net.get_div(X,Y,T).reshape(Nx, Ny).transpose(1,0) / (net.w_func(X,Y).max()**3).data.cpu().numpy()
    corr  = net.get_corr(X,Y,T).reshape(Nx, Ny).transpose(1,0)
    total = net.weights[0]*np.abs(conv) + net.weights[1]*np.abs(div) + net.weights[2]*np.abs(corr)
    plt.figure(dpi=200)
    
    plt.imshow(np.where(total-eps>=0, total, np.nan), cmap='magma', extent=[size[0],size[1],size[3],size[2]])
    plt.gca().invert_yaxis()
    
    plt.colorbar()
    
    mask = (net.t_PDE.data.cpu().numpy() == t)
    plt.scatter(net.x_PDE.data.cpu()[mask], net.y_PDE.data.cpu().numpy()[mask],alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
    plt.title(f't = {t} s')


def plot_boundary_fields(net, mesh_yt, mesh_ty, px, py, c, u_out, plot_linear:False):
  plt.figure(figsize=(15,12))

  N = int(net.N_BC2/4)
  M = net.N_BC2

  # TOP BOUNDARY
  plt.subplot(221)
  plt.title('Top')
  if plot_linear:
    plt.scatter(net.x_BC[:N].data.cpu(), net.t_BC[:N].data.cpu(),
                alpha=0.5, c='lightblue', edgecolors='white', s=7, linewidths=0.5)
  plt.scatter(net.x_BC[M:2*M].data.cpu(), net.t_BC[M:2*M].data.cpu(),
              alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
  plt.imshow(np.abs(py[:,-1,:]), cmap='magma',extent=[0,1,1,0])
  plt.gca().invert_yaxis()
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')

  # BOTTOM BOUNDARY
  plt.subplot(222)
  plt.title('Bottom')
  if plot_linear:
    plt.scatter(net.x_BC[N:2*N].data.cpu(), net.t_BC[N:2*N].data.cpu(),
                alpha=0.5, c='lightblue', edgecolors='white', s=7, linewidths=0.5)
  plt.scatter(net.x_BC[M+net.N_BC2:M+2*net.N_BC2].data.cpu(), net.t_BC[M+net.N_BC2:M+2*net.N_BC2].data.cpu(),
              alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
  plt.imshow(np.abs(py[:,0,:]), cmap='magma', extent=[0,1,1,0])
  plt.gca().invert_yaxis()
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')

  # LEFT BOUNDARY
  cc = np.where(mesh_ty<=net.times[0], net.c_cond[0], net.c_cond[1])*np.ones_like(c[:,:,-1])
  plt.subplot(223)
  plt.title('Left')
  if plot_linear:
    plt.scatter(net.y_BC[2*N:3*N].data.cpu(), net.t_BC[2*N:3*N].data.cpu(),
                alpha=0.5, c='lightblue', edgecolors='white', s=7, linewidths=0.5)
  plt.scatter(net.y_BC[3*M:4*M].data.cpu(), net.t_BC[3*M:4*M].data.cpu(),
              alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
  plt.imshow(np.abs(px[:,:,0] - -np.where(np.abs(mesh_yt-net.size[3]/2)<=net.chi/2,1,0)*12*net.u_in/net.w_func(0*mesh_yt,mesh_yt)**2*
                    np.where(mesh_ty<=net.times[0],misc.viscosity(net.mu0, net.c_cond[0],net.cmax),misc.viscosity(net.mu0, net.c_cond[1],net.cmax))) +
            np.abs(np.where(np.abs(mesh_yt-net.size[3]/2)<=net.chi/2, c[:,:,0] - cc, 0)), extent=[0,1,1,0], cmap='magma')
  plt.gca().invert_yaxis()
  plt.colorbar()
  plt.xlabel('y')
  plt.ylabel('t')

  # RIGHT BOUNDARY
  plt.subplot(224)
  plt.title('Right')
  if plot_linear:
    plt.scatter(net.y_BC[3*N:4*N].data.cpu(), net.t_BC[3*N:4*N].data.cpu(),
                alpha=0.5, c='lightblue', edgecolors='white', s=7, linewidths=0.5)
  plt.scatter(net.y_BC[4*M:].data.cpu(), net.t_BC[4*M:].data.cpu(),
              alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
  plt.imshow(np.abs(px[:,:,-1] - -np.where(np.abs(mesh_yt-net.size[3]/2)<=net.chi/2,1,0)*12*misc.viscosity(net.mu0, 0, net.cmax)*u_out/net.w_func(1+0*mesh_yt,mesh_yt)**2), extent=[0,1,1,0], cmap='magma')
  plt.gca().invert_yaxis()
  plt.colorbar()  
  plt.xlabel('y')
  plt.ylabel('t')


def anim_result(data,
                steps,
                streamplot_data=False,
                clims=False,
                title=False,
                path:str=False,
                name:str=False,
                colour:str='viridis',
                savetogif:bool=False,
                showMe:bool=False,
                ):

    def animate(i):   
        if title == False:
          title_ = ax.set_title('f(x,y,t)' + f' at t = {np.round(i*steps,3)}')
        else:
          title_ = ax.set_title(title + f' at t = {np.round(i*steps,3)}')
        ax1.cla()
        ax1.axis('off')
        strm = ax1.streamplot(streamplot_data[0],
                              streamplot_data[1],
                              streamplot_data[2][i],
                              streamplot_data[3][i],
                              linewidth=1,
                              color='lightgrey',
                              density=[1,0.8])
        line.set_array(data[i])
        return line, title_, strm
    
    size_t = data.shape[0]
    size_x = data[0].shape[1]
    size_y = data[0].shape[0]

    fig, ax = plt.subplots()
    fig.dpi = 150
    ax1 = ax.twinx()
    ax1.axis('off')
    plt.gca().invert_yaxis()

    if streamplot_data!=False:
       strm = ax1.streamplot(streamplot_data[0],
                             streamplot_data[1],
                             streamplot_data[2][0],
                             streamplot_data[3][0],
                             linewidth=1,
                             color='lightgrey',
                             density=[1,0.8])

    line = ax.imshow(data[0], cmap = colour, extent = [0, 1, 0, 1], interpolation='none')
    plt.colorbar(line, ax=ax)
    if clims!=False:
      line.set_clim(clims[0],clims[1])

    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')

    if title == False:
       ax.set_title('f(x,y,t)'+ ' at t = 0')
    else:
       ax.set_title(title + ' at t = 0')

    ani = animation.FuncAnimation(fig, animate, interval=30, frames = size_t)

    if showMe == True:
        plt.show()

    if name==False: name = f"t={size_t}, max_x={size_x}, max_y={size_y}"
    if path==False: path = ""
    else: path +="/"

    if savetogif == True:
      if os.path.exists(path+name+".gif"):
          k = 1
          while os.path.exists(path+name+f"_{k}"+".gif"):
            k += 1                    
          writer = animation.PillowWriter(fps=120, metadata=dict(artist='Doofenshmirtz Evil Incorporated'), bitrate=1800)
          ani.save(path+name+f"_{k}"+".gif", writer=writer)
      else:
        writer = animation.PillowWriter(fps=120, metadata=dict(artist='Doofenshmirtz Evil Incorporated'), bitrate=1800)
        ani.save(path+name+".gif", writer=writer)