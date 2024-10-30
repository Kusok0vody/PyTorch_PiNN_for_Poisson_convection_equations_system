import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def plot_results(data,
                 limits:list,
                 t:np.float64,
                 Nx:int,
                 Ny:int,
                 clims:list=False):
  """
  Plots data from Neural Network

  Parameters:
  -----------
  data : array of arrays
    Array with c and p
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
  x_min = limits[0]
  x_max = limits[1]
  y_min = limits[2]
  y_max = limits[3]

  dx = (x_max - x_min) / (Nx - 1)
  dy = (y_max - y_min) / (Ny - 1)
  x = np.arange(x_min, x_max+dx/2, dx)
  y = np.arange(y_min, y_max+dy/2, dy)
  tt = [0,t,0,t]
  
  mesh_x, mesh_y = np.meshgrid(x, y)
  images = []
  
  fig, axes = plt.subplots(nrows=2,ncols=3,dpi=150, 
                  gridspec_kw={"width_ratios":[1,1,0.05]})
  fig.subplots_adjust(wspace=0.6)

  for i in range(2):
      for j in range(2):
        im = axes[i,j].pcolormesh(mesh_x, mesh_y, data[2*i+j], cmap='turbo')
        axes[i,j].set_title(f"t={tt[2*i+j]}")
        axes[i,j].set_aspect('equal', 'box')
        axes[i,j].set_xticks(np.arange(x_min, x_max+dx/2, 0.2))
        axes[i,j].set_yticks(np.arange(y_min, y_max+dy/2, 0.2))
        images.append(im)
        if clims==False:
          images[-1].set_clim(np.min((data[2*i], data[2*i+1])), np.max((data[2*i], data[2*i+1])))
        else:
          images[-1].set_clim(clims[2*i],clims[2*i+1])
      fig.colorbar(im, cax=axes[i,j+1])
  fig.tight_layout()
  plt.show()


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
    plt.plot(x, (data[-1] - data[-2]) / dy, c='black')
  else:
    plt.plot(x, data[0], c='black')
  plt.xticks(np.arange(np.min(x), np.max(x)+dx/2, 10*dx))
  plt.grid()
  plt.title('Top')

  plt.subplot(222)
  if BC_types[1]==1:
    plt.plot(x, (data[0] - data[1])/ dy, c='black')
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
  if BC_types[3]==1:
    plt.plot(y, (data[:,-1] - data[:,-2]) / dx, c='black')
  else:
    plt.plot(y, data[:,-1], c='black')
  plt.xticks(np.arange(np.min(x), np.max(x)+dx/2, 10*dx))
  plt.grid()
  plt.title('Right')


def plot_loss(loss):
  plt.semilogy(loss, c='black', label=loss[-1])
  plt.title('Loss')
  plt.xlabel('Iterations')
  plt.legend()
  plt.grid()


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