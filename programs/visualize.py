import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def plot_results(data,
                 limits:list,
                 t:list,
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
    x_min = limits[0]
    x_max = limits[1]
    y_min = limits[2]
    y_max = limits[3]
    
    if title==None:
      title = np.array([f'$f_{i}$ at $t={t[0]}$' for i in range(1,4)] + [f'$f_{i}$ at $t={t[1]}$' for i in range(1,4)])

    if lims==None:
        lims = [np.min(data[0]), np.max(data[3]), np.min(data[1]), np.max(data[4]), np.min(data[2]), np.max(data[5])]

    fig = plt.figure(figsize=(12,10))

    i = 0

    plt.subplot(231)
    plt.imshow(data[i], cmap=cmaps[i], vmin=lims[0], vmax=lims[1], extent=[x_min,x_max,y_max,y_min])
    plt.title(title[i])
    plt.colorbar(location='bottom')
    plt.gca().invert_yaxis()
    i+=1

    plt.subplot(232)
    plt.imshow(data[i], cmap=cmaps[i], vmin=lims[2], vmax=lims[3], extent=[x_min,x_max,y_max,y_min])
    plt.title(title[i])
    plt.colorbar(location='bottom')
    plt.gca().invert_yaxis()
    i+=1

    plt.subplot(233)
    plt.imshow(data[i], cmap=cmaps[i], vmin=lims[4], vmax=lims[5], extent=[x_min,x_max,y_max,y_min])
    plt.title(title[i])
    plt.colorbar(location='bottom')
    plt.gca().invert_yaxis()
    i+=1

    plt.subplot(234)
    plt.imshow(data[i], cmap=cmaps[i], vmin=lims[0], vmax=lims[1], extent=[x_min,x_max,y_max,y_min])
    plt.title(title[i])
    plt.gca().invert_yaxis()
    i+=1
    
    plt.subplot(235)
    plt.imshow(data[i], cmap=cmaps[i], vmin=lims[2], vmax=lims[3], extent=[x_min,x_max,y_max,y_min])
    plt.title(title[i])
    plt.gca().invert_yaxis()
    i+=1

    plt.subplot(236)
    plt.imshow(data[i], cmap=cmaps[i], vmin=lims[4], vmax=lims[5], extent=[x_min,x_max,y_max,y_min])
    plt.title(title[i])
    plt.gca().invert_yaxis()
    i+=1

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