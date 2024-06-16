import numpy as np
from jax import numpy as jnp
from texttable import Texttable
import time
import matplotlib.pyplot as plt

class Solver():

    def __init__(self, mu0, cmax, c_in, chi, Q, w,
                 H, L, shape_x, shape_y):
        self.mu0 = mu0
        self.cmax = cmax
        self.c_in = c_in
        self.chi = chi
        self.Q = Q
        self.w = w

        self.CFL = 0.5

        self.sh = (shape_x, shape_y)
        self.H = H
        self.L = L
        self.dx = self.L / (self.sh[0] - 1)
        self.dy = self.H / (self.sh[1] - 1)

        self.c = np.zeros(shape=(1,) + self.sh)
        self.p = np.array([])
        self.t = [0]

        self.print_tab = Texttable()
        self.times = []
        self.dt = []


    def viscosity(self, c):
        return self.mu0 * np.power((1 - c / self.cmax), -2.5)
    

    def save(self, paths):
        np.save(paths[0], self.c)
        np.save(paths[1], self.p)
        np.save(paths[2], self.t)


    def load(self, paths):
        self.c = np.load(paths[0])
        self.p = np.load(paths[1])
        self.t = np.load(paths[2])

    def make_Poisson_matrix(self, mobility):
        """"
        matrix for vectorized Poisson equation 
        Ap=f <=> (Lamda*p_x)_x + (Lamda*p_y)_y = 0
        BC:
        p_y = 0, y=0, y=H
        p_x = -+Q*psi/Lambda, x=0, x=L

        Parameters:
        --------
        Lambda - mobility - matrix NxM: N - height, M - width

        hx - x-step

        hy - y-step
        """

        # compute Lambda ij+-1/2 - Size N(M) - 1 #forward differences
        mobility_x = (mobility[:,1:] - mobility[:,:-1]) / (2 * self.dx)
        mobility_y = (mobility[1:] - mobility[:-1]) / (2 * self.dy)
        # Lambda 1/2, ..., N-1/2, j
        LambdaY = mobility[:-1] + mobility_y * self.dy / 2 #ij -> Lambda i+1/2, j
        # Lambda i, 1/2, ..., M-1/2
        LambdaX = mobility[:,:-1] + mobility_x * self.dx / 2 #ij -> Lambda i, j+1/2

        a = self.dx / self.dy
        b = 1 / a

        #Create Matrix
        MatrixA = np.zeros(shape=(self.sh[0]*self.sh[1], self.sh[0]*self.sh[1]))

        for i in range(self.sh[1]):
            #BC top
            MatrixA[i,i] = 1 / self.dy
            MatrixA[i,i+self.sh[1]] = -1 / self.dy

            #BC botttom
            MatrixA[-1-i,-1-i] = 1 / self.dy
            MatrixA[-1-i,-1-(i+self.sh[1])] = -1 / self.dy

        for i in range(1, self.sh[0]-1):
            #BC left
            MatrixA[i*self.sh[1], i*self.sh[1]] = 1 / self.dx
            MatrixA[i*self.sh[1], i*self.sh[1]+1] = -1 / self.dx

            #BC right
            MatrixA[(i+1)*self.sh[1]-1, (i+1)*self.sh[1]-1] = 1 / self.dx
            MatrixA[(i+1)*self.sh[1]-1, (i+1)*self.sh[1]-1 -1] = -1 / self.dx

            for j in range (1,self.sh[1]-1):
                #Poisson p_i,j
                MatrixA[i*self.sh[1]+j,i*self.sh[1]+j] = -((LambdaY[i-1,j] + LambdaY[i,j]) * a 
                                                           + (LambdaX[i,j-1] + LambdaX[i,j]) * b)
                #Poisson p_i-1,j
                MatrixA[i*self.sh[1]+j,i*self.sh[1]+j+self.sh[1]] = LambdaY[i-1,j] * b
                #Poisson p_i+1,j
                MatrixA[i*self.sh[1]+j,i*self.sh[1]+j-self.sh[1]] = LambdaY[i,j] * b
                #Poisson p_i,j-1
                MatrixA[i*self.sh[1]+j,i*self.sh[1]+j-1] = LambdaX[i,j-1] * a
                #Poisson p_i,j+1
                MatrixA[i*self.sh[1]+j,i*self.sh[1]+j+1] = LambdaX[i,j] * a

        return MatrixA
    

    def Poisson_Solver(self, mobility, psi):
        """
        (Lamda*p_x)_x + (Lamda*p_y)_y = 0
        BC:
        p_y = 0, y=0, y=H
        p_x = -+Q*psi/Lambda, x=0, x=L

        Parameters:
        --------
        Lamda - mobility - matrix NxM: N - height, M - width

        hx - x-step

        hy - y-step
        
        gap_width

        Q - const(?)
        """

        # Заменить на более быстрый вариант
        left_side_matrix = self.make_Poisson_matrix(mobility)
        pinv = np.linalg.pinv(left_side_matrix)
        
        right_side = np.zeros(shape=(self.sh[0],self.sh[1]))
        right_side[:,0] = -self.Q * psi / mobility[0]
        right_side[:,-1] = self.Q * psi / mobility[0]

        # Ap=f
        right_side_vector = right_side.reshape(-1)
        solution_vector = jnp.matmul(pinv, right_side_vector)
        solution = solution_vector.reshape(self.sh[0], self.sh[1])

        return solution
    

    def Transport_Solver(self, t_end):
        """
        (Lamda*p_x)_x + (Lamda*p_y)_y = 0
        BC:
        p_y = 0, y=0, y=H
        p_x = -+Q*psi/Lambda, x=0, x=L

        and
        
        w*c_t + div(c*w*velocity_vector)=right_side
        
        q = cw
        q_t + div(q*velocity_vector) = right_side
        """
        
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,15,1,15,1,25,1,30,1])
        self.print_tab.add_rows([['|','Iteration','|', 'Time, s','|','speed of computing, it/s','|', 'estimated computation time, s', '|']])
        print(self.print_tab.draw())

        y_line = np.arange(0,self.sh[0]*self.dy, self.dy)
        psi = 1 / self.chi * np.where(np.abs(y_line - y_line[-1] / 2) < self.chi/2, 1, 0)

        self.c[0,:,0] = self.c_in * psi * self.chi

        k = 0
        while np.round(self.t[-1],5) < t_end:
            start_time = time.time()

            # mobility at t = k*dt
            mobility = np.power(self.w, 3) / (12 * self.viscosity(self.c[k]))

            # p at t = k*dt
            p = self.Poisson_Solver(mobility, psi)
            if k==0:
                self.p = np.array((p, ))
            else:
                self.p = np.concatenate((self.p, (p, )))
            p_x = (p[:,2:] - p[:,:-2]) / (2 * self.dx)
            p_y = (p[2:] - p[:-2])     / (2 * self.dy)

            # velocities at t = k*dt
            vx, vy = -mobility / self.w, -mobility / self.w

            vx[:,1:-1] *= p_x
            vx[:,0] = vx[:,-1] = psi * self.Q / self.w
            
            vy[1:-1] *= p_y
            vy[0], vy[-1] = 0, 0

            # CFL
            dt = np.min((self.dx, self.dy)) * self.CFL / (np.max((np.abs(vx), np.abs(vy))))
            if self.t[-1]+dt >t_end:
                dt = t_end - self.t[-1]
            self.dt.append(dt)
            self.t.append(dt + self.t[-1])

            # Lax-Friedrichs 
            # for i in range(1,N-1):
            #     for j in range(1,M-1):
            #         q[k+1,i,j]= (q[k,i-1,j]+q[k,i,j+1]+q[k,i,j-1]+q[k,i+1,j])/4 + dt/(2*dx)*(vx[i,j+1]*q[k,i,j+1]-vx[i,j-1]*q[k,i,j-1])+ dt/(2*dy)*(vy[i+1,j]*q[k,i+1,j]-vy[i-1,j]*q[k,i-1,j])

            self.c = np.concatenate((self.c, np.zeros((1, ) + self.sh)))

            self.c[k+1,1:-1,1:-1] = ((self.c[k,:-2,1:-1] + self.c[k,2:,1:-1] + self.c[k,1:-1,:-2] + self.c[k,1:-1,2:]) / 4 
                                     + dt / (2 * self.dx) * (vx[1:-1,2:] * self.c[k,1:-1,2:] - vx[1:-1,:-2] * self.c[k,1:-1,:-2]) 
                                     + dt / (2 * self.dy) * (vy[2:,1:-1] * self.c[k,2:,1:-1] - vy[:-2,1:-1] * self.c[k,:-2,1:-1]))

            self.c[k+1,:,0] = self.c_in * psi * self.chi
            self.c[k+1,:,-1] = self.c[k+1,:,-2] * psi * self.chi

            end_time = time.time()
            self.times.append(np.round(end_time - start_time, 5))

            self.print_tab.add_rows([['|',f'{k}\t','|',
                                    f'{np.round(self.t[-1],5)}\t','|',
                                    f'1/{self.times[-1]}\t','|',
                                    f'{np.round(t_end/self.dt[0]*np.mean(self.times), 5)}', '|']])
            print(self.print_tab.draw())

            k += 1
