
# coding: utf-8

# # Shallow water equations in 2D for inertia-gravity waves

# The equation we are going to sove is
# $$
# \frac{\partial u}{\partial t} - fv = -g \frac{\partial \eta}{\partial x} \\
# \frac{\partial v}{\partial t} + fu = -g \frac{\partial \eta}{\partial y} \\
# \frac{\partial \eta}{\partial t} + H\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right)=0
# $$

# We are going to use the C-grid, where u, v, and $\eta$ are specified on the different points as the following diagram.
#   |         |         |
#---+--- v ---+--- v ---+---  j+1
#   |         |         |
#   u   eta   u   eta   u     j
#   |         |         |
#---+--- v ---+--- v ---+---  j
#   |         |         |
#   u   eta   u   eta   u     j-1
#   |         |         |
#---+--- v ---+--- v ---+---  j-1
#   |         |         |
#  i-1  i-1   i    i   i+1
# Then, we can discretize the equations as the following.
# $$
# \frac{\partial u_{i,j}}{\partial t} = -g \frac{\eta_{i,j} - \eta_{i-1,j}}{\Delta x} + \frac{(f_{i-1,j}+f_{i,j})}{2}\frac{\left(v_{i-1,j}+v_{i,j}+v_{i-1,j+1}+v_{i,j+1} \right)}{4}\\
# \frac{\partial v_{i,j}}{\partial t} = -g \frac{\eta_{i,j} - \eta_{i,j-j}}{\Delta y} - \frac{(f_{i,j-1}+f_{i,j})}{2}\frac{\left(u_{i,j-1}+u_{i+1,j-1}+u_{i,j}+u_{i+1,j} \right)}{4}\\
# \frac{\partial \eta_{i,j}}{\partial t} = -H\left(\frac{u_{i+1,j}-u_{i,j}}{\Delta x} + \frac{v_{i,j+1} - v_{i,j}}{\Delta y} \right)
# $$

# #### Code begins here

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML


# Define functions for the boundary condition and Heun method

# In[2]:


def bcapply(u, v, h, bcflag='periodic'):
    '''
    We are using periodic boundary condition.
    In west-east direction, we replace the first and the last column 
    with the second last and second column, respectively.
    In south-north direction, we replace the first and the last row 
    with the second last and second row, respectively.
    '''
    if bcflag=='periodic':
        # periodic boundary condition in zonal direction
        u[:,0] = u[:,-2]
        v[:,0] = v[:,-2]
        h[:,0] = h[:,-2]

        u[:,-1] = u[:,1]
        v[:,-1] = v[:,1]
        h[:,-1] = h[:,1]
    elif bcflag=='wall':
        # periodic boundary condition in zonal direction
        u[:,0] = 0
        v[:,0] = v[:,1]
        h[:,0] = h[:,1]

        u[:,-1] = 0
        v[:,-1] = v[:,-2]
        h[:,-1] = h[:,-2]
        
    # periodic boundary condition in meridional direction
    u[0,:] = u[-2,:]
    v[0,:] = v[-2,:]
    h[0,:] = h[-2,:]

    u[-1,:] = u[1,:]
    v[-1,:] = v[1,:]
    h[-1,:] = h[1,:]


    return u, v, h


# In[3]:


def rhs_cgrid(u, v, h, Cor, delx, dely):
    """
    The right-hand-side term in the equation expressed in the C-grid.
    """
    u_f = 0.5*(Cor[1:-1,1:-1]+Cor[1:-1,:-2])              *0.25*(v[1:-1,0:-2]+v[1:-1,1:-1]+v[2:,:-2]+v[2:,1:-1])              - (g/(delx))*(h[1:-1,1:-1] - h[1:-1,:-2])
    v_f = -0.5*(Cor[1:-1,1:-1]+Cor[:-2,1:-1])              *0.25*(u[0:-2,1:-1]+u[:-2,2:]+u[1:-1,1:-1]+u[1:-1,2:])              - (g/(dely))*(h[1:-1,1:-1] - h[:-2,1:-1])
    h_f = - H[1:-1,1:-1]*((u[1:-1,2:] - u[1:-1,1:-1])/delx                          +(v[2:,1:-1] - v[1:-1,1:-1])/dely)

    return u_f, v_f, h_f


# ### Parameters

# In[4]:


g = 9.81             # m s-2
f0 = 1e-5         # s-1
beta = 2.0e-11    # m-1 s-1
delx = 100e3    # meter
dely = 100e3    # meter
nx = 500
ny = 250
dt = 600    # second
tottime = 15 # days
dumpfreq = 86400 # second
bcflag='periodic'


# In[5]:


tidx = np.arange(0, tottime*86400+dt, dt)
xgrid = np.arange(0, nx*delx, delx)
ygrid = np.arange(0, ny*dely, dely)
[X, Y] = np.meshgrid(xgrid, ygrid)


# In[6]:


# bottom topography
topo = 'flat'
if topo=='flat':
    H = 1e2 * np.ones([ny, nx])    # m


# In[7]:


# initial condition of the height of the fluid
std_blob = 25*dely    # Standard deviation of blob (m)
h = 2*np.exp(-((X-np.mean(xgrid))**2+(Y-np.mean(ygrid))**2)/
           (2*std_blob**2))

# coriolis force
#Cor = f0+Y*0
Cor = f0 + beta*Y
#Cor = f0 + beta*(Y-ygrid.mean())

# initialize
u = np.zeros([ny, nx])
v = np.zeros([ny, nx])


# In[24]:


ii = 0
time = []
for n, it in enumerate(tidx):

    u_tmp = u.copy()
    v_tmp = v.copy()
    h_tmp = h.copy()

    # Heun method

    # C-grid
    [u_accel, v_accel, h_accel] = rhs_cgrid(u_tmp, v_tmp, h_tmp, Cor, delx, dely)

    u_p = np.zeros([ny,nx])
    v_p = np.zeros([ny,nx])
    h_p = np.zeros([ny,nx])
    u_p[1:-1,1:-1] = u_tmp[1:-1,1:-1] + dt*u_accel
    v_p[1:-1,1:-1] = v_tmp[1:-1,1:-1] + dt*v_accel
    h_p[1:-1,1:-1] = h_tmp[1:-1,1:-1] + dt*h_accel

    [u_p, v_p, h_p] = bcapply(u_p, v_p, h_p, bcflag)

    [u_p_accel, v_p_accel, h_p_accel]        =rhs_cgrid(u_p, v_p, h_p, Cor, delx, dely)

    u[1:-1,1:-1] += 0.5*dt*(u_accel + u_p_accel)
    v[1:-1,1:-1] += 0.5*dt*(v_accel + v_p_accel)
    h[1:-1,1:-1] += 0.5*dt*(h_accel + h_p_accel)

    [u, v, h] = bcapply(u, v, h, bcflag)
        
    if np.remainder(it, dumpfreq)==0:
        time.append(it)
        if ii==0:
            u_save = u.copy()
            v_save = v.copy()
            h_save = h.copy()
            ii+=1
        else:
            u_save = np.dstack((u_save, u))
            v_save = np.dstack((v_save, v))
            h_save = np.dstack((h_save, h))

        plt.clf()
        plt.subplot(3,1,1)
        plt.imshow(u, origin='lower')
        plt.subplot(3,1,2)
        plt.imshow(v, origin='lower')
        plt.subplot(3,1,3)
        plt.imshow(h, origin='lower',vmin=-1, vmax=1, cmap='bwr')
        #plt.subplot(3,1,4)
        #plt.plot(h[:,30])
        #plt.ylim([-300,300])
        plt.pause(.1)

