
# coding: utf-8

# # MITgcm #1: Barotropic Ocean gyre
# We will set up the model for a single layer ocean forced by wind stress.

# ## 1. Problem
# We simulate the wind-driven circulation in the mid-latitude setting within the basin whose size is 1200 km x 1200 km.  
# See the page 117 in the MITgcm manual for a complete description.

# ### 1.1 Equations
# ![eqn](eqn.png)

# ## 2. Settings 
# This experiment was designed to reproduce the problem described
# analytically by *Stommel in 1948* and numerically in *Holland et. al, 1978*.

# ### 2.1 Model grid
# ![domain](domain.png)
# 
# We consider the wind forcing with only x-component and it varies with latitude following 
# $$
#  \tau_x(y) = \tau_0 \sin\left(\pi \frac{y}{L_y} \right),
# $$
# where $\tau_0$ = 0.1 N m$^{-2}$, and $L_y$ = 1200 km.  
# 
# This domain has a flat bottom topography, and we consider the beta-plane so that
# $$
# f(y) = f_0 + \beta y,
# $$
# where $f_0=10^{-4}$ s$^{-1}$ and $\beta = 10^{-11}$ s$^{-1}$ m$^{-1}$.
# 
# (For the generation of the model grid and wind forcing file, you may use the Matlat script, ```gendata.m``` in the ```input``` directory.)  
# 
# The default resolution is set to $\Delta x = \Delta y = 20$ km. This means that there should be 60 points in each direction. You need to specify the size of the model.

# ### 2.2 Compilation
# In the ```code``` directory, you can find ```SIZE.h``` file. In here, you can specify the size of the model.  
# In addition to specify the size of the model, you can divide your model for the parellel computation.  
# If you want to change the resolution, don't forget to edit this file accordingly.

# ### 2.2 Dissipation
# Dissipation form that is commonly used in the MITgcm.
# $$
# D_v = A_h\nabla^2_hv + A_v\frac{\partial^2 v}{\partial z^2} + A_4\nabla^4_h v,
# $$
# where $A_h$ and $A_v$ are horizontal and vertical viscosity coefficients and $A_4$ is the horizontal coefficient for biharmonic friction.  
# 
# In ```data```, ```viscAh``` and ```viscAz``` represent $A_h$ and $A_v$, respectively.  
# ```viscA4``` is for $A_4$, and it is zero otherwise specified.

# ### 2.3 packages
# MITgcm can call varied packages (e.g. sea-ice model, biogeochemical model).  
# There are two places to specify the packages that you want to use in the model.
# 1. ```code/packages.conf``` : you need to list the name of the packages so that the appropriate packages are to be compiled.
# 2. ```input/data.pkg``` : you may skip the packages in the actual run using this file even if those packages were compiled. 

# ## 3. Running the model

# ### 3.1 Setting up the ```data``` file
# First, let's set 
# ``` 
# endTime=31104000,
# monitorFreq=2592000.0,
# ```
# By assigning ```endTime=31104000```, the model will be integrated for one year (unit for ```endTime``` is *second*). 
# Alternatively, you can provide the total number of time step. 
# In this case, ```nTimeSteps=25920``` gives you the same length of integration.  
# 
# By default, each cpu will write the output from the tile it is assigned. If you are using more than one cpu, it means that you will get multiple output files for a single time step.  
# If you want to avoid this, you can let only the master MPI write the output after combining the values from all tiles. To do this, you need to add the following line
# ```
# useSingleCpuIO=.TRUE.,
# ```
# under ```PARM01``` in ```data``` file.

# ### 3.2 Setting up the ```data.pkg``` file
# In some cases, you may skip to use a certain package even though you included it during compilation.
# In the ```data.pkg``` file, you can specify what packages you want to use among those being compiled.  
# 
# In my case, I am not using the netcdf package although it was included in the compilation. So I will edit this file not to include the netcdf file in the run.
# ```
#  useMNC=.FALSE.,
# ```

# ### 3.3 Run
# If your executable file name is ```mitgcmuv```, you can run the model and pring the output in a text file.
# ```
# ./mitgcmuv > stdout
# ```
# The run finishes successfully if you see the following phrase.
# ```
# STOP NORMAL END
# ```

# ## 3. Analysis

# ### 3.1 Using ```rdmds```
# Once your run is done, you want to check the output. ```MITgcm``` provides a very useful fucntion called ```rdmds```.
# There are both Matlab and Python version of this function, and they can be found under the folder ```utils```.  
# For Python, you can set up the ```PYTHONPATH``` in your shell environment file.  
# 
# Also, you may find the python script that I wrote to read the grid variables useful.  
# You can get it from
# https://github.com/hajsong/MITgcmdiag/tree/master/pytools .
# Then you have to edit the file ```mitgcmgrid.py``` for this experiment.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from MITgcmutils import rdmds
from hspython import loadgrid
get_ipython().run_line_magic('matplotlib', 'inline')

grd = loadgrid('gyre2d', varname=['XC', 'YC'])
# In[3]:


rdir = '/Users/hajsong/Yonsei/Classes/ATM9107/mylectures/MITgcm_2Dgyre/run/'
dt = 1200
XC = rdmds(rdir+'XC')
YC = rdmds(rdir+'YC')


# In[4]:


Eta, its, _ = rdmds(rdir+'Eta', np.nan, returnmeta=True)    
# Read all the sea level output


# In[5]:


Eta.shape


# In[7]:


f, ax = plt.subplots(4, 3, figsize=(15,15))
ax = ax.flatten()
for i in range(len(ax)):
    ax[i].contour(XC*1e-3, YC*1e-3, Eta[i+1,...], np.arange(-.1,.11,.01), colors='black')
    ax[i].set_title('Sea level at t='+str((its[i+1]*dt)/86400.))


# ### 3.2 Packages for diagnostics
# In the default setting, the model dumps out only snapshots. 
# For the time averaged field, you need to use either ```timeavg``` or ```Diagnostics``` package.

# #### 3.2.1 ```timeavg``` package
# If you want to save the time averages, you turn this package on by setting the time frequency for the time average under ```PARM03``` in ```data``` file.
# For the monthly mean data, you can 
# ```
# taveFreq=2592000.0,
# ```
# But! this package does not provide flexibility much. So let's try the second option.

# #### 3.2.2 ```diagnostic``` package
# This package is quite useful because you can save not only standard variables but post-processed variables as both snapshots and time-mean.  To turn on this package during the run, you need to set 
# ```
# useDIAGNOSTICS=.TRUE.,
# ```
# in ```data.pkg```.
# A file called ```data.diagnostics``` is necessary to inform MITgcm your choice on the diagnostic variables.  
# You may copy the following file to your run directory.
# ```
# tutorial_advection_in_gyre/input/data.diagnostics
# ```
# 
# If you open this ```data.diagnostics``` file, you can see these lines on top.
# ```
# #  fileName(n) : prefix of the output file name (max 80c long) for outp.stream n
# #  frequency(n):< 0 : write snap-shot output every |frequency| seconds
# #               > 0 : write time-average output every frequency seconds
# #  timePhase(n)     : write at time = timePhase + multiple of |frequency|
# #    averagingFreq  : frequency (in s) for periodic averaging interval
# #    averagingPhase : phase     (in s) for periodic averaging interval
# #    repeatCycle    : number of averaging intervals in 1 cycle
# #  levels(:,n) : list of levels to write to file (Notes: declared as REAL)
# #                when this entry is missing, select all common levels of this list
# #  fields(:,n) : list of selected diagnostics fields (8.c) in outp.stream n
# #                (see "available_diagnostics.log" file for the full list of diags)
# #  missing_value(n) : missing value for real-type fields in output file "n"
# #  fileFlags(n)     : specific code (8c string) for output file "n"
# #--------------------
# ```
# It provides a lot of flexibility to analyze the model result, and we want to edit this file according to our goal.  
# But let's just try to run the model first with this ```data.diagnostics```.

# You will get an error, but you can see that you have a new file called ```available_diagnostics.log```.  
# This file shows you the list of diagnostic variables MITgcm offers with the current setup.  
# Then, you can choose the diagnostic variables from the list and edit ```data.diagnostics``` accordingly.  
# In my case, I chose to save ```ETAN```.  
# 
# **It is important to have exact 8 spaces in the double quoation marks!**

# If you try to have many diagnostic variables, you may have to edit the file ```DIAGNOSTICS_SIZE.h``` that can be found under the directory ```pkg/diagnostics```. In this case, you need to compile the code again.

# In[8]:


eta, its, _ = rdmds(rdir+'Diags/eta', np.nan, returnmeta=True)


# In[9]:


eta.shape


# In[10]:


u = rdmds(rdir+'Diags/vel', np.nan, rec=0)
v = rdmds(rdir+'Diags/vel', np.nan, rec=1)


# In[11]:


# We want to put u and v to the center point of the grid cell for the interior points.
uc = 0.5*(u[:, 1:-1, 2:] + u[:, 1:-1, 1:-1])
vc = 0.5*(v[:, 2:, 1:-1] + v[:, 1:-1, 1:-1])
spd = np.sqrt(uc**2 + vc**2)


# In[13]:


f, ax = plt.subplots(4, 3, figsize=(15,15))
ax = ax.flatten()
vi = 5    # the interval between points in the vector plot
for i in range(len(ax)):
    ax[i].quiver(XC[1:-1:vi, 1:-1:vi]*1e-3, YC[1:-1:vi,1:-1:vi]*1e-3, 
                 uc[i,::vi,::vi], vc[i,::vi,::vi], spd[i,::vi,::vi], cmap='rainbow')
    ax[i].contour(XC*1e-3, YC*1e-3, eta[i,...], np.arange(-.1,.11,.01))
    endt = its[i]*dt/86400.
    strt = endt - 30
    ax[i].set_title('mean Sea level, t=['+str(strt)+':'+str(endt)+']')


# In[14]:


mke = [np.mean(0.5*spd[i,:,:]**2) for i in range(np.shape(spd)[0])]   


# In[15]:


plt.plot(np.asarray(its)*dt/86400, np.asarray(mke))                            


# In[16]:


#
#  Frequency
#
dx = (XC[0,1] - XC[0,0])*1e-3    # from m to km
dy = (YC[1,0] - YC[0,0])*1e-3
ny, nx = XC[1:-1,1:-1].shape
p=np.fft.fftfreq(nx,d=dx);
q=np.fft.fftfreq(ny,d=dy);
p=np.fft.fftshift(p);
q=np.fft.fftshift(q);
k=p[p>=0];
delk=k[1]-k[0];
[pp,qq]=np.meshgrid(p,q)
pqs=np.sqrt(pp**2+qq**2);


# In[18]:


#
# The power spectrum for the kinetic energy
#
kepower=np.empty((len(k),len(its)));
for i in range(len(its)):
    #
    #  Detrend (remove slope, Errico (1985))
    #
    up=uc[i,:,:]*0
    vp=vc[i,:,:]*0
    [ly,lx]=up.shape;
    for ix in range(0,lx):
        su=(uc[i,-1,ix]-uc[i,0,ix])/(ly-1)
        sv=(vc[i,-1,ix]-vc[i,0,ix])/(ly-1)
        up[:,ix]=uc[i,:,ix]-(2*np.arange(1,ly+1)-ly-1)*su/2
        vp[:,ix]=vc[i,:,ix]-(2*np.arange(1,ly+1)-ly-1)*sv/2

    for iy in range(0,ly):
        su=(up[iy,-1]-up[iy,0])/(lx-1)
        sv=(vp[iy,-1]-vp[iy,0])/(lx-1)
        up[iy,:]=up[iy,:]-(2*np.arange(1,lx+1)-lx-1)*su/2
        vp[iy,:]=vp[iy,:]-(2*np.arange(1,lx+1)-lx-1)*sv/2
    #
    #  Do FFT
    #
    upower=(np.fft.fft2(up))
    vpower=(np.fft.fft2(vp))
    upower=np.abs(upower)**2/lx/ly
    vpower=np.abs(vpower)**2/lx/ly
    upower=np.fft.fftshift(upower);
    vpower=np.fft.fftshift(vpower);
    #
    #  merge 2d power to 1d
    #
    [pp,qq]=np.meshgrid(p,q)
    pqs=np.sqrt(pp**2+qq**2);
    up1d=np.zeros(len(k));up1d[0]=upower.max()
    vp1d=np.zeros(len(k));vp1d[0]=vpower.max()
    for ik in range(len(k)):
        krng=[k[ik],k[ik]+delk]
        [yy,xx]=np.nonzero((pqs>krng[0])&(pqs<=krng[1]))
        for s in range(len(yy)):
            up1d[ik]=up1d[ik]+upower[yy[s],xx[s]]
            vp1d[ik]=vp1d[ik]+vpower[yy[s],xx[s]]

    kepower[:,i]=(up1d+vp1d)/2.0


# In[19]:


h=plt.loglog(k,kepower[:,-1])


# ## 4. Exercise

# What if $\beta = 0$?

# What if $A_h = 100$ m$^2$ s$^{-1}$?

# what if both $\beta=0$ and $A_h$ = 100 m$^2$ s$^{-1}$?

# What if we use sperical polar grid?
# 
# ```
# usingSphericalPolarGrid=.TRUE.,
# ygOrigin=0.,
# delX=60*1.,
# delY=60*1.,
# ```
