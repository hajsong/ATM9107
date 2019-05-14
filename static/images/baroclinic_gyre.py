
# coding: utf-8

# # MITgcm #2: Baroclinic Ocean gyre
# We will set up the model for the four-layer ocean forced by wind stress.

# ## 1. Problem
# This ocean is driven by the zonal wind stress

# ### 1.1 Equations
# $$
# \frac{D u}{D t} - fv + \frac{1}{\rho}\frac{\partial p'}{\partial \lambda} - A_h\nabla^2_h u - A_z\frac{\partial^2 u}{\partial z^2} = \mathcal{F}_{\lambda} \\
# \frac{D v}{D t} + fu + \frac{1}{\rho}\frac{\partial p'}{\partial \phi} - A_h\nabla^2_h v - A_z\frac{\partial^2 v}{\partial z^2} = 0 \\
# \frac{\partial \eta}{\partial z} + \frac{\partial H\hat{u}}{\partial \lambda} + \frac{\partial H \hat{v}}{\partial \phi} = 0 \\
# \frac{D \theta}{D t} - K_h\nabla^2_h \theta - K_z\frac{\partial^2 \theta}{\partial z^2} = 0 \\
# p' = \rho_0 g \eta + \int^0_{-z} \rho' dz \\
# \rho' = -\alpha_{\theta}\rho_0\theta' \\
# \mathcal{F}_{\lambda} = \frac{\tau_{\lambda}}{\rho_0\Delta z}
# $$

# ## 2. Settings 
# This example has the setting similar to the barotropic gyre experiment, but it is on the spherical coordinate, and it is no more on the beta plane.  
# There are vertically four levels.

# ### 2.1 Model grid
# ![domain](domain_bc.png)  
# 
# We consider the circulation in a closed domain. To do so, we have to set up the wall at each boundary. This can be done with the topography file.  
# 
# On the spherical coordinate, the Coriolis parameter, $f$ is expresses as
# $$
# f(\phi) = 2\Omega\sin(\phi),
# $$
# where the rotation rate, $\Omega$ set to $2\pi / (86400\ s)$.  
# Then the wind stress is 
# $$
# \tau_{\lambda}(\phi) = \tau_0 \sin\left( \pi \frac{\phi}{L_{\phi}} \right),
# $$
# where $L_{\phi}$ is the lateral domain extent (60$^{\circ}$) and $\tau_0 = 0.1$ N m$^{-2}$.  
# Initially, a constant value is assigned to each layer as the temperature as shown in the figure.  
# We do not simulate the salinity, so that the water density is estimated using
# $$
# \rho' = -\rho_0 \alpha_{\theta} \theta',
# $$
# with $\rho_0 = 999.8$ kg m$^{-3}$ and $\alpha_{\theta} = 2\times 10^{-4}$ degC$^{-1}$.
# 
# For the generation of the model grid and wind forcing file, you may use the Matlat script, ```gendata.m``` in the ```input``` directory.

# ### 2.2 Compilation
# We employ the resolution of 1 degree in both longitudinal and latitudianl directions.  
# It means that there should be 60 grid points in each direction.  
# 
# In the ```code``` directory, you can find ```SIZE.h``` file. Make sure that the setting in this file is consistent with our intended simulation.  
# This has to be edited according to your computing envirionment.

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
# 
# Let's include the package ```diagnostics``` and ```gfd```.

# ## 3. Running the model

# ### 3.1 Prepare the run directory
# You may copy all the input files(```data```, ```data.pkg```,...) to the run directory.  
# Also do not forget to copy the executable file from the build directory to the run directory.

# ### 3.2 Setting up the ```data``` file
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

# ### 3.3 Setting up the ```data.pkg``` file
# In some cases, you may skip to use a certain package even though you included it during compilation.
# In the ```data.pkg``` file, you can specify what packages you want to use among those being compiled.  
# 
# In my case, I am not using the netcdf package although it was included in the compilation. So I will edit this file not to include the netcdf file in the run.
# ```
#  useDIAGNOSTICS=.TRUE.,
#  
# ```

# ### 3.4 Setting up the ```data.diagnostics``` file
# You can copy and paste the following lines after creating a file called ```data.diagnostics```
# ```
# # Diagnostic Package Choices
# #--------------------
# #  dumpAtLast (logical): always write output at the end of simulation (default=F)
# #  diag_mnc   (logical): write to NetCDF files (default=useMNC)
# #--for each output-stream:
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
#  &DIAGNOSTICS_LIST
#    fields(1,1) = 'ETAN    ',
#    fileName(1) = 'Diags/eta',
#   frequency(1) = 2592000.,
# 
#    fields(1,2) = 'ETAN    ',
#    fileName(2) = 'Diags/eta_tp',
#   frequency(2) = 2592000.,
#   timePhase(2) = 1296000.,
# 
#    fields(1:3,3) = 'UVELMASS','VVELMASS','WVELMASS',
#    fileName(3) = 'Diags/vel',
#   frequency(3) = 2592000.,
# 
#    fields(1:2,4) = 'THETA   ','RHOAnoma',
#    fileName(4) = 'Diags/theta',
#   frequency(4) = 2592000.,
#  &
#  ```

# ### 3.5 Run
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

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from MITgcmutils import rdmds
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


rdir = '/Users/hajsong/Yonsei/Classes/ATM9107/mylectures/MITgcm_3Dgyre/run/'
dt = 1200
XC = rdmds(rdir+'XC')
YC = rdmds(rdir+'YC')
RC = rdmds(rdir+'RC')
rho0 = 999.8
alphaT = 2e-4
g=9.81


# In[3]:


Eta, its, _ = rdmds(rdir+'Eta', np.nan, returnmeta=True)    
# Read all the sea level output


# In[4]:


f, ax = plt.subplots(4, 3, figsize=(15,15))
ax = ax.flatten()
for i in xrange(len(ax)):
    ax[i].contour(XC, YC, Eta[i+1,...], np.arange(-.1,.11,.01), colors='black')
    ax[i].set_title('Sea level at t='+str((its[i+1]*dt)/86400.))


# ### 3.2 Packages for diagnostics
# We will look at the monthly mean model outputs.

# ### 3.3 Sea level and mean current

# #### 3.3.1 Read the sea level data

# In[5]:


eta, its, _ = rdmds(rdir+'Diags/eta', np.nan, returnmeta=True)
# eta, its, _ = rdmds(rdir+'Eta', np.nan, returnmeta=True)


# #### 3.3.2 Read the velocity data

# In[6]:


u = rdmds(rdir+'Diags/vel', np.nan, rec=0)
v = rdmds(rdir+'Diags/vel', np.nan, rec=1)


# In[7]:


# We want to put u and v to the center point of the grid cell for the interior points.
uc = 0.5*(u[:, :, 1:-1, 2:] + u[:, :, 1:-1, 1:-1])
vc = 0.5*(v[:, :, 2:, 1:-1] + v[:, :, 1:-1, 1:-1])
spd = np.sqrt(uc**2 + vc**2)


# #### 3.3.3 Read the density anomaly data

# In[8]:


rho = rdmds(rdir+'Diags/theta', np.nan, rec=1)


# #### 3.3.4 Make monthly mean plots

# In[9]:


vi = 4    # the interval between points in the vector plots
ik=0      # the index for the level that we make plots

f, ax = plt.subplots(4, 3, figsize=(15,15))
ax = ax.flatten()
for i in xrange(len(ax)):
    ax[i].quiver(XC[1:-1:vi, 10:-1:vi], YC[1:-1:vi,10:-1:vi],
                 uc[i,ik,::vi,9::vi], vc[i,ik,::vi,9::vi], 
                 spd[i,ik,::vi,9::vi], cmap='rainbow',scale=.1)
    ax[i].contour(XC[1:-1,1:-1], YC[1:-1,1:-1], 
                  rho0*g*eta[i,1:-1,1:-1] + 
                  5*np.sum(rho[i,0:ik,1:-1,1:-1], axis=0), 10, cmap='Blues')
#     ax[i].contour(XC, YC, eta[i,...], np.arange(-1e-2,1.1e-2,1e-2), colors='black')
    endt = its[i]*dt/86400.
    strt = endt - 30
    ax[i].set_title('mean Sea level, t=['+str(strt)+':'+str(endt)+']')


# ### 3.4 Kinetic energy and energy power spectrum

# #### 3.4.1 Kinetic energy time series

# In[10]:


mke = [np.mean(0.5*spd[i,:,:]) for i in xrange(np.shape(spd)[0])]   


# In[11]:


plt.plot(np.asarray(its)*dt/86400, np.asarray(mke))                            


# #### 3.4.2 Kinetic energy power spectrum

# In[12]:


#
#  Frequency
#
dx = (XC[0,1] - XC[0,0])   # degree
dy = (YC[1,0] - YC[0,0])
ny, nx = XC[1:-1,1:-1].shape
p=np.fft.fftfreq(nx,d=dx);
q=np.fft.fftfreq(ny,d=dy);
p=np.fft.fftshift(p);
q=np.fft.fftshift(q);
k=p[p>=0];
delk=k[1]-k[0];
[pp,qq]=np.meshgrid(p,q)
pqs=np.sqrt(pp**2+qq**2);


# In[13]:


#
# The power spectrum for the kinetic energy
#
iz = 0  # the index of the vertical level for the power spectrum

kepower=np.empty((len(k),len(its)));
for i in xrange(len(its)):
    #
    #  Detrend (remove slope, Errico (1985))
    #
    up=uc[i,iz,:,:]*0
    vp=vc[i,iz,:,:]*0
    [ly,lx]=up.shape;
    for ix in xrange(0,lx):
        su=(uc[i,iz,-1,ix]-uc[i,iz,0,ix])/(ly-1)
        sv=(vc[i,iz,-1,ix]-vc[i,iz,0,ix])/(ly-1)
        up[:,ix]=uc[i,iz,:,ix]-(2*np.arange(1,ly+1)-ly-1)*su/2
        vp[:,ix]=vc[i,iz,:,ix]-(2*np.arange(1,ly+1)-ly-1)*sv/2

    for iy in xrange(0,ly):
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
    for ik in xrange(len(k)):
        krng=[k[ik],k[ik]+delk]
        [yy,xx]=np.nonzero((pqs>krng[0])&(pqs<=krng[1]))
        for s in xrange(len(yy)):
            up1d[ik]=up1d[ik]+upower[yy[s],xx[s]]
            vp1d[ik]=vp1d[ik]+vpower[yy[s],xx[s]]

    kepower[:,i]=(up1d+vp1d)/2.0


# In[14]:


h=plt.loglog(k,kepower[:,-1])


# In[15]:


f, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].imshow(u[-1,0,:,:], origin='lower', vmin=-0.1, vmax=0.1, cmap='bwr')
ax[1].imshow(v[-1,0,:,:], origin='lower', vmin=-0.1, vmax=0.1, cmap='bwr')


# ### 3.5 Vertical velocity

# #### 3.5.1 Read vertical velocity

# In[16]:


wvel = rdmds(rdir+'Diags/vel', np.nan, rec=2)


# #### 3.5.2 Estimate the vertical velocity at the surface using the sea level

# In[17]:


eta_snap = rdmds(rdir+'Diags/eta_s',np.nan)


# In[18]:


wvel_eta = (eta_snap[1:,:,:] - eta_snap[:-1,:,:])/((its[1]-its[0])*dt)


# #### 3.5.3 Estimate the vertical velocity using divergence of the flow
# $$
# \frac{\partial \eta}{\partial z} + \frac{\partial H\hat{u}}{\partial \lambda} + \frac{\partial H \hat{v}}{\partial \phi} = 0 \\
# $$

# In[19]:


# We need to know the width and the thickness of the grid to compute the vertically integrated momentum flux.
DXG = rdmds(rdir+'DXG')
DYG = rdmds(rdir+'DYG')
DRF = rdmds(rdir+'DRF')
RAC = rdmds(rdir+'RAC')
[nt, nz, ny, nx] = wvel.shape


# In[20]:


DIV = np.zeros([nt, nz, ny, nx])
wvel_div = np.zeros([nt, ny, nx])


for it in xrange(nt):
    for k in xrange(nz):
        Ut = u[it, k, :, :]*DYG*DRF[k]
        Vt = v[it, k, :, :]*DXG*DRF[k]
        DIV[it, k, :-1, :-1] = Ut[0:ny-1, 0:nx-1]-Ut[0:ny-1, 1:nx]                            +Vt[0:ny-1, 0:nx-1]-Vt[1:ny, 0:nx-1]

    wvel_div[it, :, :] = np.sum(DIV[it,:,:,:], axis=0)/RAC


# #### 3.5.4 Compare the estimated vertical velocities.

# In[21]:


f, ax = plt.subplots(1,3,figsize=(15,5))
ax0 = ax[0].imshow(wvel[3,0,...], vmin=-1e-8, vmax=1e-8, cmap='bwr')
ax1 = ax[1].imshow(wvel_eta[3,...], vmin=-1e-8, vmax=1e-8, cmap='bwr')
ax2 = ax[2].imshow(wvel_div[3,...], vmin=-1e-8, vmax=1e-8, cmap='bwr')


# ## 4. Exercise

# What if
# ```
#  usingCartesianGrid=.TRUE.,
#  usingSphericalPolarGrid=.FALSE.,
#  delX=60*20E3,
#  delY=60*20E3,
# ```

# What if 
# ```
#  ygOrigin=30.,
#  delX=60*0.2,
#  delY=60*0.2,
# ```

# What if $A_h = 100$ m$^2$ s$^{-1}$?

# What if the thickness of the first layer is 5 m?
