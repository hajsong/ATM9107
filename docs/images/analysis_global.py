
# coding: utf-8

# # Analyzing the global ocean model output

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from MITgcmutils import rdmds
import cartopy.crs as ccrs
import cartopy.feature as cfeature
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Parameters

# In[3]:


rdir = '/Users/hajsong/Yonsei/Classes/ATM9107/mylectures/MITgcm_global/run/'
dt = 86400


# ### Load the model grid

# In[104]:


XC = rdmds(rdir+'XC')
YC = rdmds(rdir+'YC')
RC = rdmds(rdir+'RC')
XG = rdmds(rdir+'XG')
hFacC = rdmds(rdir+'hFacC')
hFacS = rdmds(rdir+'hFacS')
hFacW = rdmds(rdir+'hFacW')
Depth = rdmds(rdir+'Depth')
DRF = rdmds(rdir+'DRF')
DXG = rdmds(rdir+'DXG')
DYG = rdmds(rdir+'DYG')
RAC = rdmds(rdir+'RAC')

RC = np.squeeze(RC)
DRF = np.squeeze(DRF)

nz, ny, nx = hFacC.shape

CellVol = np.zeros([nz, ny, nx])
for ik in range(nz):
    CellVol[ik,:,:] = RAC*DRF[ik]*hFacC[ik,:,:]
CellVol[np.nonzero(CellVol==0)] = np.nan


# In[48]:


mskC = hFacC.copy()
mskC[hFacC==0]=np.nan
mskS = hFacS.copy()
mskS[hFacS==0]=np.nan
mskW = hFacW.copy()
mskW[hFacW==0]=np.nan


# In[77]:


# Atlantic mask
atlantic_hfacs = hFacS.copy()
ixso = 4,73    # southern ocean
iyso = 0,24
atlantic_hfacs[:,iyso[0]:iyso[-1],ixso[0]:ixso[-1]] = np.nan
ixnp = 13,65    # northern pacific
iynp = 24,36
atlantic_hfacs[:,iynp[0]:iynp[-1],ixnp[0]:ixnp[-1]] = np.nan
ixsoc = 0,90    # complete southern ocean
iysoc = 0,12
atlantic_hfacs[:,iysoc[0]:iysoc[-1],ixsoc[0]:ixsoc[-1]] = np.nan

# Pacific mask  
pacific_hfacs = hFacS.copy();
ixatl1 = 0,26  # most of altantic and indian ocean
ixatl2 = 73,90 # most of altantic and indian ocean
iyatl = 0,ny
pacific_hfacs[:,iyatl[0]:iyatl[-1],ixatl1[0]:ixatl1[-1]] = np.nan;
pacific_hfacs[:,iyatl[0]:iyatl[-1],ixatl2[0]:ixatl2[-1]] = np.nan;
ixatlf = 72; # small fix
iyatlf = 30;
pacific_hfacs[:,iyatlf,ixatlf] = np.nan;
ixgm = 65,73; # Gulf of Mexico
iygm = 24,30;
pacific_hfacs[:,iygm[0]:iygm[-1],ixgm[0]:ixgm[-1]] = np.nan;
ixhb = 67,73; # Hudson Bay and Baffin Bay
iyhb = 35,40;
pacific_hfacs[:,iyhb[0]:iyhb[-1],ixhb[0]:ixhb[-1]] = np.nan;
ixsop = 0,nx; # southern ocean
iysop = 0,12;
pacific_hfacs[:,iysop[0]:iysop[-1],ixsop[0]:ixsop[-1]] = np.nan;
ixio = 26,32; # rest of indian ocean
iyio = 12,19;
pacific_hfacs[:,iyio[0]:iyio[-1],ixio[0]:ixio[-1]] = np.nan;

# Indian ocean
indic_hfacs = hFacS.copy();
ixsoi = 0,90; # southern ocean
iysoi = 0,12;
indic_hfacs[:,iysoi[0]:iysoi[-1],ixsoi[0]:ixsoi[-1]] = np.nan;
ixap1 = 0,4   # atlantic and pacific
ixap2 = 32,90 # atlantic and pacific
iyap = 0,40;
indic_hfacs[:,iyap[0]:iyap[-1],ixap1[0]:ixap1[-1]] = np.nan;
indic_hfacs[:,iyap[0]:iyap[-1],ixap2[0]:ixap2[-1]] = np.nan;
iynps = 34,40; # north polar sea
indic_hfacs[:,iynps[0]:iynps[-1],:] = np.nan
ixmeds = 0,10;  # mediterraenan
iymeds = 28,31;
indic_hfacs[:,iymeds[0]:iymeds[-1],ixmeds[0]:ixmeds[-1]] = np.nan;
ixscs = 26,32; # south china sea
iyscs = 20,29;
indic_hfacs[:,iyscs[0]:iyscs[-1],ixscs[0]:ixscs[-1]] = np.nan;
indic_hfacs[:,38,4] = np.nan # singular point near Murmansk


# ### Read T and S

# In[8]:


T, its, _ = rdmds(rdir+'T', np.nan, returnmeta=True)
S = rdmds(rdir+'S', np.nan)


# ### Plot the surface field

# In[46]:


plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m')
c = ax.pcolormesh(XC, YC, T[-1, 0, :, :]*mskC[0, :, :], vmin=-1, vmax=30, cmap='rainbow',
              transform=ccrs.PlateCarree())
# ax.stock_img()
ax.add_feature(cfeature.LAND)
ax.set_title('Sea Surface Temperature, year='+str(its[-1]/360))
ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
plt.colorbar(c, ax=ax, shrink=0.8)


# In[47]:


plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m')
c = ax.pcolormesh(XC, YC, S[-1, 0, :, :]*mskC[0, :, :], vmin=32, vmax=36, cmap='jet',
              transform=ccrs.PlateCarree())
# ax.stock_img()
ax.add_feature(cfeature.LAND)
ax.set_title('Sea Surface Salinity, year='+str(its[-1]/360))
ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
plt.colorbar(c, ax=ax, shrink=0.8)


# ### Barotropic streamfunction
# Barotropic streamfunction depicts the horizontal ocean circulation. In a mathematical form, the streamfunction $\psi_{xy}$ satisfies
# $$
# \frac{\partial \psi_{xy}}{\partial y} = -\int^{\eta}_{-H} u dz \\
# \frac{\partial \psi_{xy}}{\partial x} = \int^{\eta}_{-H} v dz \\
# $$
# In practice, barotropic streamfunction is computed by integrating the zonal barotropic flow from north to south.

# In[61]:


# Compute the volume of each grid cell
vol = np.zeros(hFacC.shape)
for k in range(nz):
    vol[k, :, :] = DYG * DRF[k] * hFacW[k, :, :]

# vertical integration of U
u = rdmds(rdir+'U', its[-1])
ubar = np.sum(u*vol, axis=0)

# Integration from north to south
psi = np.zeros(ubar.shape)
psi = np.cumsum(np.flipud(ubar), axis=0)
psi = np.flipud(psi)


# In[71]:


plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m')
c = ax.contourf(XG, YC, psi*mskW[0, :, :]*1e-6, np.arange(-130, 131,10), 
                  cmap='bwr', extend='both', transform=ccrs.PlateCarree())
# ax.stock_img()
ax.add_feature(cfeature.LAND)
ax.set_title('Barotropic streamfunction (Sv), year='+str(its[-1]/360))
ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
plt.colorbar(c, ax=ax, shrink=0.8)


# ### Meridional overturning circulation
# We can compute the streamfunction $\psi_{zy}$ to show the meridional overturning circulation. $\psi_{zy}$ satisfies
# $$
# \frac{\partial \psi_{zy}}{\partial y} = \int^{x_2}_{x_1} w dx \\
# \frac{\partial \psi_{zy}}{\partial z} = -\int^{x_2}_{x_1} v dx
# $$
# In practice, we can compute $\phi$ using $v$ by integrating from the bottom to the top.

# In[72]:


# Read the annual mean v
vvelmass = rdmds(rdir+'Diags/Velmass', 3600, rec=1)    # vvelmass = vvel * hFacS
[nz, ny, nx] = vvelmass.shape
psiy = rdmds(rdir+'Diags/GMvel', 3600, rec=1)    # Bolus transport streamfunction, V(m^2/s)


# In[81]:


# mean overturning circulation
mvar = np.zeros([nz+1, ny])        # global
mvar_atl = np.zeros([nz+1, ny])    # Atlantic
mvar_pac = np.zeros([nz+1, ny])    # Pacific
for k in range(nz-1,-1,-1):
    velwgt = vvelmass[k,:,:] * DRF[k] * DXG * mskS[k,:,:]
    velwgt_sum = np.nansum(velwgt, axis=1)
    mvar[k,:] = mvar[k+1,:] - velwgt_sum
    
    velwgt_atl = vvelmass[k,:,:] * DRF[k] * DXG * atlantic_hfacs[k,:,:]
    velwgt_atl = np.nansum(velwgt_atl, axis=1)
    mvar_atl[k,:] = mvar_atl[k+1,:] - velwgt_atl
    
    velwgt_pac = vvelmass[k,:,:] * DRF[k] * DXG * pacific_hfacs[k,:,:]
    velwgt_pac = np.nansum(velwgt_pac, axis=1)
    mvar_pac[k,:] = mvar_pac[k+1,:] - velwgt_pac


# In[89]:


# circulation by eddies
gm_psiy = np.zeros([nz+1,ny, nx])
gm_psiy[:nz,:,:] = psiy;
bolusv = psiy*0;
for k in range(nz):
    bolusv[k,:,:] = (gm_psiy[k+1,:,:] - gm_psiy[k,:,:])/DRF[k];
    
pvar = np.zeros([nz+1,ny])
pvar_atl = np.zeros([nz+1,ny])
pvar_pac = np.zeros([nz+1,ny])
for k in range(nz-1,-1,-1):
    velwgt = bolusv[k,:,:] * DRF[k] * DXG * mskS[k,:,:]
    velwgt_sum = np.nansum(velwgt, axis=1)
    pvar[k,:] = pvar[k+1,:] - velwgt_sum
    
    velwgt_atl = bolusv[k,:,:] * DRF[k] * DXG * atlantic_hfacs[k,:,:]
    velwgt_atl = np.nansum(velwgt_atl, axis=1)
    pvar_atl[k,:] = pvar_atl[k+1,:] - velwgt_atl
    
    velwgt_pac = bolusv[k,:,:] * DRF[k] * DXG * pacific_hfacs[k,:,:]
    velwgt_pac = np.nansum(velwgt_pac, axis=1)
    pvar_pac[k,:] = pvar_pac[k+1,:] - velwgt_pac


# In[88]:


[X, Y] = np.meshgrid(YC[:,0], RC)

f, ax = plt.subplots(3, 1,figsize=(15,12))
ax = ax.flatten()
c = ax[0].contourf(X, Y, mvar[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', extend='both')
ax[0].set_title('Global mean meridional overturning circulation (Sv)')
plt.colorbar(c, ax=ax[0])

ax[1].contourf(X, Y, mvar_atl[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', extend='both')
ax[1].set_title('Atlantic mean meridional overturning circulation (Sv)')
plt.colorbar(c, ax=ax[1])

ax[2].contourf(X, Y, mvar_pac[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', extend='both')
ax[2].set_title('Pacific mean meridional overturning circulation (Sv)')
plt.colorbar(c, ax=ax[2])


# In[92]:


[X, Y] = np.meshgrid(YC[:,0], RC)

f, ax = plt.subplots(3, 1,figsize=(15,12))
ax = ax.flatten()
c = ax[0].contourf(X, Y, pvar[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', extend='both')
ax[0].set_title('Global meridional overturning circulation by eddies(Sv)')
plt.colorbar(c, ax=ax[0])

ax[1].contourf(X, Y, pvar_atl[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', extend='both')
ax[1].set_title('Atlantic meridional overturning circulation by eddies(Sv)')
plt.colorbar(c, ax=ax[1])

ax[2].contourf(X, Y, pvar_pac[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', extend='both')
ax[2].set_title('Pacific meridional overturning circulation by eddies (Sv)')
plt.colorbar(c, ax=ax[2])


# In[94]:


[X, Y] = np.meshgrid(YC[:,0], RC)

f, ax = plt.subplots(3, 1,figsize=(15,12))
ax = ax.flatten()
c = ax[0].contourf(X, Y, (mvar+pvar)[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', 
                   extend='both')
ax[0].set_title('Global residual meridional overturning circulation (Sv)')
plt.colorbar(c, ax=ax[0])

ax[1].contourf(X, Y, (mvar_atl+pvar_atl)[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', 
               extend='both')
ax[1].set_title('Atlantic residual meridional overturning circulation (Sv)')
plt.colorbar(c, ax=ax[1])

ax[2].contourf(X, Y, (mvar_pac+pvar_pac)[:-1, :]*1e-6, np.arange(-25,26,2), cmap='bwr', 
               extend='both')
ax[2].set_title('Pacific residual meridional overturning circulation (Sv)')
plt.colorbar(c, ax=ax[2])


# ### Heat budget
# We try to close the heat budget using variables from the package ```diagnostics```.  
# In ```data.diagnostics```, we told the model to save the variables that are necessary for the heat budget computation as the following.
# 
# ```
#   fields(1:4,7) = 'TOTTTEND','ADVr_TH ','ADVx_TH ','ADVy_TH ',
#   fileName(7) = 'Diags/Tbudget1',
#   frequency(7) = 31104000.,
#   timePhase(7) = 0.,
#   
#   fields(1:5,8) = 'DFrE_TH ','DFrI_TH ','DFxE_TH ','DFyE_TH ','WTHMASS ',
#   fileName(8) = 'Diags/Tbudget2',
#   frequency(8) = 31104000.,
#   timePhase(8) = 0.,
# ```

# In[100]:


# parameters
rhoConst = 1035.0    # kg/m3
cp = 3994.0     # J/kg/K


# In[95]:


# Tendency by advection

ADVr_TH = rdmds(rdir+'Diags/Tbudget1', 3600, rec=1)  # degC m^3/s
ADVx_TH = rdmds(rdir+'Diags/Tbudget1', 3600, rec=2)
ADVy_TH = rdmds(rdir+'Diags/Tbudget1', 3600, rec=3)

advx = (ADVx_TH[0:nz-1, 0:ny-1, 0:nx-1] - ADVx_TH[0:nz-1, 0:ny-1, 1:nx])
advy = (ADVy_TH[0:nz-1, 0:ny-1, 0:nx-1] - ADVy_TH[0:nz-1, 1:ny, 0:nx-1])
advz = (ADVr_TH[1:nz, 0:ny-1, 0:nx-1] - ADVr_TH[0:nz-1, 0:ny-1, 0:nx-1])
adv = advx + advy + advz


# In[96]:


# Tendency by diffusion

DFrE_TH = rdmds(rdir+'Diags/Tbudget2', 3600, rec=0)
DFrI_TH = rdmds(rdir+'Diags/Tbudget2', 3600, rec=1)
DFxE_TH = rdmds(rdir+'Diags/Tbudget2', 3600, rec=2)
DFyE_TH = rdmds(rdir+'Diags/Tbudget2', 3600, rec=3)

difx = (DFxE_TH[0:nz-1, 0:ny-1, 0:nx-1] - DFxE_TH[0:nz-1, 0:ny-1, 1:nx])
dify = (DFyE_TH[0:nz-1, 0:ny-1, 0:nx-1] - DFyE_TH[0:nz-1, 1:ny, 0:nx-1])
dife = (DFrE_TH[1:nz, 0:ny-1, 0:nx-1] - DFrE_TH[0:nz-1, 0:ny-1, 0:nx-1])
difi = (DFrI_TH[1:nz, 0:ny-1, 0:nx-1] - DFrI_TH[0:nz-1, 0:ny-1, 0:nx-1])

dif = difx + dify + dife + difi


# In[101]:


# heat flux
TFLUX = rdmds(rdir+'Diags/sfcfrc', 3600, rec=2)    # W/m^2
Tflx_tend = TFLUX/(rhoConst * cp * DRF[0])


# In[102]:


# correction term
WTHMASS = rdmds(rdir+'Diags/Tbudget2', 3600, rec=4)    # degC m/s
Surf_corr_tend = (- WTHMASS[0,:,:]) / (DRF[0] * hFacC[0,:,:])


# In[117]:


# RHS
RHS = (adv + dif)/CellVol[:-1,:-1,:-1]    # degC / s
RHS[0,:,:] += Tflx_tend[:-1,:-1] + Surf_corr_tend[:-1,:-1]
RHS = RHS * 86400 * 360.   # degC/y

# LHS
Ttend = rdmds(rdir+'Diags/Tbudget1', 3600, rec=0)*360     # degC/day -> degC/y


# In[127]:


plt.figure(figsize=(10, 15))
ax = np.zeros(3)
showz = 0
X, Y = XC[:-1, :-1], YC[:-1, :-1]
for ip in range(3):
    if ip==0:
        var = Ttend[showz, :-1, :-1]
        titletext = 'heat budget at z='+str(RC[showz])+', LHS'
    elif ip==1:
        var = RHS[showz,...]
        titletext = 'heat budget at z='+str(RC[showz])+', RHS'
    elif ip==2:
        var = Ttend[showz, :-1, :-1] - RHS[showz,...]
        titletext = 'heat budget at z='+str(RC[showz])+', LHS - RHS'
        
    ax = plt.subplot(3, 1, int(ip+1), projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines(resolution='110m')
    c = ax.contourf(X, Y, var, np.arange(-1e-1, 1.1e-1, 1e-2), 
                      cmap='bwr', extend='both', transform=ccrs.PlateCarree())
    # ax.stock_img()
    ax.add_feature(cfeature.LAND)
    ax.set_title(titletext+', year='+str(its[-1]/360))
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    plt.colorbar(c, ax=ax, shrink=0.8)


# ### Salt budget
# We try to close the salt budget using variables from the package ```diagnostics```.  
# In ```data.diagnostics```, we told the model to save the variables that are necessary for the salt budget computation as the following.
# 
# ```
#   fields(1:4,9) = 'TOTSTEND','ADVr_SLT','ADVx_SLT','ADVy_SLT',
#   fileName(9) = 'Diags/Sbudget1',
#   frequency(9) = 31104000.,
#   timePhase(9) = 0.,
#   
#   fields(1:5,10) = 'DFrE_SLT','DFrI_SLT','DFxE_SLT','DFyE_SLT','WSLTMASS',
#   fileName(10) = 'Diags/Sbudget2',
#   frequency(10) = 31104000.,
#   timePhase(10) = 0.,
# ```

# In[128]:


# Tendency by advection
ADVr = rdmds(rdir+'Diags/Sbudget1', 3600, rec=1)    # psu m^3/s
ADVx = rdmds(rdir+'Diags/Sbudget1', 3600, rec=2)
ADVy = rdmds(rdir+'Diags/Sbudget1', 3600, rec=3)

advx_S = (ADVx[0:nz-1, 0:ny-1, 0:nx-1] - ADVx[0:nz-1, 0:ny-1, 1:nx])
advy_S = (ADVy[0:nz-1, 0:ny-1, 0:nx-1] - ADVy[0:nz-1, 1:ny, 0:nx-1])
advz_S = (ADVr[1:nz, 0:ny-1, 0:nx-1] - ADVr[0:nz-1, 0:ny-1, 0:nx-1])
adv_S = advx_S + advy_S + advz_S


# In[129]:


# Tendency by diffusion
DFrE = rdmds(rdir+'Diags/Sbudget2', 3600, rec=0)    # psu m^3/s
DFrI = rdmds(rdir+'Diags/Sbudget2', 3600, rec=1)
DFxE = rdmds(rdir+'Diags/Sbudget2', 3600, rec=2)
DFyE = rdmds(rdir+'Diags/Sbudget2', 3600, rec=3)

difx_S = (DFxE[0:nz-1, 0:ny-1, 0:nx-1] - DFxE[0:nz-1, 0:ny-1, 1:nx])
dify_S = (DFyE[0:nz-1, 0:ny-1, 0:nx-1] - DFyE[0:nz-1, 1:ny, 0:nx-1])
dife_S = (DFrE[1:nz, 0:ny-1, 0:nx-1] - DFrE[0:nz-1, 0:ny-1, 0:nx-1])
difi_S = (DFrI[1:nz, 0:ny-1, 0:nx-1] - DFrI[0:nz-1, 0:ny-1, 0:nx-1])

dif_S = difx_S + dify_S + dife_S + difi_S


# In[131]:


# salt flux
SFLUX = rdmds(rdir+'Diags/sfcfrc', 3600, rec=3)    # g/m2/s
Sflx_tend = SFLUX/(rhoConst * DRF[0])    # psu/s


# In[132]:


# correction term
WSLTMASS = rdmds(rdir+'Diags/Sbudget2', 3600, rec=4)    # psu m/s
Surf_corr_tend_S = (- WSLTMASS[0,:,:]) / (DRF[0] * hFacC[0,:,:])    # psu/s


# In[133]:


# RHS
RHS = (adv_S + dif_S)/CellVol[:-1,:-1,:-1]
RHS[0,:,:] += Sflx_tend[:-1,:-1] + Surf_corr_tend_S[:-1,:-1]    # psu/s
RHS = RHS * 86400 * 360   #psu/s -> psu/year

# LHS
Stend = rdmds(rdir+'Diags/Sbudget1', 3600, rec=0)*360.   # psu/day -> psu/year


# In[135]:


plt.figure(figsize=(10, 15))
ax = np.zeros(3)
showz = 0
X, Y = XC[:-1, :-1], YC[:-1, :-1]
for ip in range(3):
    if ip==0:
        var = Stend[showz, :-1, :-1]
        titletext = 'heat budget at z='+str(RC[showz])+', LHS'
    elif ip==1:
        var = RHS[showz,...]
        titletext = 'heat budget at z='+str(RC[showz])+', RHS'
    elif ip==2:
        var = Stend[showz, :-1, :-1] - RHS[showz,...]
        titletext = 'heat budget at z='+str(RC[showz])+', LHS - RHS'
        
    ax = plt.subplot(3, 1, int(ip+1), projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines(resolution='110m')
    c = ax.contourf(X, Y, var, np.arange(-1e-2, 1.1e-2, 1e-3), 
                      cmap='bwr', extend='both', transform=ccrs.PlateCarree())
    # ax.stock_img()
    ax.add_feature(cfeature.LAND)
    ax.set_title(titletext+', year='+str(its[-1]/360))
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    plt.colorbar(c, ax=ax, shrink=0.8)


# ### Meridional heat transport integrated over each basin 

# In[141]:


VT = ADVy_TH + DFyE_TH    # degC m3/s

MHT = np.nansum(rhoConst*cp*VT*mskS, axis=0) * 1e-15   # (J/s or W) to PW
MHT_ATL = np.nansum(rhoConst*cp*VT*mskS*atlantic_hfacs, axis=0) * 1e-15   # PW
MHT_PAC = np.nansum(rhoConst*cp*VT*mskS*pacific_hfacs, axis=0) * 1e-15   # PW
MHT_IND = np.nansum(rhoConst*cp*VT*mskS*indic_hfacs, axis=0) * 1e-15   # PW


# In[145]:


plt.plot(YC[:,0],np.nansum(MHT, axis=1), label='global')
plt.plot(YC[:,0],np.nansum(MHT_ATL, axis=1), label='Atlantic')
plt.plot(YC[:,0],np.nansum(MHT_PAC, axis=1), label='Pacific')
plt.plot(YC[:,0],np.nansum(MHT_IND, axis=1), label='Indian')
plt.legend()


# ### Age tracer

# In[147]:


trc = rdmds(rdir+'Diags/TRAC01', 3600)


# In[152]:


showz = 2
plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m')
c = ax.contourf(XC, YC, trc[showz, :, :]/86400/360, np.arange(0, 10.1, 0.5), 
                  cmap='Reds', extend='both', transform=ccrs.PlateCarree())
# ax.stock_img()
ax.add_feature(cfeature.LAND)
ax.set_title('The age of the water at level '+str(RC[showz])+', year='+str(its[-1]/360))
ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
plt.colorbar(c, ax=ax, shrink=0.8)

