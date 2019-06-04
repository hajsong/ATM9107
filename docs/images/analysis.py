import numpy as np
import matplotlib.pyplot as plt
import MITgcmutils as mit
import MITgcmutils.cs as cs
import cs_tools as cst

# 1. interpolation onto the regular lat-lon grid
eta = mit.rdmds('../run/Eta', 75600)
xi = np.arange(-179, 181, 2)
yi = np.arange(-89, 90, 2)
x = mit.rdmds('../run/XC')
y = mit.rdmds('../run/YC')

z = cst.cubeZ2latlon(x.flatten(), y.flatten(), eta.flatten(), xi, yi)

# 2. adjust U and V so that they represent the current in E-W and S-N
AngleSN = mit.rdmds('../run/AngleSN')
AngleCS = mit.rdmds('../run/AngleCS')
u = mit.rdmds('../run/U', 75600)
v = mit.rdmds('../run/V', 75600)

xg = mit.rdmds('../run/XG')
yg = mit.rdmds('../run/YG')

uE, vN = cst.rotate_uv2uvEN(u, v, AngleCS, AngleSN)

plt.figure(1)
cs.pcol(xg, yg, u[0,...])

plt.figure(2)
cs.pcol(xg, yg, uE[0,...])

plt.show()
