import numpy as np
import scipy.io as sio

def split_C_cub(v3d, kad=1):
    """
    % [v6t] = split_C_cub(v3d,[kad])
    %---------------------------------------------
    % split 2d/3d arrays V, center, to 2d/3d x 6 faces
    % and (kad=1): add 1 column + 1 row <== at the begining !!!
    %  => output is v6t(ny+1,ny+1,[nr],6)
    % and (kad=2): add also 1 column + 1 row <== at the end !!!
    %  => output is v6t(ny+2,ny+2,[nr],6)
    %----------------------------------------------
    % Written by jmc@ocean.mit.edu, 2005.
    adopted to python by hajsong@yonsei.ac.kr, 2019
    """

    if kad not in [0, 1, 2]:
        print('kad has be either 0, 1, or 2')
        return
    
    dims = len(v3d.shape)
    if dims==3:
        nr, ny, nx = v3d.shape
    elif dims==2:
        ny, nx = v3d.shape
        nr = 1
    nyp = ny + 1
    n2p = ny + 2
    nye = ny + kad
    
    #=================================================================
    #- split on to 6 tiles with overlap in i+1 & j+1 :
    #v3d = v3d.reshape([nr, ny, nx])
    v6t = np.zeros([nr, 6, nye, nye])
    
    if kad == 0:
        v6t = np.transpose(v3d.reshape([nr, ny, 6, ny]), [0, 2, 1, 3])
    else:
        for n in range(6):
            v6t[:, n, 1:, 1:] = v3d[:, :ny, n*ny:(n+1)*ny]
    
    #- add overlap in i=1 & j=1 :
    
        v6t[:, 0, 1:, 0] = v6t[:, 4, -1, -1:0:-1]
        v6t[:, 2, 1:, 0] = v6t[:, 0, -1, -1:0:-1]
        v6t[:, 4, 1:, 0] = v6t[:, 2, -1, -1:0:-1]
        v6t[:, 1, 1:, 0] = v6t[:, 0, 1:, -1]
        v6t[:, 3, 1:, 0] = v6t[:, 2, 1:, -1]
        v6t[:, 5, 1:, 0] = v6t[:, 4, 1:, -1]
      
        v6t[:, 0, 0, :] = v6t[:, 5, -1, :]
        v6t[:, 2, 0, :] = v6t[:, 1, -1, :]
        v6t[:, 4, 0, :] = v6t[:, 3, -1, :]
        v6t[:, 1, 0, 1:] = v6t[:, 5, -1:0:-1, -1]
        v6t[:, 3, 0, 1:] = v6t[:, 1, -1:0:-1, -1]
        v6t[:, 5, 0, 1:] = v6t[:, 3, -1:0:-1, -1]
    
        v6t[:, 1, 0, 0] = v6t[:, 0, 1, -1]
        v6t[:, 3, 0, 0] = v6t[:, 2, 1, -1]
        v6t[:, 5, 0, 0] = v6t[:, 4, 1, -1]

    #- Put back to standard shape:
    v6t = np.transpose(v6t, [1, 0, 2, 3])
    if dims == 2:
        v6t = np.squeeze(v6t)

    return v6t

#=============================================
def split_UV_cub(u3d, v3d, kad=1):
    """
    % [u6t,v6t] = split_UV_cub(u3d,v3d,[ksign,kad])
    %---------------------------------------------
    % Split 2d/3d vector field u,v (on C-grid if kad > 0, on A-grid if kad < 0)
    % into 3d x 6 faces and:
    %  kad=1: (C-grid) add 1 column to U and one row to V <== at the end !!!
    %   => output is u6t(nc+1,nc,[nr],6) & v6t(nc,nc+1,[nr],6)
    % kad=2: add 1 column + 2 rows to U and 2 columns + 1 row to V
    %   => output is u6t(nc+1,nc+2,[nr],6) & v6t(nc+2,nc+1,[nr],6)
    % kad=-1: assuming input u3d & v3d are on A-grid,
    %           add 1 column to U and one row to V <== at the begining !!!
    %   => output is u6t(nc+1,nc,[nr],6) & v6t(nc,nc+1,[nr],6)
    %----------------------------------------------
    % Written by jmc@ocean.mit.edu, 2005.
    adopted to python by hajsong@yonsei.ac.kr, 2019
    """

    if kad not in [0, 1, 2]:
        print('kad has be either 0, 1, or 2')
        return

    dims = len(v3d.shape)
    if dims==3:
        nr, nc, nx = v3d.shape
    elif dims==2:
        nc, nx = v3d.shape
        nr = 1
    ncp = nc + 1
    n2p = nc + 2

    #=================================================================
    u3d = u3d.reshape(nr, nc, 6, nc)
    v3d = v3d.reshape(nr, nc, 6, nc)
    u3d = np.transpose(u3d, [2, 0, 1, 3])
    v3d = np.transpose(v3d, [2, 0, 1, 3])

    #- split on to 6 tiles with overlap in i+1 & j+1 :

    if kad == 0:
        u6t = u3d.copy()
        v6t = v3d.copy()
        ncp = nc
    else:
        #for n in range(6):
        #    v6t[:, n, 1:, 1:] = v3d[:, :ny, n*ny:(n+1)*ny]
        u6t = np.zeros([6, nr, nc, ncp])
        v6t = np.zeros([6, nr, ncp, nc])
        u6t[:, :, :, :nc] = u3d
        v6t[:, :, :nc, :] = v3d

        #%- split on to 6 faces with overlap in i+1 for u and j+1 for v :

        u6t[0, :, :nc, -1] = u3d[1, :, :nc, 0]
        u6t[1, :, :nc, -1] = v3d[3, :, ::-1, 0]
        u6t[2, :, :nc, -1] = u3d[3, :, :nc, 0]
        u6t[3, :, :nc, -1] = v3d[5, :, 0, ::-1]
        u6t[4, :, :nc, -1] = u3d[5, :, 0, :nc]
        u6t[5, :, :nc, -1] = v3d[1, :, 0, ::-1]
      
        v6t[0, :, -1, :nc] = u3d[2, :, ::-1, 0]
        v6t[1, :, -1, :nc] = v3d[2, :, 0, :nc]
        v6t[2, :, -1, :nc] = u3d[4, :, ::-1, 0]
        v6t[3, :, -1, :nc] = v3d[4, :, 0, :nc]
        v6t[4, :, -1, :nc] = u3d[0, :, ::-1, 0]
        v6t[5, :, -1, :nc] = v3d[0, :, 0, :nc]

    #%- restore the right shape:
    if dims == 2:
      u6t = np.squeeze(u6t)
      v6t = np.squeeze(v6t)
    else:
      u6t = u6t.reshape(6, nr, nc, ncp)
      v6t = v6t.reshape(6, nr, ncp, nc)
    
    return u6t, v6t


#=====================================
def calcBolusPsiCube(d, g, GMform, blkFile):
    """
    % [PsiB,ylat]=calcBolusPsiCube(d,g,GMform,blkFile);
    %
    % Compute bolus streamfunction from GM scheme
    %
    % Input arguments:
    %   The incoming field data (d) and grid data (g) must be in a structured
    %   array format (which is the format that comes from rdmnc):
    %       d       [Field data]  Kwx,Kwy
    %       g       [Grid data ]  drF,rA,dxC,dyC,dxG,dyG,HFacW,HFacS
    %       GMform  [string]      GM form 'Skew' or 'Advc'
    %       blkFile [file name]   Broken line file
    % Output arguments:
    %       PsiB : bolus streamfunction at interface level (in Sv)
    %       ylat : meridional coordinate of PsiB
    %
    % Comments:
    %   -For Skew-flux form:
    %        PsiB computed from Kwx & Kwy divided by 2.
    %        first average Kwx and Kwy at u- and v-points:
    %        psiX=(rAc*Kwx)_i / (dXc*dYg) ; psiY=(rAc*Kwy)_j / dYc ;
    %        and then "zonally" average along broken lines
    %   -For Advective form:
    %        PsiB computed from PsiX and PsiY
    %        just need to "zonally" average along broken lines
    %
    %---------------------------------------------------------------------
    adopted to python by hajsong@yonsei.ac.kr, 2019
    """
    #================
    #  Prepare grid
    #================
    nc = g.XC.shape[0]
    nr = len(g.drF)
    nt = d.GM_Kwx.shape[0]
    
    #--- areas :
    ra  = g.rAC;
    dxc = g.dxC.reshape(1, 6*nc*nc) #dxc = reshape(g.dxC(1:6*nc,1:nc),[6*nc*nc,1]);
    dyc = g.dyC.reshape(1, 6*nc*nc) #dyc = reshape(g.dyC(1:6*nc,1:nc),[6*nc*nc,1]);
    dxg = g.dxG.reshape(1, 6*nc*nc) #dxg = reshape(g.dxG(1:6*nc,1:nc),[6*nc*nc,1]);
    dyg = g.dyG.reshape(1, 6*nc*nc) #dyg = reshape(g.dyG(1:6*nc,1:nc),[6*nc*nc,1]);
    
    rAu=dxc*dyg
    rAv=dyc*dxg
    
    #--- masks :
    hw = g.HFacW.reshape(nr, 6*nc*nc) #hw = reshape(g.HFacW(1:6*nc,1:nc,1:nr),[6*nc*nc,nr]);
    hs = g.HFacS.reshape(nr, 6*nc*nc) #hs = reshape(g.HFacS(1:6*nc,1:nc,1:nr),[6*nc*nc,nr]);
    mskw = np.ceil(hw); mskw = np.minimum(1, mskw) #mskw=ceil(hw); mskw=min(1,mskw);
    msks = np.ceil(hs); msks = np.minimum(1, msks) #msks=ceil(hs); msks=min(1,msks);

    #===========================
    #  Read / prepare GM fields
    #===========================
    psiX_all = np.zeros([nt, nr, 6*nc*nc])
    psiY_all = np.zeros([nt, nr, 6*nc*nc])
    
    if GMform == 'Skew':
    
        kwx_all = 0.5*d.GM_Kwx
        kwy_all = 0.5*d.GM_Kwy
        for it in range(nt):
            kwx = kwx_all[it, :, :, :]
            kwy = kwy_all[it, :, :, :]
    
            #-- K*ra + add 1 overlap :
            kwx = np.tile(ra, [nr, 1, 1]) * kwx    # kwx = repmat(ra,[1 1 nr]).*kwx;
            kwy = np.tile(ra, [nr, 1, 1]) * kwy    # kwy = repmat(ra,[1 1 nr]).*kwy;
            v6X = split_C_cub(kwx, 1)
            v6Y = split_C_cub(kwy, 1)
            k6x = v6X[:, :, 1:, :]
            k6y = v6Y[:, :, :, 1:]
    
            #-----------------
            v6X = None; v6Y = None
            v6X = 0.5 * (k6x[:, :, :, 1:] + k6x[:, :, :, :-1])
            v6Y = 0.5 * (k6y[:, :, 1:, :] + k6y[:, :, :-1, :])
    
            psiX = np.zeros([nr, nc, 6*nc])
            psiY = np.zeros([nr, nc, 6*nc])
    
            for n in range(6):
                psiX[:, :, n*nc:(n+1)*nc] = v6X[n, :, :, :]
                psiY[:, :, n*nc:(n+1)*nc] = v6Y[n, :, :, :]
    
            psiX = psiX.reshape(nr, 6*nc*nc)
            psiY = psiY.reshape(nr, 6*nc*nc)
    
            psiX_all[it, :, :] = mskw * psiX / np.tile(rAu, [nr, 1])
            psiY_all[it, :, :] = msks * psiY / np.tile(rAv, [nr, 1])

    elif GMform == 'Advc':
    
        psiX_all = d.GM_PsiX[:, :, :, :6*nc].reshape(nt, nr, 6*nc*nc)
        psiY_all = d.GM_PsiY[:, :, :nc, :].reshape(nt, nr, 6*nc*nc)
    
    #=======================================
    # Zonally integrate along broken lines
    #=======================================

    dmat = sio.loadmat(blkFile)
    bkl_Ylat = np.squeeze(dmat['bkl_Ylat'])
    bkl_Flg = dmat['bkl_Flg'].T
    bkl_IJuv = dmat['bkl_IJuv'].T
    bkl_Npts = np.squeeze(dmat['bkl_Npts'])
    ydim = len(bkl_Ylat)
    ylat = np.append(np.append(-90, bkl_Ylat), 90)
    ufac = np.remainder(bkl_Flg, 2)
    vfac = np.fix(bkl_Flg/2)
    
    PsiB = np.zeros([nt, nr+1, ydim+2])
    
    for it in range(nt):
        for k in range(nr):
            psixt = dyg * psiX_all[it, k, :]; psixt = np.squeeze(psixt)
            psiyt = dxg * psiY_all[it, k, :]; psiyt = np.squeeze(psiyt)
            for jl in range(ydim):
                ie = bkl_Npts[jl]
                PsiB[it, k, jl+1] = np.sum(ufac[jl, :ie] * psixt[bkl_IJuv[jl, :ie]]
                                   + vfac[jl, :ie] * psiyt[bkl_IJuv[jl, :ie]])

    return PsiB


#================================================
def rotate_uv2uvEN(u, v, AngleCS, AngleSN, Grid='C', maskW=None, maskS=None):
    """
    [uE,vN,msk] = rotate_uv2uvEN(u,v,AngleCS,AngleSN,[Grid,maskW,maskS])
    
    Rotate cube sphere U and V vector components to east-west (uE) and
    north-south (vN) components located on cube sphere grid centers.
    
    Incoming u and v matricies are assumed to be cube sphere A-grid or C-grid
    vector fields (defaut is C-grid) where the first two dimensions are (6*nc
    nc), where nc is the cube face resolution.  There may up to 4 additional
    dimensions (likely z and time, trials, etc.) beyond this.
    Optional maskW & maskS can be provided (for C-grid vector input case)
    and used for cell-center interpolation
    
    e.g.
    
    >> uC=rdmds('uVeltave.0000513360');
    >> vC=rdmds('vVeltave.0000513360');
    >> AngleCS=rdmds('AngleCS');
    >> AngleSN=rdmds('AngleSN');
    >> [uE,vN] = rotate_uv2uvEN(uC,vC,AngleCS,AngleSN);
   
    >> uA=rdmds('uVeltaveA.0000513360');
    >> vA=rdmds('vVeltaveA.0000513360');
    >> AngleCS=rdmds('AngleCS');
    >> AngleSN=rdmds('AngleSN');
    >> [uE,vN] = rotate_uv2uvEN(uA,vA,AngleCS,AngleSN,'A');

    adopted to python by hajsong@yonsei.ac.kr, 2019
    """
    if maskW is None:
        UVmsk = 0
    else:
        UVmsk = 1
 
    # get dimension
    ndim = len(u.shape)
    nc = u.shape[-2]
    if ndim==2:
        nz = 1
        if UVmsk==1:
            n3d = 1
            maskW = maskW[0, :, :]
            maskS = maskS[0, :, :]
    elif ndim==3:
        nz = int(np.prod(u.shape)/np.prod(u.shape[-2:]))
        if UVmsk==1:
            n3d = u.shape[0]

    if UVmsk==1:
        dim3d = [n3d, u.shape[-2], u.shape[-1]]

        # preprocessing the data
        if n3d == nz:
            nt = 1
            u = u * maskW
            v = v * maskS
        else:
            nt = u.shape[0]
            u = u.reshape(nt, 6*nc*nc*n3d)
            v = v.reshape(nt, 6*nc*nc*n3d)
            maskW = maskW.reshape(1, 6*nc*nc*n3d)
            maskS = maskS.reshape(1, 6*nc*nc*n3d)
            u = u * np.tile(maskW, [nt, 1])
            v = v * np.tile(maskS, [nt, 1])
    else:
        dim3d = u.shape

    u = u.reshape(nz, nc, 6*nc)
    v = v.reshape(nz, nc, 6*nc)

    # Do simple average to put u,v at the cell center (A-grid) as needed.
    [uu,vv] = split_UV_cub(u,v,1)
    uu = uu.reshape(6, nz, nc, nc+1)
    vv = vv.reshape(6, nz, nc+1, nc)

    if UVmsk == 1:
        maskW = maskW.reshape(n3d, nc, 6*nc)
        maskS = maskS.reshape(n3d, nc, 6*nc)
        [mu,mv] = split_UV_cub(maskW,maskS,0);
        mu = mu.reshape(6, n3d, nc, nc+1)
        mv = mu.reshape(6, n3d, nc+1, nc)
        um = 0.5*(mu[:, :, :, :-1] + mu[:, :, :, 1:])
        vm = 0.5*(mv[:, :, :-1, :] + mv[:, :, 1:, :])
        msk = 0.5*(um + vm)
        msk = msk.reshape(dim3d)
        #-----
        u = 0.5*(uu[:, :, :, :-1] + uu[:, :, :, 1:])
        v = 0.5*(vv[:, :, :-1, :] + vv[:, :, 1:, :])
        u = np.transpose(u, [1, 2, 0, 3]).reshape(nt, 6*nc*nc*n3d)
        v = np.transpose(v, [1, 2, 0, 3]).reshape(nt, 6*nc*nc*n3d)
        um = np.transpose(um, [1, 2, 0, 3]).reshape(1, 6*nc*nc*n3d)
        vm = np.transpose(vm, [1, 2, 0, 3]).reshape(1, 6*nc*nc*n3d)
        um[np.where(um==0)] = 1
        vm[np.where(vm==0)] = 1
        um = 1/um
        vm = 1/vm
        u = u * np.tile(um, [nt, 1])
        v = v * np.tile(vm, [nt, 1])
        u = u.reshape(nz, 6*nc*nc)
        v = v.reshape(nz, 6*nc*nc)
    else:
        msk = np.ones(dim3d)
        u = 0.5*(uu[:, :, :, :-1] + uu[:, :, :, 1:])
        v = 0.5*(vv[:, :, :-1, :] + vv[:, :, 1:, :])
        u = np.transpose(u, [1, 2, 0, 3]).reshape(nz, 6*nc*nc)
        v = np.transpose(v, [1, 2, 0, 3]).reshape(nz, 6*nc*nc)

    # Make rotation to find uE, vN.
    uE = np.nan * np.zeros([nz, 6*nc*nc])
    vN = np.nan * np.zeros([nz, 6*nc*nc])
    for k in range(nz):
        uE[k, :] = AngleCS.flatten() * u[k, :]\
                   - AngleSN.flatten() * v[k, :]
        vN[k, :] = AngleSN.flatten() * u[k, :]\
                   + AngleCS.flatten() * v[k, :]

    uE = uE.reshape(dim3d)
    vN = vN.reshape(dim3d)
 
    return uE, vN


def cube2latlon_preprocess(x, y, xi, yi):
    """
    % del=cube2latlon_preprocess(x,y,xi,yi);
    %
    % Calculates the data required for fast interpolation from the cube to a
    % latitude-longitude grid
    %  x,y   are 2-D arrays of the cell-centered coordinates
    %  xi,yi are vectors of the new regular lat-lon grid to interpolate to.
    %  del   is the transformation data used by cube2latlon_fast
    %
    % e.g.
    % >> x=rdmds('XC');
    % >> y=rdmds('YC');
    % >> t=rdmds('Ttave.0000513360');
    % >> xi=-179:2:180;yi=-89:2:90;
    % >> del=cube2latlon_preprocess(x,y,xi,yi);
    % >> ti=cube2latlon_fast(del,t);
    %
    % Written by adcroft@.mit.edu, 2004.
    """


def cubeZ2latlon(x, y, c, xi, yi):
    """
    % z=cubeZ2latlon(x,y,c,xi,yi);
    %
    % Re-grids model output on expanded spherical cube to lat-lon grid.
    %  x,y   are 1-D arrays (reshaped Horiz. grid) of the cell-corner coordinates
    %  c     is a 1-D or 2-D scalar field
    %  xi,yi are vectors of the new regular lat-lon grid to interpolate to.
    %  z     is the interpolated data with dimensions of size(xi) by size(yi).
    %
    % e.g.
    % >> xg=rdmds('XG'); nPts=prod(size(xg)); x=rehsape(xg,nPts,1);
    % >> yg=rdmds('YG'); y=rehsape(yg,nPts,1);
    % >> PsiH=rdmds('psiH.0000513360');
    % >> xi=-179:2:180;yi=-89:2:90;
    % >> psiI=cubeZ2latlon(x,y,psiH,xi,yi);
    % can also add the 2 missing corner :
    % >> nc=sqrt(nPts/6);
    % >> x(end+1)=mean(xg([1:2*nc:6*nc],nc)); y(end+1)=mean(yg([1:2*nc:6*nc],nc));
    % >> x(end+1)=mean(xg([2*nc:2*nc:6*nc],1)); y(end+1)=mean(yg([2*nc:2*nc:6*nc],1));
    %
    % Written by jmc@ocean.mit.edu, 2005.
    adopted to python by hajsong@yonsei.ac.kr, 2019
    """
    from scipy.interpolate import griddata

    XX, YY = np.meshgrid(xi, yi)
    NN = c.shape
    if len(c.shape)==1:
        nz = 1
        nPt2 = len(c)
        c = c.reshape(nz, nPt2)
    elif len(c.shape)==2:
        nz, nPt2 = c.shape
    nc = int(np.fix(np.sqrt(nPt2/6)))
    nPts = 6*nc*nc
    
    z = np.zeros([nz, len(yi), len(xi)])
    for k in range(nz):
        X = np.reshape(x, [nc, 6*nc])
        Y = np.reshape(y, [nc, 6*nc])
        C = np.reshape(c[k, :nPts], [nc, 6*nc])

    
        i = 3*nc + np.arange(nc)
        j = int(np.floor(nc/2))
        X = np.append(X, (X[j, i]-360).reshape(nc, 1), axis=1)
        Y = np.append(Y, Y[j, i].reshape(nc, 1), axis=1) 
        C = np.append(C, C[j, i].reshape(nc, 1), axis=1) 
    
        i = 5*nc + int(np.floor(nc/2))
        j = np.arange(int(np.floor(nc/2)))
        X = np.append(X, np.zeros([nc, 1]), axis=1)
        Y = np.append(Y, np.zeros([nc, 1]), axis=1)
        C = np.append(C, np.zeros([nc, 1]), axis=1)
        X[j, -1] = X[j, i]-360
        Y[j, -1] = Y[j, i]
        C[j, -1] = C[j, i]
    
        #--
        j = int(np.floor(nc/2))
        i = 2*nc + j
        if Y[j, i]==90:
            X[j, i] = 180
        i = 2*nc + np.arange(int(np.floor(nc/2)), nc)
        j = int(np.floor(nc/2))
        X[i-2*nc, -1] = X[j, i] - 360
        Y[i-2*nc, -1] = Y[j, i]
        C[i-2*nc, -1] = C[j, i]
    
        j = int(np.floor(nc/2))
        i = 5*nc + j
        ij = i + j*nc*6
        if Y[j, i]==-90:
        #% fprintf('South pole: %i %i %f %f\n',i,j,X(i,j),Y(i,j));
            X[j, i] = 180
    
    
        X = X.reshape(1, np.prod(X.shape))
        Y = Y.reshape(1, np.prod(Y.shape))
        C = C.reshape(1, np.prod(C.shape))
    
        I = np.nonzero(Y==-90)[0]
    
        if len(I)==1:
            #% fprintf('South pole: %i %f %f\n',I,X(I),Y(I));
            X = np.append(X, X[I] - 360)
            Y = np.append(Y, Y[I])
            C = np.append(C, C[I])
    
        if nPt2 > nPts:
            X = np.append(X, x[nPts+1])
            Y = np.append(Y, y[nPts+1])
            C = np.append(C, c[k, nPts+1])

        if nPt2 == nPts+2:
            X = np.append(X, x[nPt2])
            Y = np.append(Y, y[nPt2])
            C = np.append(C, c[k, nPt2])
        
        point = np.zeros([X.shape[1], 2])
        point[:, 0] = X[0, :].T
        point[:, 1] = Y[0, :].T
        z[k, :, :] = griddata(point, np.squeeze(C), (XX, YY))
    
    z = np.squeeze(z)

    return z
