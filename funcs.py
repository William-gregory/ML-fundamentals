import numpy as np
import math
import matplotlib as mpl
from matplotlib.patches import Wedge


def lat2area(lats=None,lons=None,dt_lat=None,dt_lon=None):
    r1 = 6378.137 #equatorial radius (km)
    r2 = 6356.752 #polar radius (km)
    
    if (lats is None) & (dt_lat is None):
        dt_lat = 1
        lats = np.arange(90,-90,-dt_lat)
    elif (lats is None) & (dt_lat is not None):
        lats = np.arange(90,-90,-dt_lat)
    elif (lats is not None) & (dt_lat is None):
        dt_lat = abs(lats[0] - lats[1])
        
    if (lons is None) & (dt_lon is None):
        dt_lon = 1
    elif (lons is not None) & (dt_lon is None):
        dt_lon = abs(lons[0] - lons[1])
    
    Area = np.zeros(len(lats))
    for i in range(len(lats)-1):
        A1 = (lats[i]*math.pi)/180
        A2 = (lats[i+1]*math.pi)/180
        R = np.sqrt(((r1**2 * math.cos(A1))**2 + (r2**2 * math.sin(A1))**2)/((r1 * math.cos(A1))**2 + (r2 * math.sin(A1))**2))
        if abs(lats[i]) != 90:
            B = abs(math.sin(A1) - math.sin(A2))
            Area[i] = ((math.pi*R**2)/180)*B*dt_lon

    return Area


def draw_meridians(m, ax, hemisphere, width_percent=0.03, degree=30, meridians=False, fontsize=20):
    centre_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    centre_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    width = abs(centre_x) * width_percent

    inner_radius = (abs(centre_x) - width/2) + 4e4
    outer_radius = inner_radius + width

    angle_breaks = list(range(0, 361, degree))

    for i, (from_angle, to_angle) in enumerate(list(zip(angle_breaks[:-1], angle_breaks[1:]))):
        color='white' if i%2 == 0 else 'black'
        wedge = Wedge((centre_x, centre_y), outer_radius, from_angle, to_angle, width=outer_radius - inner_radius, 
                      facecolor=color,
                      edgecolor='black',
                      clip_on=False,
                      ls='solid',
                      lw=1)
        ax.add_patch(wedge)
        
 
    merid_values = np.arange(0.,360.,degree)
    if hemisphere=='north':
        rotation1 = np.arange(0,90+degree,degree)
        rotation2 = np.arange(-60,0+degree,degree)
        rotation3 = np.arange(30,90+degree,degree)
        rotation4 = np.arange(-60,0+degree,degree)
    elif hemisphere=='south':
        rotation1 = np.arange(0,-90-degree,-degree)
        rotation2 = np.arange(60,0-degree,-degree)
        rotation3 = np.arange(-30,-90-degree,-degree)
        rotation4 = np.arange(60,0-degree,-degree)
    if meridians is True: 
        meridians = m.drawmeridians(merid_values,labels=[1,1,1,1],linewidth=1,fontsize=fontsize, fmt=(lambda x: (u"%d\N{DEGREE SIGN}") % (x)))
    else:
        meridians = m.drawmeridians(merid_values,labels=[1,1,1,1],linewidth=0,fontsize=fontsize, fmt=(lambda x: (u"%d\N{DEGREE SIGN}") % (x)))
    k = -1 ; l = -1 ; p = -1 ; r = -1
    for m in meridians:
        if m <= 90:
            k = k + 1
            try:
                meridians[m][1][0].set_rotation(rotation1[k])
            except:
                pass
        elif (m<=180) & (m>90):
            l = l + 1
            try:
                meridians[m][1][0].set_rotation(rotation2[l])
            except:
                pass
        elif (m>180) & (m<=270):
            p = p + 1
            try:
                meridians[m][1][0].set_rotation(rotation3[p])
            except:
                pass
        else:
            r = r + 1
            try:
                meridians[m][1][0].set_rotation(rotation4[r])
            except:
                pass
    if hemisphere == 'north':
        pairs = [[1,2],[0.99,1.02],[1.01,1.07],[1.02,1],[0.99,0.97],[0.99,0.98],[1,1.01],[1.1,0.995],\
                 [1.1,0.97],[1.99,1],[0.98,1.02],[1.06,0.99]]
    elif hemisphere == 'south':
        pairs = [[1,1],[0.99,0.99],[1,0.98],[1.005,1],[1,1.07],[0.98,1.02],[1,1.8],[1.1,0.995],[1.1,1.03],\
                 [1.99,1],[1.1,0.96],[1.06,0.99]]
    index = -1
    for key, (lines,texts) in meridians.items():
        index += 1
        if key == 0:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 30:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 60:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 90:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 120:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 150:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 180:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 210:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 240:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 270:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 300:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
        if key == 330:
            for text in texts:
                x,y = text.get_position()
                text.set_position((x*pairs[index][0],y*pairs[index][1]))
                text.set_position((x*pairs[index][0],y*pairs[index][1]))


    
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.    
    cmap: colormap instance, eg. cm.jet.  
    N: number of colors.    
    Example
    x = resize(arange(100), (5,100))
    djet = cmap_discretize(cm.jet, 5)
    imshow(x, cmap=djet)
    """
    if type(cmap) == str:
        cmap = mpl.cm.get_cmap(cmap)
    #colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    # Fred's update ... dont' start with the colormap edges !
    colors_i = np.concatenate((np.linspace(1./N*0.5, 1-(1./N*0.5), N), (0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1) ]
    # Return colormap object.
    return mpl.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)
    
def spline(xk, yk, xnew, order=3, kind='smoothest', conds=None):
    from scipy.interpolate import _fitpack
    import scipy 

    def _dot0(a, b):
        if b.ndim <= 2:
            return np.dot(a, b)
        else:
            axes = list(range(b.ndim))
            axes.insert(-1, 0)
            axes.pop(0)
            return np.dot(a, b.transpose(axes))

    def _find_smoothest(xk, yk, order, conds=None, B=None):
        N = len(xk)-1
        K = order
        if B is None:
            B = _fitpack._bsplmat(order, xk)
        J = _fitpack._bspldismat(order, xk)
        u, s, vh = scipy.linalg.svd(B)
        ind = K-1
        V2 = vh[-ind:,:].T
        V1 = vh[:-ind,:].T
        A = np.dot(J.T,J)
        tmp = np.dot(V2.T,A)
        Q = np.dot(tmp,V2)
        p = scipy.linalg.solve(Q, tmp)
        tmp = np.dot(V2,p)
        tmp = np.eye(N+K) - tmp
        tmp = np.dot(tmp,V1)
        tmp = np.dot(tmp,np.diag(1.0/s))
        tmp = np.dot(tmp,u.T)
        return _dot0(tmp, yk)

    def splmake(xk, yk, order=3, kind='smoothest', conds=None):
        B = _fitpack._bsplmat(order, xk)
        coefs = _find_smoothest(xk, yk, order, conds, B)
        return xk, coefs, order

    def spleval(xck, xnew, deriv=0):
        (xj,cvals,k) = xck
        oldshape = np.shape(xnew)
        xx = np.ravel(xnew)
        sh = cvals.shape[1:]
        res = np.empty(xx.shape + sh, dtype=cvals.dtype)
        for index in np.ndindex(*sh):
            sl = (slice(None),)+index
            if issubclass(cvals.dtype.type, np.complexfloating):
                res[sl].real = _fitpack._bspleval(xx,xj,cvals.real[sl],k,deriv)
                res[sl].imag = _fitpack._bspleval(xx,xj,cvals.imag[sl],k,deriv)
            else:
                res[sl] = _fitpack._bspleval(xx,xj,cvals[sl],k,deriv)
        res.shape = oldshape + sh
        return res

    return spleval(splmake(xk,yk,order=order,kind=kind,conds=conds),xnew)