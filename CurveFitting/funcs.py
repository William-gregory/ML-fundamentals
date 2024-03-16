import numpy as np
import scipy 
from scipy.interpolate import _fitpack

#The following code is a spline interpolation function taken directly from scipy.interpolate.spline.
#I think this may only be present in older versions of scipy so I just copied this to import it in my 
#own code when needed.

def spline(xk, yk, xnew, order=3, kind='smoothest', conds=None):
    
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
