import numpy as np

def sourceTerm(pt):
    return 0

def exactSol(pt):
    pi = np.pi
    x = pt[:,0]
    y = pt[:,1]
    sx = np.sin(pi*x)
    cx = np.cos(pi*x)
    chx = np.cosh(pi*x)
    chy = np.cosh(pi*y)
    shx = np.sinh(pi*x)
    shy = np.sinh(pi*y)
    cth = np.cosh(pi)/np.sinh(pi)
    csch = 1/np.sinh(pi)
    u = sx*(chy-cth*shy)
    du_x = pi*cx*(chy-cth*shy)
    du_y = pi*sx*(shy-cth*chy)
    return u, du_x, du_y