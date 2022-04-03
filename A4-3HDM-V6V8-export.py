'''
computes minima via straightforward 12D minimization
checks V6 + V8 terms
computes gradient and masses via numdifftools
'''
from math import sqrt
import numpy as np
from scipy import optimize
import numdifftools as nd
import scipy.linalg as la


def KK(x):
    phi1 = np.array([x[0]+1j*x[1], x[2]+1j*x[3]])
    phi2 = np.array([x[4]+1j*x[5], x[6]+1j*x[7]])
    phi3 = np.array([x[8]+1j*x[9], x[10]+1j*x[11]])
    phi = np.array([phi1,phi2,phi3])
    phiConj = np.conj(phi.T)
    res = np.conj(phi @ phiConj)
    return res

def x_to_phi(x):
    phi1 = np.array([x[0]+1j*x[1], x[2]+1j*x[3]])
    phi2 = np.array([x[4]+1j*x[5], x[6]+1j*x[7]])
    phi3 = np.array([x[8]+1j*x[9], x[10]+1j*x[11]])
    phi = np.array([phi1,phi2,phi3])
    return phi
    
def VA4(x,coefs,coefs6,coefs8):   
    m0, Lam0, Lam1, Lam2, Lam3, Lam4 = coefs
    zeta1,zeta2,zeta3,zeta4 = coefs6
    eta1,eta2 = coefs8

    K = KK(x)
    r0 = np.trace(K)/sqrt(3)
    r3 = (K[0,0]-K[1,1])/2
    r8 = (K[0,0]+K[1,1] - 2*K[2,2])/sqrt(12)
    r1, r2 = K[0,1].real, K[0,1].imag
    r4, r5 = K[2,0].real, K[2,0].imag
    r6, r7 = K[1,2].real, K[1,2].imag
    V2 = -m0*r0
    V4 = Lam0*r0*r0 + Lam1*(r1*r1+r4*r4+r6*r6) + Lam2*(r2*r2+r5*r5+r7*r7) \
        + Lam3*(r3*r3+r8*r8) + Lam4*(r1*r2+r4*r5+r6*r7)
    V6 = zeta1*(r1*r4*r6) + zeta2*(r2*r5*r7) + zeta3*(r1*r5*r7+r2*r4*r7+r2*r5*r6) \
        + zeta4*(r1*r4*r7+r2*r4*r6+r1*r5*r6)   
    V8 = eta1*(r1**4 + r4**4 + r6**4) + eta2*(r2**4 + r5**4 + r7**4)  
    return (V2+V4+V6+V8).real

def xyzt(K):
    r0 = np.trace(K)/sqrt(3)
    r3 = (K[0,0]-K[1,1])/2
    r8 = (K[0,0]+K[1,1] - 2*K[2,2])/sqrt(12)
    r1, r2 = K[0,1].real, K[0,1].imag
    r4, r5 = K[2,0].real, K[2,0].imag
    r6, r7 = K[1,2].real, K[1,2].imag
    x = (r1*r1+r4*r4+r6*r6)/r0**2
    y = (r2*r2+r5*r5+r7*r7)/r0**2
    t = (r1*r2+r4*r5+r6*r7)/r0**2
    z = (r3*r3+r8*r8)/r0**2
    return r0.real, x.real, y.real, z.real, t.real

m0, Lam0, Lam1, Lam2, Lam3, Lam4 = 5.0, 2.0, -1.2, -1.0, 0.1, 0.0
coefs = [m0, Lam0, Lam1, Lam2, Lam3, Lam4]
coefs6 = [-1.0, 0.0, 0.0, 0.0] 
coefs8 = [0.5, 0.5]
    
for k in range(1):
    x0 = np.random.random(size = 12)
    
    VVA4 = lambda x: VA4(x,coefs,coefs6,coefs8)
    res = optimize.minimize(VVA4, x0)
    xmin = res.x
    resKK = KK(res.x)
    r0, xx, yy, zz, tt = xyzt(resKK)
    v1_2 = (xmin[0])**2 + (xmin[1])**2 + (xmin[2])**2 + (xmin[3])**2 
    v2_2 = (xmin[4])**2 + (xmin[5])**2 + (xmin[6])**2 + (xmin[7])**2 
    v3_2 = (xmin[8])**2 + (xmin[9])**2 + (xmin[10])**2 + (xmin[11])**2 
    v2 = v1_2 + v2_2 + v3_2
    np.set_printoptions(precision=4,suppress=True)
 # ---- numdifftools ---
    df = nd.Gradient(VVA4)
    ddf = nd.Hessian(VVA4)
    resdf = df(xmin)
    resddf = ddf(xmin)
    print('An extremum? (up to 1e-4)', np.allclose(resdf, np.zeros(12),atol=1e-04))
    print(resddf)
    eigvalues, eigvectors = la.eig(resddf)
    masses = eigvalues.real
    NGBosons = np.sort(masses)[:3]
    print('3 NG Bosons (up to 1e-4)?',\
          np.allclose(NGBosons, np.zeros(3), atol=1e-04))
    print(np.sort(masses)[3:])
    print(x0)
    print(res.x)
    
    print('x, y, z, t = {:.3f}, {:.3f}, {:.3f}, {:.3f},  V = {:.5f}'.\
          format(xx, yy, zz, tt, res.fun))
    print('|phi_1|^2, |phi_2|^2, |phi_3|^2  = {:.3f}, {:.3f}, {:.3f}'.\
         format(v1_2, v2_2, v3_2))
    
