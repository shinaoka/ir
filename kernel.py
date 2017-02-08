import numpy as np
from scipy.special import legendre
#import matplotlib.pylab as plt
#import sys

def safe_exp(x):
    return (0.0 if x < -60.0 else np.exp(x))

def compute_kernel_F(x, w):
    r = np.zeros_like(w)
    for i in range(len(r)):
        r[i] = compute_kernel_scalar_F(x, w[i])
    return r

def compute_kernel_scalar_F(x, w):
    if w > 100.0:
        return safe_exp(-0.5*x*w - 0.5*w)
    elif w < -100.0:
        return safe_exp(-0.5*x*w + 0.5*w)
    else:
        return safe_exp(-0.5*x*w)/(2*np.cosh(0.5*w))

def compute_kernel_B(x, w):
    r = np.zeros_like(w)
    for i in range(len(r)):
        r[i] = compute_kernel_scalar_B(x, w[i])
    return r

def compute_kernel_scalar_B(x, w):
    if np.abs(w) < 1e-10:
        return safe_exp(-0.5*x*w)
    elif w > 100.0:
        return w*safe_exp(-0.5*x*w - 0.5*w)
    elif w < -100.0:
        return -w*safe_exp(-0.5*x*w + 0.5*w)
    else:
        return w*safe_exp(-0.5*x*w)/(2*np.sinh(0.5*w))

def compute_kernel_BR(x, w):
    r = np.zeros_like(w)
    for i in range(len(r)):
        r[i] = compute_kernel_scalar_BR(x, w[i])
    return r

def compute_kernel_scalar_BR(x, w):
    if np.abs(w) < 1e-10:
        return safe_exp(-0.5*x*w)/w
    elif w > 100.0:
        return safe_exp(-0.5*x*w - 0.5*w)
    elif w < -100.0:
        return -safe_exp(-0.5*x*w + 0.5*w)
    else:
        return safe_exp(-0.5*x*w)/(2*np.sinh(0.5*w))

class KernelBasisDE:
    def __init__(self, Nw, Ntau, wmax, cutoff, kernel='Fermionic'):
        self.tmin = 3.0
        self.twmin = -4 
        Nx = Ntau
        
        twmax = np.arcsinh(np.log(wmax)*2/np.pi)
        tw_vec = np.linspace(self.twmin, twmax, Nw)
    
        w_vec = np.exp(0.5*np.pi*np.sinh(tw_vec))
        w_vec = np.r_[-w_vec[::-1], w_vec]
    
        weight_w = np.sqrt(0.5*np.pi*np.cosh(tw_vec)*np.exp(0.5*np.pi*np.sinh(tw_vec)))
        weight_w = np.r_[weight_w[::-1], weight_w]

        #DE mesh for x
        tvec = np.linspace(-self.tmin, self.tmin, Nx) #3 is a very safe option.
        xvec = np.tanh(0.5*np.pi*np.sinh(tvec))
        weight_x = np.sqrt(0.5*np.pi*np.cosh(tvec))/np.cosh(0.5*np.pi*np.sinh(tvec)) #sqrt of the weight of DE formula

        K = np.zeros((Nx, 2*Nw), dtype=float)
        if kernel=='Fermionic':
            for i in range(Nx):
                K[i,:] = weight_x[i] * compute_kernel_F(xvec[i], w_vec[:]) * weight_w[:]
        elif kernel=='Bosonic':
            for i in range(Nx):
                K[i,:] = weight_x[i] * compute_kernel_B(xvec[i], w_vec[:]) * weight_w[:]
        elif kernel=='BosonicRaw':
            for i in range(Nx):
                K[i,:] = weight_x[i] * compute_kernel_BR(xvec[i], w_vec[:]) * weight_w[:]
        else:
            raise RuntimeError("Unknown kernel type "+kernel)
        
        U, s, V = np.linalg.svd(K)
        
        self.dim = s.shape[0]
        for il in range(s.shape[0]):
            if np.abs(s[il]/s[0]) < cutoff:
                self.dim = il
                break
        
        #Rescale U and V
        for i in range(Nx):
            U[i,:] /= weight_x[i]
        for il in range(U.shape[1]):
            norm2 = (2*self.tmin/Nx)*np.sum((U[:,il]**2)*(weight_x**2))
            U[:,il] /= np.sqrt(norm2)
            if U[-1,il] < 0.0:
                U[:,il] *= -1
    
        for il in range(V.shape[0]):
            V[il,:] /= weight_w

        self.s = s[0:self.dim]
        self.xvec = xvec
        self.tvec = tvec
        self.norm_weight_x = weight_x**2
        self.U = U[:,0:self.dim]
        self.w_vec = w_vec
        self.norm_weight_w = weight_w**2
        self.Vt = V.transpose()[:,0:self.dim]
        
    def singular_values(self):
        return self.s
    
    def value(self,x):
        val = np.zeros((self.dim,), dtype=float)
        if x > -1.0 and x < 1.0:
            t = np.arcsinh((2.0/np.pi) * np.arctanh(x))
        elif x < -1.0:
            t = - self.tmin
        else:
            t = self.tmin
        dt = (2 * self.tmin)/(self.U.shape[0] - 1);    
        idx = int((t + self.tmin)/dt)            
        if idx == self.U.shape[0] - 1:
            val[:] = self.U[idx, :];
        else:
            fraction = (t - self.tvec[idx]) / dt;
            val[:] = (1.0 - fraction) * self.U[idx, :] + fraction * self.U[idx + 1, :];
        return val
    
    def x_points(self):
        return self.xvec
    
    def x_norm_weights(self):
        return self.norm_weight_x
    
    def x_basis(self):
        return self.U
    
    def omega_points(self):
        return self.w_vec
    
    def omega_basis(self):
        return self.Vt

    def omega_norm_weights(self):
        return self.norm_weight_w

    def compute_Tnl(self, n_iw):
        nx = len(self.xvec)
        exp = np.zeros((nx, n_iw), dtype=complex)
        for iw in range(n_iw):
            exp[:, iw] = np.exp(1J*(iw+0.5)*(self.xvec+1.0))

        return np.dot(exp.transpose(), np.dot(np.diag(self.norm_weight_x), self.U))/np.sqrt(2.0)
