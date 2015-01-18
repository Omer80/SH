"""
Finite Differences Method integrator for the Forced Swift Hohenberg equation.
SH Equation:
     u_t = epsilon*u + lambda*u^2 - u^3 - [ d^2/(dx)^2 + d^2/(dy)^2 + k0^2 ]^2 * u + gamma*u*cos(kf*x)
Periodic boundary conditions are assumed.
"""
__version__=1.0
__author__ = """Omer Tzuk (omertz@post.bgu.ac.il)"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import pylab

def local(u, dt, forcing, par):
    """This is the nonlinear function
       \lambda u^2 - u^3 + u * forcing   
    """
    return u + dt * (par['Lambda']*u**2 - u**3 + u*forcing); # Local terms

def laplacian(u):
    "This is the laplacian term"
    hh = par['dx']**2
    return (np.roll(u,1,axis=0) + np.roll(u,-1,axis=0) + np.roll(u,1,axis=1) + np.roll(u,-1,axis=1) -4*u)/hh

def spatial(u):
	"""
	This is the spatial term of SH
	- [ d^2/(dx)^2 + d^2/(dy)^2 + k0^2 ]^2 * u
	"""
	return dt*(laplacian(u)-(par['k0'])**2)


par = {'epsilon':0.1,
       'k0':1.0,
       'Lambda':0.2,
       'gamma':0.,
       'kf':1.2,
       'n':(128,128),
       'l':(60,60),
       'resonance':2.0,
       'periods':8,
       }
       
# initial condition
u0 = 0.5*(np.random.random((par['n'][0],par['n'][1]))-0.5)  # random initial conditions

start  = 0.0
step   = 5.0
finish = 100.0


# some extra calculations
if par['gamma'] != 0.0:
    lambda_f = 2.0*np.pi/par['kf']
    kx = par['kf']/par['resonance']
    ky = np.sqrt(par['k0']**2 - kx**2)
    lambda_x = 2.0*np.pi/kx
    lambda_y = 2.0*np.pi/ky
    par['l']=(lambda_x*par['periods']/2.0,lambda_y*par['periods']/2.0)

dx = float(par['l'][0]) / par['n'][0]
dy = float(par['l'][1]) / par['n'][1]

dt = 0.00001/(2.0 * dx**2)
print "dt", dt

par.update(dx=dx, dy=dy)
X,Y=np.mgrid[0:par['n'][0],0:par['n'][1]]
X = X*dx
Y = Y*dy
forcing = par['gamma']*np.cos( par['kf']*X )
step1 = u0.copy()            

# plot first frame (t=start)
pylab.ion()
pylab.clf()
# u0.T is transpose because first index means lines (vertical), but we want it to represent x (horizontal)
ext = [0,par['l'][0],0,par['l'][1]]
im=pylab.imshow(u0.T,origin='lower', interpolation='nearest', extent=ext, cmap='Blues')
cbar=pylab.colorbar()
title=pylab.title('time=%2.1f'%start)
if par['gamma'] != 0.0: # if forcing, then draw red dashed lines
    for r in range(int(par['periods'])):
        pylab.axvline(x=r*lambda_f,linewidth=1, color='red',linestyle='--')
pylab.draw()

t=start
# start loop
for tout in np.arange(start+step,finish+step,step):
    while t < tout:
        step2 = local(step1, dt, forcing, par)
        step1 = step2 + spatial(step1)
        t+=dt
    title.set_text('time=%.1f'%(t))
    im.set_data((step1.real).T)
    im.figure.canvas.draw()
