"""Spectral method integrator for the Forced Swift Hohenberg equation.
SH Equation:
     u_t = epsilon*u + lambda*u^2 - u^3 - [ d^2/(dx)^2 + d^2/(dy)^2 + k0^2 ]^2 * u + gamma*u*cos(kf*x)
Periodic boundary conditions are assumed.
"""
__version__=1.0
__author__ = """Yair Mau (yairmau@gmail.com)"""

import numpy
from numpy.fft import fftn, ifftn, fftfreq
import pylab

def nonlinear(u, dt, forcing, par):
    """This is the nonlinear function
       \lambda u^2 - u^3 + u * forcing   
    """
    return u + dt * (par['Lambda']*u**2 - u**3 + u*forcing); # Local terms

def linear(u,exponent):
    "This is the linear function"
    return numpy.fft.ifftn( exponent *  numpy.fft.fftn(u) ); # Spatial terms

def spectral_multiplier(par,dt):
        dx=par['dx']
        dy=par['dy']
        n0=par['n'][0]
        n1=par['n'][1]
        f0=2.0*numpy.pi*fftfreq(n0,dx)
        f1=2.0*numpy.pi*fftfreq(n1,dy)
        kx=numpy.outer(f0,numpy.ones(n0))
        ky=numpy.outer(numpy.ones(n1),f1)
        return numpy.exp(dt*(par['epsilon']-(par['k0']-kx**2-ky**2)**2))

par = {'epsilon':0.1,
       'k0':1.0,
       'Lambda':0.2,
       'gamma':0.,
       'kf':1.2,
       'n':(60,60),
       'l':(60,60),
       'resonance':2.0,
       'periods':8,
       }
       
# initial condition
u0 = 0.5*(numpy.random.random((par['n'][0],par['n'][1]))-0.5)  # random initial conditions

start  = 0.0
step   = 1.0e-1
finish = 1.0

dx = float(par['l'][0]) / par['n'][0]
dy = float(par['l'][1]) / par['n'][1]
dt = 0.00001/(2.0 * dx**2)
print "dt", dt

# some extra calculations
if par['gamma'] != 0.0:
    lambda_f = 2.0*numpy.pi/par['kf']
    kx = par['kf']/par['resonance']
    ky = numpy.sqrt(par['k0']**2 - kx**2)
    lambda_x = 2.0*numpy.pi/kx
    lambda_y = 2.0*numpy.pi/ky
    par['l']=(lambda_x*par['periods']/2.0,lambda_y*par['periods']/2.0)


par.update(dx=dx, dy=dy)
X,Y=numpy.mgrid[0:par['n'][0],0:par['n'][1]]
X = X*dx
Y = Y*dy
forcing = par['gamma']*numpy.cos( par['kf']*X )
step1 = u0.copy()            
spec_mult=spectral_multiplier(par,dt)

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
for tout in numpy.arange(start+step,finish+step,step):
    while t < tout:
        step2 = nonlinear(step1, dt, forcing, par)
        step1 = linear(step2,spec_mult)
        t+=dt
    title.set_text('time=%.1f'%(t))
    im.set_data((step1.real).T)
    im.figure.canvas.draw()
