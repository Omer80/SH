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
from matplotlib import pyplot as plt


def laplacian(u):
    "This is the laplacian term"
    hh = par['dx']**2
    #print 'hh', hh
    lap = (np.roll(u,1,axis=0) + np.roll(u,-1,axis=0) + np.roll(u,1,axis=1) + np.roll(u,-1,axis=1) -4.0*u)/hh
    #print "laplacian", lap
    return lap

def laplaciansq(u):
	"This is the laplacian^2 term"
	dx2 = par['dx']**2
	dx4 = dx**2
	dy2 = par['dy']**2
	dy4 = dy**2
	u_xx = (np.roll(u,1,axis=1) + np.roll(u,-1,axis=1) -2.0*u)/dx2
	u_xxyy = (np.roll(u_xx,1,axis=0) + np.roll(u_xx,-1,axis=0) -2.0*u_xx)/dy2
	u_xxxx = (-4.0*np.roll(u,1,axis=1) -4.0*np.roll(u,-1,axis=1) +1.0*np.roll(u,2,axis=1) +1.0*np.roll(u,-2,axis=1) +6.0*u)/dx4
	u_yyyy = (-4.0*np.roll(u,1,axis=0) -4.0*np.roll(u,-1,axis=0) +1.0*np.roll(u,2,axis=0) +1.0*np.roll(u,-2,axis=0) +6.0*u)/dy4
	#print 'dx2:',dx2, ' dx4:',dx4,' dy4:',dy4
	lap2 = (u_xxxx + 2.0*u_xxyy + u_yyyy)
	#print "laplacian squared:", lap2
	return lap2

def local(u, dt, forcing, par):
    """This is the nonlinear function
       \lambda u^2 - u^3 + u * forcing   
    """
    return (dt * (par['Lambda']*u**2 - u**3 + u*forcing))

def spatial(u):
	"""
	This is the spatial term of SH
	 -[ d^2/(dx)^2 + d^2/(dy)^2 + k0^2 ]^2 * u
	"""
	return (dt * (-1.0)*(laplaciansq(u) + 2.0*laplacian(u)*(par['k0'])+(par['k0'])**2.0)) 


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
step   = 1.0
finish = 10.0


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

dt = 0.0000001/(2.0 * dx**2)
print "dt", dt

par.update(dx=dx, dy=dy)
X,Y=np.mgrid[0:par['n'][0],0:par['n'][1]]
X = X*dx
Y = Y*dy
forcing = par['gamma']*np.cos( par['kf']*X )
u_old = u0.copy()            

# plot first frame (t=start)
plt.ion()
plt.clf()
# u0.T is transpose because first index means lines (vertical), but we want it to represent x (horizontal)
ext = [0,par['l'][0],0,par['l'][1]]
im=plt.imshow(u0.T,origin='lower', interpolation='nearest', extent=ext, cmap='Blues')
cbar=plt.colorbar()
title=plt.title('time=%2.1f'%start)
if par['gamma'] != 0.0: # if forcing, then draw red dashed lines
    for r in range(int(par['periods'])):
        plt.axvline(x=r*lambda_f,linewidth=1, color='red',linestyle='--')
        
#plt.draw()
plt.savefig('img_t_0.png', format='png', dpi=1000)

t=start
# start loop
for tout in np.arange(start+step,finish+step,step):
    while t < tout:
		#print "time:", t
		u_new = u_old + local(u_old, dt, forcing, par) + spatial(u_old)
		u_old = u_new
		t+=dt
    title.set_text('time=%.1f'%(t))
    im.set_data((u_new.real).T)
    plt.savefig('img_t_%2.0f'%(t)+'.png', format='png', dpi=1000)
    #im.figure.canvas.draw()
