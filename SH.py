"""
This is a modified version of SH.py written by Yair Mau, to integrate SH model
both in semi-spectral method and finite differences method
Spectral method integrator for the Forced Swift Hohenberg equation.
SH Equation:
     u_t = epsilon*u + lambda*u^2 - u^3 - [ d^2/(dx)^2 + d^2/(dy)^2 + k0^2 ]^2 * u + gamma*u*cos(kf*x)
Periodic boundary conditions are assumed.
"""
__version__=1.0
__author__ = """Omer Tzuk (cliffon@gmail.com)"""
from numba import jit
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import pylab


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

dx = float(par['l'][0]) / par['n'][0]
dy = float(par['l'][1]) / par['n'][1]
par.update(dx=dx, dy=dy)
X,Y=np.mgrid[0:par['n'][0],0:par['n'][1]]
X = X*dx
Y = Y*dy
forcing = par['gamma']*np.cos( par['kf']*X )      

dx2=dx**2 # To save CPU cycles, we'll compute Delta x^2
dy2=dy**2 # and Delta y^2 only once and store them.
h2 = dx2
h4 = h2**2

# initial condition
u0 = 0.5*(np.random.random((par['n'][0],par['n'][1]))-0.5)  # random initial conditions
#u0 = 0.5* np.ones((par['n'][0],par['n'][1]))  # set initial conditions of constant value of 0.5

start  = 0.0
step   = 5.
finish = 100.0
#dt     = 0.1 # Semi-spectral time step
dt     = dx2 / 100000 # FDM time step


def main():
	
	# some extra calculations
	if par['gamma'] != 0.0:
	    lambda_f = 2.0*np.pi/par['kf']
	    kx = par['kf']/par['resonance']
	    ky = np.sqrt(par['k0']**2 - kx**2)
	    lambda_x = 2.0*np.pi/kx
	    lambda_y = 2.0*np.pi/ky
	    par['l']=(lambda_x*par['periods']/2.0,lambda_y*par['periods']/2.0)
	
	
	u_old = u0.copy()            
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
	#pylab.draw()
	im_index = 1
	pylab.savefig("u_t_"+str(im_index)+".png")
	
	t=start
	u_new = u_old
	print "dx", dx, " dt", dt
	# start loop
	for tout in np.arange(start+step,finish+step,step):
	    while t < tout:
			# Working with Yair's semi spectral method
			#u_new = nonlinear(u_old, dt, forcing, par)
			#u_old = linear(u_new,spec_mult)
			
			# Working with finite difference method
			u_new = evolve_SH(u_new, dt, par)
			
			t+=dt
		
			
	    title.set_text('time=%.1f'%(t))
	    im.set_data((u_new.real).T)
	    #im.figure.canvas.draw()
	    im_index+=1
	    pylab.savefig("u_t_"+str(im_index)+".png")
	    


def spectral_multiplier(par,dt):
        dx=par['dx']
        dy=par['dy']
        n0=par['n'][0]
        n1=par['n'][1]
        f0=2.0*np.pi*fftfreq(n0,dx)
        f1=2.0*np.pi*fftfreq(n1,dy)
        kx=np.outer(f0,np.ones(n0))
        ky=np.outer(np.ones(n1),f1)
        return np.exp(dt*(par['epsilon']-(par['k0']-kx**2-ky**2)**2))

@jit
def nonlinear(u, dt, forcing, par):
    """This is the nonlinear function
       \lambda u^2 - u^3 + u * forcing   
    """
    return u + dt * (par['Lambda']*u**2 - u**3 + u*forcing); # Local terms
@jit
def linear(u,exponent):
    "This is the linear function"
    return np.fft.ifftn( exponent *  np.fft.fftn(u) ); # Spatial terms



@jit
def evolve_SH(u_old, dt, par):
	""" (ui, dt, par) -> u(i+1)
	This function implement u_new = u_old + dt * ( local(u) + spatial(u) )
	"""
	return u_old + dt*(local_SH(u_old, par) + spatial_SH(u_old, par))

@jit
def local_SH(ui, par):
	"""
	This is the nonlinear function
       \lambda u^2 - u^3 + u * forcing
	"""
	
	local = (par['epsilon']*ui + par['Lambda']*ui**2 - ui**3)
	return local

@jit
def laplacian(u_old):
	ui = u_old.copy()
	return (np.roll(ui,1,axis=0) + np.roll(ui,-1,axis=0) + np.roll(ui,1,axis=1) + np.roll(ui,-1,axis=1) -4*ui)/h2


@jit
def spatial_SH(ui, par):
	"""
	This function uses a numpy expression to
	evaluate the derivatives in the Laplacian, and
	calculates u[i,j] based on ui[i,j].
	"""
	lap = laplacian(ui)
	#laplacian_sq = (np.roll(ui,2,axis=0) + np.roll(ui,-2,axis=0) + np.roll(ui,2,axis=1) + np.roll(ui,-2,axis=1) - 4*np.roll(ui,1,axis=0) - 4*np.roll(ui,-1,axis=0) - 4*np.roll(ui,1,axis=1) - 4*np.roll(ui,-1,axis=1) +12*ui)/h4
	lap_sq = laplacian(lap)
	return (lap_sq + 2*par['k0']*lap + par['k0']**2)



if __name__ == "__main__":
	main()

