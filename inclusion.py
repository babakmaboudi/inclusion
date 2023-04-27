import torch as xp
import numpy as np
import arviz
import matplotlib.pyplot as plt
import time

# This class computes the KL-expansion for some expansion coefficient:
#
# u = sum v_i * (delta^2 + i^2)^(-s+0.5) * e_i
#
# where e_i are the fourier basis and lambda_i is a positive decaying sequance
# s     : controls the regularity
# delta : controls the correlation length
# N     : number of terms in the exapnsion
class F_map():
    def __init__(self, N=128, delta = 10.):
        self.N = N
        mode = xp.asarray(np.concatenate( [ xp.arange(0, int(N/2) ), xp.arange( -int(N/2),0 ) ] ))
        delta2 = delta*delta
        self.delta2pmode2 = delta2 + mode**2

    def compute_sqrt_lambda(self, s):
        sqrt_lambda = xp.float_power( self.delta2pmode2 , -(s+0.5) )
        norm_factor = xp.linalg.norm(sqrt_lambda)
        self.sqrt_lambda_div = sqrt_lambda/norm_factor

    def make_u_from_v(self, v):
        value = xp.fft.ifft( self.sqrt_lambda_div*v )*float(self.N)
        u =  value.real + value.imag
        return u

    def make_u_from_s_v(self, s, v):
        sqrt_lambda = xp.float_power( self.delta2pmode2 , -(s+0.5) )
        norm_factor = xp.linalg.norm(sqrt_lambda)
        sqrt_lambda_div = sqrt_lambda/norm_factor
        value = xp.fft.ifft( sqrt_lambda_div*v )*float(self.N)
        u =  value.real + value.imag
        return u

# This class constructs an image from a starshaped boundary. 
# Can also plot the uncertainty for the boundary
# input requirements:
# c : the center of the inclusion
# u : the boundary of the inclusion
# sampl_u : samples of the boundary to draw the uncertainty
# view_points : the angles of the line integrals used in the sinogram
class image():
    def __init__(self, num_pixel=128, scale=0.05, min_radius=0.1, center=np.array([0,0])):
        self.num_pixel = num_pixel
        self.scale = scale
        self.min_radius = min_radius
        self.c = center

    def set_center(self, center):
        self.c = center

    def plot_boundary(self, u, ax, c=np.zeros(2), label=None, color=None):
        r = np.zeros( len(u) + 1 )
        r[:-1] = self.scale * np.exp( u ) + self.min_radius
        r[-1] = r[0]
        theta = np.linspace( 0,2*np.pi, len(u)+1, endpoint=True )

        ax.plot(r*np.cos(theta)+self.c[0], r*np.sin(theta)+self.c[1],label=label, color=color)
        ax.set_aspect('equal')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])

    def plot_inclusion(self, u, ax, c=np.zeros(2), label=None):
        r = np.zeros( len(u) + 1 )
        r[:-1] = self.scale * np.exp( u ) + self.min_radius
        r[-1] = r[0]
        theta = np.linspace( 0,2*np.pi, len(u)+1, endpoint=True )

        ax.fill(r*np.cos(theta)+self.c[0], r*np.sin(theta)+self.c[1],label=label, color='blue')
        ax.set_aspect('equal')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])

    def plot_uq(self, samps_u, ax, c=np.zeros(2), label=None, color=None):
        r = self.scale * np.exp( samps_u ) + self.min_radius
        r = np.concatenate( [r, r[:,0].reshape(-1,1)], axis=1 )
        theta = np.linspace( 0,2*np.pi, samps_u.shape[1]+1, endpoint=True )

        hdi_intervals = []
        for i in range(r.shape[1]):
            local_interval = arviz.hdi( r[:,i], hdi_prob=.99 )
            hdi_intervals.append( local_interval.reshape(-1) )
        hdi_intervals = np.array(hdi_intervals)

        x = hdi_intervals[:,1]*np.cos(theta)+self.c[0]
        y = hdi_intervals[:,1]*np.sin(theta)+self.c[1]
        ax.fill(x,y,alpha=.3,color='b', label=r'99% HPD')
        x = hdi_intervals[:,0]*np.cos(theta)+self.c[0]
        y = hdi_intervals[:,0]*np.sin(theta)+self.c[1]
        ax.fill(x,y,'w')

    def plot_view_points(self,ax,theta):
        for t in theta:
            x = 0.95*np.cos( t )
            y = 0.95*np.sin( t )
            #ax.plot(x,y,'o',color='fuchsia')
            ax.quiver(x,y, -0.15*x, -0.15*y, color='k', angles='xy', scale_units='xy', scale=1)

# This class computes the exact sinogram of a star-shaped inclusion. We assume and exponential map to construct positive radius from u to construct the radii of the star-shape inclusion:
#
# r = scale * exp( u ) + minimum_radius
# 
# input paramters:
# center : the center of the inclusion
# scale : the cale of the radii
# minimum_radius : minimum radius of the inclusion
# num_pixel : number of pixels in the detectors
# view_angles : view angles for line integrals in the sinogram
class vector_sino():
    def __init__(self, num_pixel=128, scale=0.05, min_radius=0.1, center=np.array([0,0])):
        self.num_pixel = num_pixel
        self.scale = scale
        self.min_radius = min_radius
        self.c = center

    def make_sino(self, u, view_angles):
        self.r = np.zeros( len(u) + 1 )
        self.r[:-1] = self.scale * np.exp( u ) + self.min_radius
        self.r[-1] = self.r[0]
        self.theta = np.linspace( 0,2*np.pi, len(u)+1, endpoint=True )
        sino = []

        for t in view_angles:
            projection = self.compute_projection(t)
            sino.append(projection)

        return np.array(sino)


    def compute_projection(self,rot):
        rotated_theta = self.theta - rot
        x = self.r*np.cos(rotated_theta)+self.c[0]
        y = self.r*np.sin(rotated_theta)+self.c[1]

        x_min = np.min(x)
        x_max = np.max(x)

        bnd_points = np.concatenate( [x.reshape(-1,1),y.reshape(-1,1)], axis=1 )

        pixel_location = np.linspace( -1,1,self.num_pixel )
        projection = np.zeros_like(pixel_location)
        for j in range( len(pixel_location) ):
            w = pixel_location[j]
            if( (w<x_min) or (w>x_max) ):
                continue
            points = []

            x_coord = bnd_points[:,0] - w
            crossing = x_coord[:-1]*x_coord[1:]
            idx = np.argwhere(crossing<0).reshape(-1)

            for i in idx:
                temp = ( (y[i+1]-y[i])*w + y[i]*x[i+1] - y[i+1]*x[i] )/(x[i+1]-x[i])
                points.append( np.array([w, temp]) )

            if( len(points) == 2 ):
                projection[j] = np.linalg.norm( points[0] - points[1] )
            elif( len(points) > 2 ):
                points = np.array(points)
                points = points[points[:, 1].argsort()]
                for k in range(0,points.shape[0],2):
                    projection[j] += np.linalg.norm( points[k,:] - points[k+1,:] )
        return projection

# This class computes the exact sinogram like vector_sino class, the difference is
# that the inner loop of sinogram calculation is carried out with tensors in
# Pytorch package
class tensor_sino():
    def __init__(self, num_pixel=128, scale=0.05, min_radius=0.1, center=np.array([0,0])):
        self.num_pixel = num_pixel
        self.scale = scale
        self.min_radius = min_radius
        self.pixel_location = xp.asarray( np.linspace( -1,1,self.num_pixel ) )
        self.c=center

    def make_sino(self, u, view_angles):
        self.r = xp.zeros( len(u) + 1 )
        self.r[:-1] = self.scale * xp.exp( u ) + self.min_radius
        self.r[-1] = self.r[0]
        self.theta = xp.asarray( np.linspace( 0,2*np.pi, len(u)+1, endpoint=True ) )
        sino = xp.zeros((len(view_angles),self.num_pixel))
        for i,t in enumerate(view_angles):
            sino[i] = self.compute_projection(t)

        return sino

    def compute_projection(self,rot):
        rotated_theta = self.theta - rot
        x = self.r*xp.cos( rotated_theta )+self.c[0]
        y = self.r*xp.sin( rotated_theta )+self.c[1]

        projection = xp.zeros(self.num_pixel)
        w = self.pixel_location
        X = xp.broadcast_to(x, (w.shape[0],x.shape[0]))
        W = xp.broadcast_to(w, (x.shape[0],w.shape[0])).T

        x_rel_to_line = X-W # Check if we need cat on x (above)
        crossing_indicator = x_rel_to_line[:,:-1]*x_rel_to_line[:,1:]

        xd = xp.diff(x)
        yd = xp.diff(y)
        xs = xp.roll(x, -1)
        ys = xp.roll(y, -1)

        W2 = xp.broadcast_to(w, (yd.shape[0],w.shape[0])).T
        Yd = xp.broadcast_to(yd, (w.shape[0],yd.shape[0]))

        y_collision = (Yd*W2 + (y*xs)[:-1] - (ys*x)[:-1]) / xd
        dummy1 = (Yd*W2 + (y*xs)[:-1] - (ys*x)[:-1]) 
        
        y_collision[crossing_indicator>=0] = 0

        y_collision_sorted = xp.sort(y_collision, axis=1)[0]
        
        projection = xp.sum(xp.diff(y_collision_sorted, axis=1)[:, ::2], axis=1)

        return projection

# This class computes the exact sinogram like vector_sino class, the difference 
# is that the inner and outer loop for the sinogram calculation  is carried out
# with tensors in Pytorch package
class full_tensor_sino():
    def __init__(self, num_pixel=128, scale=0.05, min_radius=0.1,center=np.array([0,0])):
        self.num_pixel = num_pixel
        self.scale = scale
        self.min_radius = min_radius
        self.pixel_location = xp.asarray( np.linspace( -1,1,self.num_pixel ) )
        self.c = center

    def set_center(self,center):
        self.c = center

    def make_sino(self, u, view_angles):
        r = xp.zeros( len(u) + 1 )
        r[:-1] = self.scale * xp.exp( u ) + self.min_radius
        r[-1] = r[0]
        theta = xp.asarray( np.linspace( 0,2*np.pi, len(u)+1, endpoint=True ) )

        THETA = xp.broadcast_to(theta, (view_angles.shape[0],theta.shape[0]))
        ROT = xp.broadcast_to(view_angles, (theta.shape[0],view_angles.shape[0])).T
        ROTATED_THETA = THETA - ROT
        R = xp.broadcast_to(r, (view_angles.shape[0],r.shape[0]))

        X = R*xp.cos( ROTATED_THETA )+self.c[0]
        Y = R*xp.sin( ROTATED_THETA )+self.c[1]

        w = self.pixel_location
        XX = xp.broadcast_to(X, (w.shape[0],X.shape[0],X.shape[1])).permute(1,0,2)
        WW = xp.broadcast_to(w, (X.shape[0],X.shape[1],w.shape[0])).permute(0,2,1)

        x_rel_to_line = XX-WW
        crossing_indicator = x_rel_to_line[:,:,:-1]*x_rel_to_line[:,:,1:]

        Xd = xp.diff(X, dim=1)
        Yd = xp.diff(Y, dim=1)
        Xs = xp.roll(X, shifts=-1, dims=1)
        Ys = xp.roll(Y, shifts=-1, dims=1)

        WW2 = xp.broadcast_to(w, (Yd.shape[0],Yd.shape[1],w.shape[0])).permute(0,2,1)
        YYd = xp.broadcast_to(Yd, (w.shape[0],Yd.shape[0],Yd.shape[1])).permute(1,0,2)

        Y_collision = (YYd*WW2 + (Y*Xs)[:,None,:-1] - (Ys*X)[:,None,:-1]) / Xd[:,None,:]

        Y_collision[crossing_indicator>=0] = 0
        Y_collision_sorted = xp.sort(Y_collision, axis=2)[0]

        return xp.sum(xp.diff(Y_collision_sorted, axis=2)[:,:, ::2], axis=2)

