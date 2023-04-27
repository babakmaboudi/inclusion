import numpy as np
import torch as xp
import matplotlib.pyplot as plt
from inclusion import F_map, image, vector_sino, tensor_sino, full_tensor_sino

# defining the discretization of the starshaped inclusion
N = 256

# defining the KL-expansion coefficients
v = np.random.standard_normal(N)

# defining the regularity of the star-shaped inclusion
s = 1

# defining the KL explansion
F = F_map(N) # This class computes the KL expansion
u = F.make_u_from_s_v(s, v) # computing the KL expansion from v and s

# Now we want to visualize the starshape inclusion in the unit square
c = np.array( [0.5,-0.2] ) # center of the star-shaped inclusion

im = image() # this class visualizes the inlcusion
im.set_center(c) # setting the center of the inclusion
f, axes = plt.subplots(1,3) # defining the pyplots figure

im.plot_boundary(u, axes[0]) # plotting only the boundary
axes[0].set_title('inclusion boundary')
im.plot_inclusion(u, axes[1]) # plotting the inlcusion
axes[1].set_title('inclusion')

# Now we want to plot the uncertainty we sample. We sample multiple boundaries
N_samples = 100 # number of samples
v_samples = [] # array holding v_samples
u_samples = [] # array holding u_samples
for i in range(N_samples):
    v_samples.append(np.random.standard_normal(N))
    u_samples.append( F.make_u_from_s_v(s, v_samples[-1]).detach().numpy() ) # converting to numpy array from torch tensor
u_samples = np.array(u_samples)

mean_v = np.mean(v_samples, axis=0)
mean_u = F.make_u_from_s_v(s, mean_v)
im.plot_uq(u_samples, axes[2])
im.plot_boundary(mean_u, axes[2], label='mean')
im.plot_boundary(u, axes[2], label='sample')
axes[2].set_title('uncertainty')
axes[2].legend()

# Now we want to compute the sinograms
# first we define view angles
N_view = 128 # number of view angles
view_angles = np.linspace(0, np.pi, N_view)

sino_vector = vector_sino() # this uses numpy to create a sinogram
sino_tensor = full_tensor_sino() # this uses Pytorch to create a sinogram

sino1 = sino_vector.make_sino(u.detach().numpy(), view_angles)
sino2 = sino_tensor.make_sino(u, xp.asarray(view_angles) )

# visualizing the sinograms
f, axes = plt.subplots(1,2)
axes[0].imshow(sino1)
axes[1].imshow(sino2.detach().numpy())


plt.show()