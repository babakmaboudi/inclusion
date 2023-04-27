The codes for constructing an starshaped inclusion and computing the exact sinogram.

Follow script.py to create and inclusion and visualize it and then computing its sinogram.

We use a KL expansion to define the boundary of the starshaped inclusion.

$$ u = \sum_{i=1}^{N} v_i * \sqrt{\lambda_i} * e_i $$

where $v_i$ are standard normal Gaussian, $\lambda_i$ are decreasing positive values, $e_i$ are Fourier basis functions.

We can then compute the radii of the starshaped inclusion as

$$ r = s*\exp( u ) + r_0 $$

where $s$ is some scale and $r_0$ is the minimum radius. Now we can find the $x$ and $y$ corrdinates of the starshaped inclusion vertices as

$$ r_x = \cos(\theta) * r + c_x, r_y = \sin(\theta) * r + c_y $$

where $\theta \in [0,2\pi]$ and $c = [c_x,c_y]$ is the center of the inclusion.

The sinograms are evaluated exactly from $u$ (a polygon).
