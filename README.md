# connectivity_loss
Investigate the use of homology-related losses in representation learning.

Connectivity loss as a way to control the topology of encoder latent space was first introduced in http://proceedings.mlr.press/v97/hofer19a/hofer19a.pdf.
This loss penalizes latent space configurations that deviate from regular arrangements of points. 
More formally, it penalizes the L1 deviation of the 0-dimensional persistence barcode lengths to a uniform set of barcode length,
calculated on the Vieteoris-Rips filtration of the latent space. The persistence diagram encodes homological information about the 
latent space, in particular 0-homology characterizes the connectivity of the space.

Calculations of persistence diagrams are performed with Gudhi (http://gudhi.gforge.inria.fr/).
