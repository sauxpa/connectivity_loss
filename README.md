# connectivity_loss
Investigate the use of an homology-related loss in representation learning.

Connectivity loss as a way to control the topology of encoder latent space was first introduced in http://proceedings.mlr.press/v97/hofer19a/hofer19a.pdf.
This loss penalizes latent space configurations that deviate from regular arrangements of points. 
More formally, it penalizes the L1 deviation of the 0-dimensional persistence barcode lengths to a uniform set of barcode length,
calculated on the Vieteoris-Rips filtration of the latent space. The persistence diagram encodes homological information about the 
latent space, in particular 0-homology characterizes the connectivity of the space.

Calculations of persistence diagrams are performed with Gudhi (http://gudhi.gforge.inria.fr/).

* cifar_reconstruction : comparison of standard convolutional autoencoder vs connectivity-optimized ones, with and without dimension branching.

* connectivity_high_dim : investigate the connectivity of random point clouds as a function of dimension via their 0-persistence diagrams.

* gaussian_high_dim_toy_reconstruction : high-dimensional examples of autoencoder regularization via connectivity loss.

* gaussian_high_dim_toy_reconstruction_branches : high-dimensional examples of autoencoder regularization via connectivity loss using dimensiona branching.

* gaussian_toy_reconstruction : 2d and 3d examples of autoencoder regularization via connectivity loss.

* gaussian_toy_reconstruction_trainable_eta : make eta (the target barcode length) a trainable parameter that the netowrks optimize dring training, rather than a manually tuned hyperparameter.

* stronger_connectivity_denser_latent_space : similar to gaussian_toy_reconstruction, visualize the densification effect of the latent space with higher penalties on the connectivity loss.
