import numpy as np
import torch
import gudhi as gd

def persistence_lengths(batch, 
                        dim=0, 
                        device=torch.device('cpu'), 
                        max_edge_length=np.inf,
                       ):
    """Use Gudhi to calculate persistence diagrams.

    batch: point clouds input,
    dim: homology dimension (0 for connectivity),
    device: device for torch tensor,
    max_edge_length: threshold on the Vietoris-Rips scale parameter. By default, it builds
    all the simplices in the filtration.
    """
    rips_complex = gd.RipsComplex(batch, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim)
    simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    persistence_intervals = simplex_tree.persistence_intervals_in_dimension(dim)
    return torch.FloatTensor(persistence_intervals[:, 1]-persistence_intervals[:, 0]).to(device)
    
def barcode_stats(data, 
                  dim=0, 
                  device=torch.device('cpu'),
                  max_edge_length=np.inf,
                 ):
    """Min, average and max barcode lengths of data.
    data: point clouds input,
    dim: homology dimension (0 for connectivity),
    device: device for torch tensor,
    max_edge_length: threshold on the Vietoris-Rips scale parameter. By default, it builds
    all the simplices in the filtration.
    """
    # remove inf
    barcodes = np.ma.masked_invalid(persistence_lengths(data, 
                                                        dim=dim,
                                                        device=device,
                                                        max_edge_length=max_edge_length,
                                                       ))
    return [barcodes.min(), barcodes.mean(), barcodes.max()]
