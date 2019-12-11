import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gudhi as gd
from tqdm import tqdm

from .utils import triangular_from_linear_index, linear_index_from_triangular

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
#     simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
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
    
    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size, activation='ReLU'):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))
            
    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out


class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, activation='ReLU'):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out


class Autoencoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size_encoder,
                 emb_size,
                 hidden_size_decoder,
                 activation='ReLU',
                ):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size_encoder, emb_size, activation)
        self.decoder = Decoder(emb_size, hidden_size_decoder, input_size, activation)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    
class Model(nn.Module):
    """Autoencoder with connectivity penalization.
    """
    def __init__(self,
                 input_size,
                 hidden_size_encoder,
                 emb_size,
                 hidden_size_decoder,
                 batch_size=50,
                 use_cuda=False,
                 lr=0.001,
                 eta = 2.0,
                 tol=1e-4,
                 connectivity_penalty=1.0,
                 activation='ReLU',
                ):
        super(Model, self).__init__()

        self.use_cuda = use_cuda
        self.batch_size = batch_size

        # parameter for the connectivity loss
        self.eta = eta
        # numerical precision for distance lookup
        self.tol = tol
        # weight to balance reconstruction and connectivity loss during training
        self.connectivity_penalty = connectivity_penalty

        self.autoencoder = Autoencoder(
            input_size,
            hidden_size_encoder,
            emb_size,
            hidden_size_decoder,
            activation,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)

        # used for caching during training
        self.pdist = None
        self.zero_persistence_lengths = None
        
    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

    

    def indicator(self, idx):
        """Returns True if idx corresponds to a pair of points the distance of which
        corresponds to critical filtration value in the Vietoris-Rips complex.
        """
        k = linear_index_from_triangular(self.batch_size, idx[0], idx[1])
        return True in torch.isclose(self.pdist[k], self.zero_persistence_lengths, self.tol)
    
    def train(self, data, n_epochs):
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)

        tdqm_dict_keys = ['connectivity loss', 'reconstruction loss']
        tdqm_dict = dict(zip(tdqm_dict_keys, [0.0, 0.0]))

        for epoch in range(n_epochs):
            # initialize cumulative losses to zero at the start of epoch
            total_connectivity_loss = 0.0
            total_reconstruction_loss = 0.0

            with tqdm(total=len(loader),
                      unit_scale=True,
                      postfix={'connectivity loss': 0.0, 'reconstruction loss': 0.0},
                      desc="Epoch : %i/%i" % (epoch+1, n_epochs),
                      ncols=100
                     ) as pbar:
                for batch_idx, batch in enumerate(loader):
                    batch = batch.type(torch.float32).to(self.device)

                    latent = self.autoencoder.encoder(batch)
                    
                    # in pure reconstruction mode, 
                    # skip the Gudhi part to speed training up
                    if self.connectivity_penalty != 0.0:
                        # calculate pairwise distance matrix
                        # pdist is a flat tensor representing
                        # the upper triangle of the pairwise
                        # distance tensor.
                        self.pdist = F.pdist(latent)
                        self.zero_persistence_lengths = persistence_lengths(latent, dim=0, device=self.device)
                        indicators = torch.FloatTensor(
                            [self.indicator(triangular_from_linear_index(self.batch_size, k)) for k in range(self.pdist.shape[0])]
                        ).to(self.device)
                        connectivity_loss = torch.sum(indicators*torch.abs(self.eta-self.pdist)) * self.connectivity_penalty
                    else:
                        connectivity_loss = torch.FloatTensor([0.0]).to(self.device)
                        
                    reconstruction_loss = F.mse_loss(batch, self.autoencoder.decoder(latent))
                    loss = reconstruction_loss + connectivity_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_connectivity_loss += connectivity_loss.item()
                    total_reconstruction_loss += reconstruction_loss.item()

                    # logging
                    tdqm_dict['connectivity loss'] = total_connectivity_loss/(batch_idx+1)
                    tdqm_dict['reconstruction loss'] = total_reconstruction_loss/(batch_idx+1)
                    pbar.set_postfix(tdqm_dict)
                    pbar.update(1)
