import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .networks import Autoencoder, AutoencoderConv2D
from .persistence import persistence_lengths
from .utils import triangular_from_linear_index, linear_index_from_triangular
    
    
class ConnectivityLayer(nn.Module):
    def __init__(self, connectivity_penalty, eta=torch.rand(1)):
        super(ConnectivityLayer, self).__init__()
        self.eta = nn.Parameter(eta)
        self.eta.requires_grad = True
    
        self.connectivity_penalty = connectivity_penalty
    
    def forward(self, pdist, indicators):
        return torch.sum(indicators*torch.abs(self.eta-pdist)) * self.connectivity_penalty
    
class Model(nn.Module):
    """Autoencoder with connectivity penalization.
    """
    def __init__(self,
                 config_layers,
                 batch_size=50,
                 dim_batch=1,
                 use_cuda=False,
                 lr=0.001,
                 eta = 2.0,
                 tol=1e-4,
                 connectivity_penalty=1.0,
                 activation='ReLU',
                 use_trainable_eta=False,
                ):
        super(Model, self).__init__()

        self.use_cuda = use_cuda
        self.batch_size = batch_size
        
        self.config_layers = config_layers

        # number of dimensional batches
        self.dim_batch = dim_batch
        
        # dim of latent space
        self.emb_size = config_layers['emb_size']
        
        # parameter for the connectivity loss
        self.eta = eta
        # numerical precision for distance lookup
        self.tol = tol
        # weight to balance reconstruction and connectivity loss during training
        self.connectivity_penalty = connectivity_penalty

        self.connectivity_layer = ConnectivityLayer(self.connectivity_penalty, 
                                                    torch.tensor(self.eta),
                                                   )
        
        if config_layers['type'] == 'conv2d':
            self.autoencoder = AutoencoderConv2D(
                config_layers['input_size'],
                config_layers['emb_size'],
                config_layers['filters'],
                dim_batch,
                activation,
            ).to(self.device)
        elif config_layers['type'] == 'mlp':
            self.autoencoder = Autoencoder(
                config_layers['input_size'],
                config_layers['hidden_size_encoder'],
                config_layers['emb_size'],
                config_layers['hidden_size_decoder'],
                dim_batch,
                activation,
            ).to(self.device)        
        else:
            raise Exception('Unknown layers type {}'.format(config_layers['type']))

        parameters = list(self.autoencoder.parameters())+list(self.connectivity_layer.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=lr)

        # used for caching during training
        self.pdist = None
        self.zero_persistence_lengths = None
        
        # whether to learn eta during training or use a prescribed value 
        self.use_trainable_eta=use_trainable_eta
        
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
                        connectivity_loss_branches = torch.empty(self.dim_batch).to(self.device)
                        
                        # split across dimensional branches
                        for b in range(self.dim_batch):
                            # selects a dimensional batch of the latent space
                            latent_b = latent[:, b*self.emb_size//self.dim_batch:(b+1)*self.emb_size//self.dim_batch]
                            # compute the distance matrix
                            self.pdist = F.pdist(latent_b)
                            # compute the 0-barcode lengths
                            self.zero_persistence_lengths = persistence_lengths(latent_b, dim=0, device=self.device)
                            # compute the indicator of indices that correspond
                            # to pairs of points such that the intersection of their
                            # balls in the Vietoris-Rips scheme is a death event 
                            # for connected components.
                            indicators = torch.FloatTensor(
                                [self.indicator(triangular_from_linear_index(self.batch_size, k)) for k in range(self.pdist.shape[0])]
                            ).to(self.device)
                            
                            # compute connectivity loss on the current branch
                            if self.use_trainable_eta:
                                connectivity_loss_branches[b] = self.connectivity_layer.forward(
                                self.pdist,
                                indicators
                            )
                            else:
                                connectivity_loss_branches[b] = torch.sum(indicators*torch.abs(self.eta-self.pdist)) * self.connectivity_penalty + 0.01*torch.norm(self.eta-self.connectivity_layer.eta)
                        
                        
                        
                        # aggregate all the connectivity losses in the dimensional batches
                        connectivity_loss = torch.sum(connectivity_loss_branches)
                            
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
