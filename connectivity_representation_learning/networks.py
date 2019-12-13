import torch
import torch.nn as nn
import torch.nn.functional as F

from .persistence import persistence_lengths

class ILinear(nn.Linear):
    """I-Linear neural unit.
    This corresponds to dim_batch linear units operating 
    independantly on dimensional batches, or branches. This 
    is equivalent to a standard linear unit with a block 
    diagonal weight matrix.
    """
    def __init__(self, input_size, output_size_b, dim_batch):        
        assert input_size % dim_batch == 0 
        input_size_b = input_size//dim_batch
        super().__init__(input_size, output_size_b*dim_batch)      
        
        mask = torch.zeros_like(self.weight)
        for i in range(dim_batch):
            mask[i*output_size_b:(i+1)*output_size_b, 
                 i*input_size_b:(i+1)*input_size_b] = 1

        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask)
    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size, dim_batch=1, activation='ReLU'):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if dim_batch == 1:
            self.fc2 = nn.Linear(hidden_size, emb_size)
        else:
            self.fc2 = ILinear(hidden_size, emb_size//dim_batch, dim_batch)
        
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
    def __init__(self, emb_size, hidden_size, output_size, dim_batch=1, activation='ReLU'):
        super(Decoder, self).__init__()
        if dim_batch == 1:
            self.fc1 = nn.Linear(emb_size, hidden_size)
        else:
            self.fc1 = ILinear(emb_size, hidden_size//dim_batch, dim_batch)
        
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
                 dim_batch=1,
                 activation='ReLU',
                ):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size_encoder, emb_size, dim_batch, activation)
        self.decoder = Decoder(emb_size, hidden_size_decoder, input_size, dim_batch, activation)

    def forward(self, x):
        return self.decoder(self.encoder(x))
