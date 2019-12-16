import torch
import torch.nn as nn
import torch.nn.functional as F

from .persistence import persistence_lengths
from .utils import conv2d_output_shape

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

    
class LinearView(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size()[0], -1)

    
class View(nn.Module):
    def __init__(self, view_args):
        super().__init__()
        self.view_args = view_args
     
    def forward(self, input):
        return input.view(*self.view_args)
    
    
class EncoderConv2D(nn.Module):
    def __init__(self, input_size, emb_size, filters, dim_batch=1, activation='ReLU'):
        super(EncoderConv2D, self).__init__()
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))
            
        enc_layers = []
        kernel_sizes = []
        strides = []
        paddings = []
        
        for i in range(len(filters)-1):
            enc_layers.append(
                nn.Conv2d(in_channels  = filters[i], 
                          out_channels = filters[i+1], 
                          kernel_size  = 3, 
                          stride       = 2, 
                          padding      = 1, 
                          bias         = True
                         )
            )
            
            kernel_sizes.append(3)
            strides.append(2)
            paddings.append(1)    
            enc_layers.append(self.activation)
            
        self.conv_layers = nn.Sequential(*enc_layers)
        self.linear_view = LinearView()

        h_out, w_out = conv2d_output_shape(input_size[0], input_size[1], kernel_sizes, strides, paddings)
        
        self.fc = ILinear(filters[-1]*h_out*w_out, emb_size//dim_batch, dim_batch)
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = self.linear_view(out)
        out = self.fc(out)
        return out


class DecoderConv2D(nn.Module):
    def __init__(self, emb_size, filters, output_size, dim_batch=1, activation='ReLU'):
        super(DecoderConv2D, self).__init__()
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))
                        
        dec_layers = []
        kernel_sizes = []
        strides = []
        paddings = []
            
        for i in range(len(filters)-1):
            dec_layers.append(
                nn.ConvTranspose2d(in_channels    = filters[i], 
                                   out_channels   = filters[i+1], 
                                   kernel_size    = 3, 
                                   stride         = 2, 
                                   padding        = 1, 
                                   output_padding = 1, 
                                  )
            )
            
            kernel_sizes.append(3)
            strides.append(2)
            paddings.append(1)
            # do not add activation on the last layer
            if i != len(filters)-2:
                dec_layers.append(self.activation)
            
        h_out, w_out = conv2d_output_shape(output_size[0], output_size[1], kernel_sizes, strides, paddings)
        
        enc_dim = [filters[0], h_out, w_out]
        dec_layers = [View(tuple([-1] + enc_dim))] + dec_layers
        
        self.fc = ILinear(emb_size, filters[0]*h_out*w_out//dim_batch, dim_batch)
        self.conv_layers = nn.Sequential(*dec_layers)
        
    def forward(self, x):
        out = self.fc(x)
        out = self.conv_layers(out)
        return out


class AutoencoderConv2D(nn.Module):
    def __init__(self,
                 input_size,
                 emb_size,
                 filters=[3,16,32,64], 
                 dim_batch=1,
                 activation='ReLU',
                ):
        super(AutoencoderConv2D, self).__init__()
        self.encoder = EncoderConv2D(input_size, emb_size, filters, dim_batch, activation)
        self.decoder = DecoderConv2D(emb_size, list(reversed(filters)), input_size, dim_batch, activation)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))