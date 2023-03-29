import torch
from torch import nn
#from torchmeta.modules import (MetaModule, MetaSequential)
#from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F
import torch
from torch import nn
from collections import OrderedDict

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)



class BatchLinear(nn.Linear, ): #MetaModule):
    #A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    #hypernetwork.
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']
        #print(input.shape)
        #print(weight.shape)

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

'''
class SineLayer(nn.Linear, MetaModule):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        #print(input.shape)
        #print(self.in_features)
        #print(self.out_features)
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
 '''
'''
class LipBoundedPosEnc(nn.Module):

    def __init__(self, inp_features, n_freq=5, cat_inp=True, retain_z=True):
        super().__init__()
        self.inp_feat = inp_features
        self.n_freq = n_freq
        self.cat_inp = cat_inp
        self.out_dim = 2 * self.n_freq * self.inp_feat
        if self.cat_inp:
            self.out_dim += self.inp_feat
        
    def forward(self, x):
        if retain_z:
            z = x[..., [-1]]
            x = x[..., [0, 1]] 
        """
        :param x: (bs, npoints, inp_features)
        :return: (bs, npoints, 2 * out_features + inp_features)
        """
        #assert len(x.size()) == 3
        bs, npts = x.size(0), x.size(1)
        const = (2 ** torch.arange(self.n_freq) * np.pi).view(1, 1, 1, -1)
        const = const.to(x)

        # Out shape : (bs, npoints, out_feat)
        cos_feat = torch.cos(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        sin_feat = torch.sin(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        out = torch.cat(
            [sin_feat, cos_feat], dim=-1).view(
            bs, npts, 2 * self.inp_feat * self.n_freq)
        const_norm = torch.cat(
            [const, const], dim=-1).view(
            1, 1, 1, self.n_freq * 2).expand(
            -1, -1, self.inp_feat, -1).reshape(
            1, 1, 2 * self.inp_feat * self.n_freq)

        if self.cat_inp:
            out = torch.cat([out, x], dim=-1)
            const_norm = torch.cat(
                [const_norm, torch.ones(1, 1, self.inp_feat).to(x)], dim=-1)

            return out / const_norm / np.sqrt(self.n_freq * 2 + 1)
        else:

            return out / const_norm / np.sqrt(self.n_freq * 2)
'''


class LipBoundedPosEnc(nn.Module):

    def __init__(self, inp_features, n_freq=5, cat_inp=True):
        super().__init__()
        self.inp_feat = inp_features #- 1
        self.n_freq = n_freq
        self.cat_inp = cat_inp
        self.out_dim = 2 * self.n_freq * self.inp_feat #+ 1
        if self.cat_inp:
            self.out_dim += self.inp_feat

    def forward(self, x):
        #z = x[..., [-1]]
        x = x[..., [0, 1, 2]]

        """
        :param x: (bs, npoints, inp_features)
        :return: (bs, npoints, 2 * out_features + inp_features)
        """
        # assert len(x.size()) == 3
        bs, npts = x.size(0), x.size(1)
        const = (2 ** torch.arange(self.n_freq) * np.pi).view(1, 1, 1, -1)
        const = const.to(x)

        # Out shape : (bs, npoints, out_feat)
        cos_feat = torch.cos(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        sin_feat = torch.sin(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        out = torch.cat(
            [sin_feat, cos_feat], dim=-1).view(
            bs, npts, 2 * self.inp_feat * self.n_freq)
        const_norm = torch.cat(
            [const, const], dim=-1).view(
            1, 1, 1, self.n_freq * 2).expand(
            -1, -1, self.inp_feat, -1).reshape(
            1, 1, 2 * self.inp_feat * self.n_freq)

        if self.cat_inp:
            out = torch.cat([out, x], dim=-1)
            const_norm = torch.cat(
                [const_norm, torch.ones(1, 1, self.inp_feat).to(x)], dim=-1)

            xyz_out =  out / const_norm / np.sqrt(self.n_freq * 2 + 1)
        else:

            xyz_out = out / const_norm / np.sqrt(self.n_freq * 2)

        #xyz_out = torch.cat((xy_out, z), dim=-1)
        return xyz_out

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.LayerNorm(out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features ,
                                            1 / self.in_features )
                #self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                #                            np.sqrt(6 / self.in_features) / self.omega_0)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        #print(input.shape)
        #print(self.in_features)
        #print(self.out_features)
        x = torch.sin(self.omega_0 * self.linear(input))
        #x = F.dropout(x, p=0.05, training=self.training)

        return x #/self.omega_0

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
'''

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, composer, bias=True,
                 is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = torch.tensor(1.)#0omega_0
        self.is_first = is_first
        #self.composer = composer
        self.composer = nn.Parameter(data=torch.tensor([0, 2, 4]).float(), requires_grad=True)
        self.weights = nn.Parameter(data=torch.tensor([1, 0.1, 0.01]).float(), requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / torch.exp(self.omega_0),
                                            np.sqrt(6 / self.in_features) / torch.exp(self.omega_0)),

    def forward(self, input):
        #print(input.shape)
        #print(self.in_features)
        #print(self.out_features)
        #print(self.composer)
        subwaves = torch.sin(self.linear(input)[:, :, :, None] * torch.exp(self.composer)[None, None, None, :])
        wave = torch.sum(subwaves*self.weights[None, None, None, :], dim=-1)
        return wave
'''

'''
class Siren(nn.Module):

    def __init__(self,
                 attibutes,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 device,
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()
        self.net_parts = {}

        for ith_attri in attibutes:
            self.net_parts[ith_attri] = \
                NeuralProcessImplicitHypernet(
                         in_features,
                         out_features,
                         attribute_dim=1,
                         hidden_features=hidden_features,
                         hidden_layers=hidden_layers,
                         encoder_nl='sine',
                device=device)

    def forward(self, coords, attributes):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        y=0.

        hypo_params = []
        for ith_key in attributes.keys():
            y = y + self.net_parts[ith_key](coords, attributes[ith_key][:, None].float())['model_out']
            coords = self.net_parts[ith_key](coords, attributes[ith_key][:, None].float())['model_in']
            hypo_params += list(self.net_parts[ith_key].parameters())
        model_output = {'model_in':coords,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params}
        return model_output#, coords
'''

'''
class Siren_per_attri(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.in_features = in_features + 1
        self.net = []
        self.net.append(SineLayer(self.in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        #print(attri.shape)
        #attri = attri.repeat(1, coords.shape[1]).float()[:, :, None]
        #print(coords.shape)
        #print(attri.shape)
        #coords = torch.cat((coords, attri), dim=-1)
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output#, coords

    def forward_with_activations(self, coords, retain_grad=False):

        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

'''

class BaseDeepSDFSiren(nn.Module):
    def __init__(self,
                 in_features,
                 latent_size,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 latent_in=[4],
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 zero_init_last_layer=False):
        super().__init__()

        self.in_features = in_features
        self.net = []
        self.latent_in = latent_in
        self.net.append(SineLayer(in_features + latent_size, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            if i + 1 in self.latent_in:
                self.net.append(SineLayer(hidden_features, hidden_features - in_features,
                                      is_first=False, omega_0=hidden_omega_0))#, composer=self.composer))
            else:
                self.net.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))  # , composer=self.composer))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            #with torch.no_grad():
            #    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
            #                                 np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
             self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nn.Tanh()))


        self.net = nn.Sequential(*self.net)
        print(self.net)
        '''
        if zero_init_last_layer:
            if outermost_linear:
                torch.nn.init.constant_(self.net[-1].weight, 0)
                torch.nn.init.constant_(self.net[-1].bias, 0)
            else:
                torch.nn.init.constant_(self.net[-2].weight, 0)
                torch.nn.init.constant_(self.net[-2].bias, 0)
        else:
            if outermost_linear:
                nn.utils.spectral_norm(self.net[-1])
            else:
                nn.utils.spectral_norm(self.net[-1][0])
        '''

    def forward(self, embedding, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        model_input = torch.cat((coords, embedding), dim=-1)

        for net_i in range(len(self.net)-1):
            output = self.net[net_i](model_input)
            if net_i in self.latent_in:
                model_input = torch.cat((coords, output), dim=-1)
            else:
                model_input = output

        output = self.net[-1](model_input)
        return output#, coords


class BaseDeepVFSiren(nn.Module):
    def __init__(self,
                 in_features,
                 latent_size,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 latent_in=[4],
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 zero_init_last_layer=False):
        super().__init__()

        self.in_features = in_features
        self.net = []
        self.latent_in = latent_in
        self.net.append(SineLayer(in_features + latent_size, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            if i + 1 in self.latent_in:
                self.net.append(SineLayer(hidden_features, hidden_features - in_features,
                                          is_first=False, omega_0=hidden_omega_0))  # , composer=self.composer))
            else:
                self.net.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))  # , composer=self.composer))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            # with torch.no_grad():
            #    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
            #                                 np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nn.Tanh()))

        self.net = nn.Sequential(*self.net)
        print(self.net)

    def forward(self, t, model_input):

        coords = model_input[:,:, 0:3]
        embedding = model_input[:,:, 3::]

        # coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        model_input = torch.cat((coords, embedding), dim=-1)
        model_input_0_padding = torch.cat((coords, torch.zeros_like(embedding)), dim=-1)

        for net_i in range(len(self.net) - 1):
            output = self.net[net_i](model_input)
            if net_i in self.latent_in:
                model_input = torch.cat((coords, output), dim=-1)
            else:
                model_input = output

        output_embd = self.net[-1](model_input)



        for net_i in range(len(self.net) - 1):
            output = self.net[net_i](model_input_0_padding)
            if net_i in self.latent_in:
                model_input_0_padding = torch.cat((coords, output), dim=-1)
            else:
                model_input_0_padding = output

        output_0_padding = self.net[-1](model_input_0_padding)
        output = output_embd - output_0_padding
        output = torch.cat((output,torch.zeros_like(embedding)), dim=-1)

        return output  # , coords

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, composer, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., zero_init_last_layer=False):
        super().__init__()

        self.in_features = in_features
        self.net = []
        #self.composer = composer
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=False, omega_0=first_omega_0))#, composer=self.composer))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))#, composer=self.composer))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            self.net.append(final_linear)
        else:
            #self.net.append(SineLayer(hidden_features, out_features,
            #                          is_first=False, omega_0=hidden_omega_0))
             self.net.append(nn.Linear(hidden_features, out_features))
             self.net.append(nn.Tanh())

        self.net = nn.Sequential(*self.net)
        '''
        if zero_init_last_layer:
            if outermost_linear:
                torch.nn.init.constant_(self.net[-1].weight, 0)
                torch.nn.init.constant_(self.net[-1].bias, 0)
            else:
                torch.nn.init.constant_(self.net[-2].weight, 0)
                torch.nn.init.constant_(self.net[-2].bias, 0)
        else:
            if outermost_linear:
                nn.utils.spectral_norm(self.net[-1])
            else:
                nn.utils.spectral_norm(self.net[-2])
        '''



    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input


        output = self.net(coords)
        return output#, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

'''
class FCBlock(MetaModule):
    #A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    #Can be used just as a normal neural network though, as well.
    

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=self.get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        #Returns not only model output, but also intermediate activations.
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = self.get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=self.get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations
'''
class BaseFCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        #self.first_layer_init = None
        self.in_features = in_features
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(alpha=0.2,inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nn.Tanh(),
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)


    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output#, coords

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = self.get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                #if isinstance(sublayer, BatchLinear):
                #    x = sublayer(x, params=self.get_subdict(subdict, '%d' % j))
                #else:
                x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations

class BaseDeepSDF(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self,
                 in_features,
                 latent_size,
                 out_features,
                 num_hidden_layers,
                 hidden_features,
                 latent_in=[4],
                 outermost_linear=False,
                 nonlinearity='relu',
                 weight_init=None):
        super().__init__()

        #self.first_layer_init = None
        self.in_features = in_features
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(alpha=0.2,inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
        bn = nn.LayerNorm(hidden_features)
        bn_ = nn.LayerNorm(hidden_features-in_features)
        dp = nn.Dropout(p=0.2)
        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.latent_in = latent_in
        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features + latent_size, hidden_features), bn, nl, dp
        ))

        for i in range(num_hidden_layers):
            if i+1 in self.latent_in:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features, hidden_features - in_features), bn_, nl, dp
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features, hidden_features), bn, nl, dp
                ))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features,  out_features)))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nn.Tanh(),
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)


    def forward(self, embedding, coords):
        model_input = torch.cat((coords, embedding), dim=-1)
        for net_i in range(len(self.net)-1):
            output = self.net[net_i](model_input)
            if net_i in self.latent_in:
                model_input = torch.cat((coords, output), dim=-1)
            else:
                model_input = output
        output = self.net[-1](model_input)
        return output#, coords

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = self.get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                #if isinstance(sublayer, BatchLinear):
                #    x = sublayer(x, params=self.get_subdict(subdict, '%d' % j))
                #else:
                x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations





class ICFCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        #self.first_layer_init = None
        self.in_features = in_features
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nn.Tanh(),
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)


    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output#, coords

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = self.get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                #if isinstance(sublayer, BatchLinear):
                #    x = sublayer(x, params=self.get_subdict(subdict, '%d' % j))
                #else:
                x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations






class LipLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, nonlinearity='relu'):
        super().__init__()
        self.nonlinearity = nonlinearity

        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}
        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
        self.nl = nl

        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias, requires_grad=True)

        # initialize weights and biases
        if nonlinearity == 'sine':
            self.omega_0 = 30
            self.init_weights()
        else:
            nn.init.kaiming_normal_(self.weights, a=0.0, nonlinearity=nonlinearity, mode='fan_in')
        nn.init.zeros_(self.bias)  # bias init
        # initializing c
        #c= torch.Tensor(size_out, size_in)
        self.c = torch.nn.Parameter(data=torch.max(torch.sum(torch.abs(self.weights), dim=1)), requires_grad=True)


    def init_weights(self):
        with torch.no_grad():
            self.weights.uniform_(-np.sqrt(6 / self.size_in) / self.omega_0,
                                            np.sqrt(6 / self.size_in) / self.omega_0)

    def forward(self, input):
        x = input[0]
        loss = input[1]
        self.weights.data = self.weight_normalization(self.weights.data, self.softplus(self.c.data))
        y = self.nl(torch.add(torch.matmul(x[:,None,  :, :], self.weights.t()[None, :, :])[:, 0, :, :], self.bias))
        loss *= self.get_lipschitz_loss()
        return y, loss

    def softplus(self, c):
        y = torch.log(torch.tensor(1.0) + torch.exp(c))
        #y = torch.logaddexp(c, torch.tensor(0).to(c.device))
        return y

    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(torch.tensor(1.0).to(W.device), softplus_c / absrowsum)

        return W * scale[:, None]

    def get_lipschitz_loss(self):
        """
        This function computes the Lipschitz regularization
        """
        loss_lip = self.softplus(self.c)
        return loss_lip

    def normalize_params(self, ):
        """
        (Optional) After training, this function will clip network [W, b] based on learned lipschitz constants. Thus, one can use normal MLP forward pass during test time, which is a little bit faster.
        """
        W = self.weight_normalization(self.weights, self.softplus(self.c))
        return W

    def forward_eval_single(self, x):
        """
        (Optional) this is a standard forward pass of a mlp. This is useful to speed up the performance during test time
        """
        # concatenate coordinate and latent code
        y = self.nl(torch.matmul(self.weights, x) + self.bias)
        return y



class ExuLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, nonlinearity='relu'):
        super().__init__()
        self.nonlinearity = nonlinearity

        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}
        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
        self.nl = nl

        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias, requires_grad=True)

        # initialize weights and biases
        if nonlinearity == 'sine':
            self.omega_0 = 30
            self.init_weights()
        else:
            nn.init.kaiming_normal_(self.weights, a=0.0, nonlinearity=nonlinearity, mode='fan_in')
        nn.init.zeros_(self.bias)  # bias init
        # initializing c
        #c= torch.Tensor(size_out, size_in)
        self.c = torch.nn.Parameter(data=torch.max(torch.sum(torch.abs(self.weights), dim=1)), requires_grad=True)


    def init_weights(self):
        with torch.no_grad():
            self.weights.uniform_(-np.sqrt(6 / self.size_in) / self.omega_0,
                                            np.sqrt(6 / self.size_in) / self.omega_0)

    def forward(self, input):
        x = input[0]
        loss = input[1]
        self.weights.data = self.weight_normalization(self.weights.data, self.softplus(self.c.data))
        y = self.nl(torch.add(torch.matmul(x[:,None,  :, :], self.weights.t()[None, :, :])[:, 0, :, :], self.bias))
        loss *= self.get_lipschitz_loss()
        return y, loss

    def softplus(self, c):
        y = torch.log(torch.tensor(1.0) + torch.exp(c))
        #y = torch.logaddexp(c, torch.tensor(0).to(c.device))
        return y

    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(torch.tensor(1.0).to(W.device), softplus_c / absrowsum)

        return W * scale[:, None]

    def get_lipschitz_loss(self):
        """
        This function computes the Lipschitz regularization
        """
        loss_lip = self.softplus(self.c)
        return loss_lip

    def normalize_params(self, ):
        """
        (Optional) After training, this function will clip network [W, b] based on learned lipschitz constants. Thus, one can use normal MLP forward pass during test time, which is a little bit faster.
        """
        W = self.weight_normalization(self.weights, self.softplus(self.c))
        return W

    def forward_eval_single(self, x):
        """
        (Optional) this is a standard forward pass of a mlp. This is useful to speed up the performance during test time
        """
        # concatenate coordinate and latent code
        y = self.nl(torch.matmul(self.weights, x) + self.bias)
        return y



class LipMLPBlock(nn.Module):
    #def __init__(self, hyperParams):
    #    self.hyperParams = hyperParams

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False):
        super().__init__()

        #self.first_layer_init = None
        self.in_features = in_features
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        self.net = []
        self.net.append(nn.Sequential(
            LipLinearLayer(in_features, hidden_features, nonlinearity='relu')
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                LipLinearLayer(hidden_features, hidden_features, nonlinearity='relu')
            ))

        if outermost_linear:
            self.net.append(LipLinearLayer(hidden_features, out_features))
        else:
            self.net.append(nn.Sequential(LipLinearLayer(hidden_features, out_features, nonlinearity='tanh')))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output, loss_lip = self.net(coords)
        return output, loss_lip#, coords



'''Modules for hypernetwork experiments, Paper Sec. 4.4
'''


'''
class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):

        #Args:
        #    hyper_in_features: In features of hypernetwork
        #    hyper_hidden_layers: Number of hidden layers in hypernetwork
        #    hyper_hidden_features: Number of hidden units in hypernetwork
        #    hypo_module: MetaModule. The module whose parameters are predicted.
        
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                                 outermost_linear=True, nonlinearity='sine')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        #
        #Args:
        #    z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)#

        #Returns:
        #    params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        #
        #print(z.shape)
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class NeuralProcessImplicitHypernet(nn.Module):
    #A canonical representation hypernetwork mapping 2D coords to out_features.
    def __init__(self,
                 in_features,
                 out_features,
                 attribute_dim,
                 hidden_features,
                 hidden_layers,
                 encoder_nl='sine',
                 device='cpu'):
        super().__init__()

        #latent_dim = 1
        #self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
        #                                    in_features=2)
        self.hypo_net =  FCBlock(in_features, out_features, num_hidden_layers=3, hidden_features=hidden_features, outermost_linear=False, nonlinearity='sine') #Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,first_omega_0=30, hidden_omega_0=30.).to(device)

            #Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,first_omega_0=30, hidden_omega_0=30.).to(device)
        #FCBlock(in_features, out_features, num_hidden_layers, hidden_features,
         #        outermost_linear=False, nonlinearity='sine', weight_init=None) #Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,first_omega_0=30, hidden_omega_0=30.).to(device)
        self.hyper_net = HyperNetwork(hyper_in_features=attribute_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net).to(device)
        #self.set_encoder = modules.SetEncoder(in_features=in_features, out_features=latent_dim, num_hidden_layers=2,
        #                                      hidden_features=latent_dim, nonlinearity=encoder_nl)
        print(self)

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def get_hypo_net_weights(self, model_input):
        #pixels, coords = model_input['img_sub'], model_input['coords_sub']
        #ctxt_mask = model_input.get('ctxt_mask', None)
        #embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        hypo_params = self.hyper_net(model_input)
        return hypo_params, model_input

    def forward(self, model_input, attri):
        hypo_params = self.hyper_net(attri)
        model_output = self.hypo_net(model_input, hypo_params)
        return {'model_in':model_input,
                'model_out':model_output,
                'latent_vec':attri,
                'hypo_params':hypo_params}
'''



############################
# Initialization schemes
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
    y = x.clone()
    y[..., 1::2] = -1 * y[..., 1::2]
    return y


def compl_div(x, y):
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def compl_mul(x, y):
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


class NAIVF_withtempl(nn.Module):
    def __init__(self,
                 template_attributes,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 device,
                 backbone='mlp',
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 pos_enc=True):
        super().__init__()
        self.net_parts = nn.ModuleDict({})
        print(template_attributes)
        self.template_attributes = template_attributes
        self.device = device
        self.composer = torch.nn.Parameter(data=torch.tensor(10.0).float(), requires_grad=True)
        #self.net_parts['initial'] = Siren(in_features,
        #                                  hidden_features,
        #                                  hidden_layers,
        #                                  out_features,
        #                                  outermost_linear=True,
        #                                  first_omega_0=30,
        #                                 hidden_omega_0=30.).to(device)

        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=5, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features

        if backbone == 'mlp':
            self.net_parts['initial'] = BaseFCBlock(in_features,
                                                    out_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=False,
                                                    nonlinearity='relu',
                                                    weight_init=None).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(in_features,
                                              hidden_features,
                                              hidden_layers,
                                              out_features,
                                              self.composer,
                                              outermost_linear=False,
                                              first_omega_0=5,
                                             hidden_omega_0=5.).to(device)


        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)
            if backbone == 'mlp':
                self.net_parts[ith_attri] = BaseFCBlock(self.in_features + 1,
                                                        in_features,
                                                        hidden_layers,
                                                        hidden_features,
                                                        outermost_linear=True,
                                                        nonlinearity='relu',
                                                        weight_init=None).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(self.in_features+1,
                                                hidden_features,
                                                hidden_layers,
                                                in_features,
                                                  self.composer,
                                                outermost_linear=True,
                                                first_omega_0=5,
                                                hidden_omega_0=5.).to(device)

        if backbone == 'siren':
            self.aggregator = Siren(hidden_features,
                                    hidden_features,
                                    hidden_layers,
                                    in_features,
                                    self.composer,
                                    outermost_linear=True,
                                    first_omega_0=5,
                                    hidden_omega_0=5.).to(device)
        elif backbone == 'mlp':
            self.aggregator = BaseFCBlock(in_features + 1,
                                                    hidden_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=False,
                                                    nonlinearity='relu',
                                                    weight_init=None).to(device)


    def forward(self, coords, attributes):
        coords_init = coords.clone().detach().requires_grad_(True)
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        '''
        get input array of the network
        '''

        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init

        '''
        get input array of the network
        '''
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        input = torch.cat([enc_coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)

        '''
        template shape
        '''
        #y_coords = input.index_select(-1, torch.arange(coords.shape[-1]).to(input.device))
        y_init = self.net_parts['initial'](coords_init)

        model_map = {'initial': y_init}
        vec_fields = {'overall': torch.zeros_like(coords).to(self.device)}
        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())

        '''
        fit shape
        '''
        y_displacement = torch.zeros_like(coords).to(self.device)
        for ith_key in range(len(self.attributes)):
            '''
            input of current vector field
            '''
            slt_index =torch.tensor(list(range(enc_coords.shape[-1])) + [ith_key + enc_coords.shape[-1]]).to(input.device)
            '''
            current vector field
            '''
            current_delta_coords = self.net_parts[self.attributes[ith_key]](input.index_select(-1, slt_index))#[:, :, slt_index])
            vec_fields[self.attributes[ith_key]] = current_delta_coords
            '''
            overall vector field
            '''
            vec_fields['overall'] += current_delta_coords
            '''
            current deformation
            '''
            y_displacement += current_delta_coords
            '''
            input of individual evolution
            '''
            coords_current = coords_init - current_delta_coords
            '''
            individual evolution
            '''
            model_map[self.attributes[ith_key]] = self.net_parts['initial'](coords_current)
            '''
            regularize for template
            '''
            current_template_vec = input.index_select(-1, slt_index).clone()
            current_template_vec[:, :, -1] = self.template_attributes[self.attributes[ith_key]]
            vec_fields[self.attributes[ith_key] + '_templ'] = self.net_parts[self.attributes[ith_key]](current_template_vec)
            #model_map[self.attributes[ith_key] + '_velocity_field'] = current_delta_coords.clone()
            hypo_params += list(self.net_parts[self.attributes[ith_key]].parameters())

        '''
        final output of the implicit functino
        '''
        y = self.net_parts['initial'](coords_init - y_displacement)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))
        '''
        gradient map
        '''
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':coords_init,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords