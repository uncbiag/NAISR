import torch
from torch import nn
from collections import OrderedDict
from naisr.modules import SineLayer, Siren, BaseFCBlock, LipMLPBlock, LipBoundedPosEnc, BaseDeepSDF, BaseDeepSDFSiren #,HyperNetwork, NeuralProcessImplicitHypernet
import numpy as np

import naisr.diff_operators as diff_operators
import torch.nn.functional as F

'''
class NAISR(nn.Module):
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
        print(attibutes)
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

class NAISiren(nn.Module):
    def __init__(self,
                 attributes,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 device,
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()
        self.net_parts = nn.ModuleDict({})
        print(attributes)
        self.attributes = attributes
        self.composer = torch.nn.Parameter(data=torch.tensor(10.0).float(), requires_grad=True)
        #self.net_parts['initial'] = Siren(in_features,
        #                                  hidden_features,
        #                                  hidden_layers,
        #                                  out_features,
        #                                 outermost_linear=True,
        #                                 first_omega_0=30,
        #                                 hidden_omega_0=30.).to(device)
        self.net_parts['initial'] = Siren(in_features,
                                              hidden_features,
                                              hidden_layers,
                                              out_features,
                                              self.composer,
                                              outermost_linear=False,
                                              first_omega_0=5,
                                             hidden_omega_0=5.).to(device)
        for ith_attri in attributes:
            #self.net_parts[ith_attri] = Siren_per_attri(in_features,
            #                                            hidden_features,
            #                                            hidden_layers,
            #                                            out_features,
            #                                            outermost_linear=True,
            #                                            first_omega_0=30,
            #                                            hidden_omega_0=30.).to(device)
            self.net_parts[ith_attri] = Siren(in_features+1,
                                              hidden_features,
                                              hidden_layers,
                                              out_features,
                                              self.composer,
                                              outermost_linear=False,
                                              first_omega_0=5,
                                             hidden_omega_0=5.).to(device)
    def forward(self, coords, attributes):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(coords.shape[1], 1).float()
        input = torch.cat([coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)


        y = self.net_parts['initial'](input[..., 0:coords.shape[-1]])
        y_init = y.clone()
        model_map = {'initial': y_init}
        #y_map = {}
        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())
        for ith_key in range(len(self.attributes)):
            slt_index = list(range(coords.shape[-1])) + [ith_key + coords.shape[-1]]
            y_current = self.net_parts[self.attributes[ith_key]](input[..., slt_index])#attributes[ith_key].float())

            y = y + y_current #self.net_parts[ith_key](coords, attributes[ith_key].float())
            model_map[self.attributes[ith_key]] = y_current.clone()

            '''
            grad
            
            gradient = diff_operators.gradient(y_current, input)[:, :,[ith_key+coords.shape[-1]]]
            y_map[self.attributes[ith_key] + '_grad'] = gradient
            
            '''
            hypo_params += list(self.net_parts[self.attributes[ith_key]].parameters())
        #y_map_overall = torch.ones_like(y_map['initial'])
        #for ith_key in range(len(self.attributes)):
        #    y_map_overall = y_map_overall + y_map[self.attributes[ith_key]]
        #y_map['leftover'] = 1 / (y_map_overall + 1e-5)
        #for ith_key in range(len(self.attributes)):
        #    y_map[self.attributes[ith_key]]  = y_map[self.attributes[ith_key]]  / (y_map_overall + 1e-5)

        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[...,[i_attri+coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient


        model_output = \
            {   'all_input': input,
                'model_in':coords,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords




class NAIVF_with3dtempl(nn.Module):
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
        '''
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
                                              first_omega_0=30.,
                                             hidden_omega_0=30.).to(device)
        '''
        self.net_parts['initial'] = Siren(in_features=in_features,
                                          hidden_features=512,
                                          hidden_layers=3,
                                          out_features=out_features,
                                          composer=self.composer,
                                          outermost_linear=False,
                                          first_omega_0=30.,
                                          hidden_omega_0=30.).to(device)

        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)

            if backbone == 'mlp':
                self.net_parts[ith_attri] = BaseFCBlock(self.in_features + 1,
                                                        in_features ,
                                                        hidden_layers,
                                                        hidden_features,
                                                        outermost_linear=False,
                                                        nonlinearity='relu',
                                                        weight_init=None).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(self.in_features+1,
                                                hidden_features,
                                                hidden_layers,
                                                in_features,
                                                  self.composer,
                                                outermost_linear=True,
                                                first_omega_0=10.,
                                                hidden_omega_0=10.).to(device)

        if backbone == 'siren':
            self.aggregator = Siren(hidden_features,
                                    hidden_features,
                                    hidden_layers,
                                    in_features,
                                    self.composer,
                                    outermost_linear=True,
                                    first_omega_0=10.,
                                    hidden_omega_0=10.).to(device)
        elif backbone == 'mlp':
            self.aggregator = BaseFCBlock(in_features + 1,
                                                    hidden_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=False,
                                                    nonlinearity='relu',
                                                    weight_init=None).to(device)


           #  Siren(1,
           # 8,
           # 1,
           # 8-2,
           # self.composer,
           # outermost_linear=True,
           # first_omega_0=30.,
           # hidden_omega_0=30.).to(device)
            #BaseFCBlock(1,
            #            hidden_features - 2,
            #            hidden_layers,
            #            hidden_features,
            #            outermost_linear=False,
            #            nonlinearity='relu',
            #            weight_init=None).to(device)
            #Siren(1,hidden_features,
            #    hidden_layers,
            #    hidden_features-2,
            #    self.composer,
            #    outermost_linear=True,
            #    first_omega_0=10.,
            #    hidden_omega_0=10.).to(device)


        '''
        Siren(1,hidden_features,
                hidden_layers,
                hidden_features-2,
                self.composer,
                outermost_linear=True,
                first_omega_0=10.,
                hidden_omega_0=10.).to(device)
        '''



    def forward(self, coords, attributes, template_coords=None):
        coords_init = coords.clone().detach().requires_grad_(True)
        template_coords_init = template_coords.clone().detach().requires_grad_(True)        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        '''
        get input array of the network
        '''

        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
            template_enc_coords = self.pos_encoder(template_coords_init)
        else:
            enc_coords = coords_init
            template_enc_coords = template_coords_init

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
        y_init = self.net_parts['initial'](template_coords_init)

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
            #current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
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
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':coords_init,
                'template_in': template_coords_init,
                'coords_with_disp': coords_init - y_displacement,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords


class DeepSDF(nn.Module):
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
                 pos_enc=False,
                 latent_size=256):
        super().__init__()

        self.template_attributes = template_attributes
        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
        self.device = device
        self.composer = torch.nn.Parameter(data=torch.tensor(5.0).float(), requires_grad=True)

        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=5, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features
        if backbone == 'mlp':
            self.net = BaseDeepSDF(self.in_features,
                                    latent_size=latent_size,
                                    num_hidden_layers = hidden_layers,
                                    hidden_features=hidden_features,
                                    out_features=out_features,
                                    latent_in=[hidden_layers//2],
                                    outermost_linear=False,
                                    nonlinearity='relu',
                                    weight_init=None).to(device)

        elif backbone == 'siren':
            self.net = BaseDeepSDFSiren(
                              self.in_features,
                              latent_size=latent_size,
                              hidden_features=hidden_features,
                              hidden_layers=hidden_layers,
                              out_features=out_features,
                              latent_in = [hidden_layers//2],
                              outermost_linear=False,
                              first_omega_0=30.,
                              hidden_omega_0=30.).to(device)

    def forward(self, coords, attributes, embedding):
        coords_init = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input\
        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init

        output = self.net(embedding, enc_coords)
        #get map

        model_output = {'model_in': coords_init,
                        'all_input': coords_init,
                        'embedding_in': embedding,
                        'model_out':output,}

        return model_output#, coords



'''

class DeepSDF(nn.Module):
    def __init__(
        self,
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
            pos_enc=False,
            latent_size=256,

    ):
        super(DeepSDF, self).__init__()

        def make_sequence():
            return []
        self.device = device
        dims = [latent_size + 3] + [ 512, 512, 512, 512, 512, 512, 512, 512 ] + [1]

        self.num_layers = len(dims)
        self.norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
        self.latent_in = [4]
        self.latent_dropout = False
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = False
        self.weight_norm = True

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if self.weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not self.weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = True
        if self.use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = 0.2
        self.dropout = [0, 1, 2, 3, 4, 5, 6, 7]
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, coords, attributes, embedding):
        coords_init = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        input = torch.cat(( embedding, coords_init), dim=-1)
        xyz = coords_init
        #print(xyz.shape)

        if input.shape[-1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], -1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], -1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        #if hasattr(self, "th"):
        #    x = self.th(x)

        #return x
        model_output = {'model_in': coords_init,
                        'all_input': coords_init,
                        'embedding_in': embedding,
                        'model_out':x,}

        return model_output#, coords
'''

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


        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=5, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features

        if backbone == 'mlp':
            self.net_parts['initial'] = BaseFCBlock(8,
                                                    out_features,
                                                    6,
                                                    hidden_features,
                                                    outermost_linear=True,
                                                    nonlinearity='relu',
                                                   weight_init=None).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(8,
                                              hidden_features,
                                              1,
                                              out_features,
                                              self.composer,
                                              outermost_linear=True,
                                              first_omega_0=30.,
                                             hidden_omega_0=30.).to(device)


        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)
            if backbone == 'mlp':
                self.net_parts[ith_attri] = BaseFCBlock(self.in_features + 1,
                                                        in_features -1,
                                                        hidden_layers,
                                                        hidden_features,
                                                        outermost_linear=True,
                                                        nonlinearity='relu',
                                                        weight_init=None).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(self.in_features+1,
                                                hidden_features,
                                                hidden_layers,
                                                in_features-1,
                                                  self.composer,
                                                outermost_linear=True,
                                                first_omega_0=30.,
                                                hidden_omega_0=30.).to(device)

        if backbone == 'siren':
            self.aggregator = Siren(hidden_features,
                                    hidden_features,
                                    hidden_layers,
                                    in_features,
                                    self.composer,
                                    outermost_linear=True,
                                    first_omega_0=10.,
                                    hidden_omega_0=10.).to(device)
        elif backbone == 'mlp':
            self.aggregator = BaseFCBlock(in_features + 1,
                                                    hidden_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=False,
                                                    nonlinearity='relu',
                                                    weight_init=None).to(device)

        if backbone == 'siren':
            self.template_encoder =\
            Siren(1,
            hidden_features,
            hidden_layers,
            8-2,
            self.composer,
            outermost_linear=True,
            first_omega_0=30.,
            hidden_omega_0=30.).to(device)

        elif backbone == 'mlp':
            self.template_encoder = \
                BaseFCBlock(1,
                        8-2 ,
                        hidden_layers,
                        hidden_features,
                        outermost_linear=False,
                        nonlinearity='relu',
                        weight_init=None).to(device)


    def implicit_template(self, coords):
        encoded_template = self.template_encoder(coords.index_select(-1, torch.tensor([self.in_features-1,]).to(coords.device)))
        encoded_template_input = torch.cat((encoded_template, coords.index_select(-1, torch.tensor(list(range(self.in_features-1))).to(coords.device))), dim=-1)
        y = self.net_parts['initial'](encoded_template_input)
        return y

    def infer(self, coords, attributes):
        y_displacement = torch.zeros_like(coords).to(self.device)

        for ith_name, ith_data in attributes.items():
            current_attribute = ith_data[:, None]
            current_attribute = current_attribute.repeat(1, coords.shape[1], 1).float()
            current_coords_with_attri = torch.cat((coords, current_attribute), dim=-1).clone().detach().requires_grad_(True)
            '''
            current vector field
            '''
            current_delta_coords = self.net_parts[ith_name](current_coords_with_attri)#[:, :, slt_index])
            current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
            y_displacement += current_delta_coords
        y = self.implicit_template(coords - y_displacement)
        return y

    def template(self, coords):
        y = self.implicit_template(coords)
        return y

    def forward(self, coords, attributes, template_coords=None):
        coords_init = coords.clone().detach().requires_grad_(True)
        template_coords_init = template_coords.clone().detach().requires_grad_(True)        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        '''
        get input array of the network
        '''

        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
            template_enc_coords = self.pos_encoder(template_coords_init)
        else:
            enc_coords = coords_init
            template_enc_coords = template_coords_init

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
        y_init = self.implicit_template(template_enc_coords)

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
            current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
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
            if self.pos_enc:
                coords_current = self.pos_encoder(coords_current)
            model_map[self.attributes[ith_key]] = self.implicit_template(coords_current)
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
        y_displaced = coords_init - y_displacement
        if self.pos_enc:
            y_displaced = self.pos_encoder(y_displaced)
        y = self.implicit_template(y_displaced)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))
        '''
        gradient map
        '''
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':coords_init,
                'template_in': template_coords_init,
                'coords_with_disp': coords_init - y_displacement,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords




class NAIVF_fixedtempl(nn.Module):
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


        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=5, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features

        if backbone == 'mlp':
            self.net_parts['initial'] = BaseFCBlock(self.in_features -1 + 8, #8
                                                    out_features,
                                                    6,
                                                    hidden_features,
                                                    outermost_linear=True,
                                                    nonlinearity='relu',
                                                   weight_init=None).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(self.in_features -1 + 8, #8
                                              hidden_features,
                                              1,
                                              out_features,
                                              self.composer,
                                              outermost_linear=True,
                                              first_omega_0=30.,
                                             hidden_omega_0=30.).to(device)


        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)
            if backbone == 'mlp':
                self.net_parts[ith_attri] = BaseFCBlock(self.in_features + 1,
                                                        in_features -1,
                                                        hidden_layers,
                                                        hidden_features,
                                                        outermost_linear=True,
                                                        nonlinearity='relu',
                                                        weight_init=None).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(self.in_features+1,
                                                hidden_features,
                                                hidden_layers,
                                                in_features-1,
                                                  self.composer,
                                                outermost_linear=True,
                                                first_omega_0=30.,
                                                hidden_omega_0=30.).to(device)

        if backbone == 'siren':
            self.aggregator = Siren(hidden_features,
                                    hidden_features,
                                    hidden_layers,
                                    in_features,
                                    self.composer,
                                    outermost_linear=True,
                                    first_omega_0=10.,
                                    hidden_omega_0=10.).to(device)
        elif backbone == 'mlp':
            self.aggregator = BaseFCBlock(in_features + 1,
                                                    hidden_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=False,
                                                    nonlinearity='relu',
                                                    weight_init=None).to(device)

        if backbone == 'siren':
            self.template_encoder =\
            Siren(1,
            hidden_features,
            hidden_layers,
            8,
            self.composer,
            outermost_linear=True,
            first_omega_0=30.,
            hidden_omega_0=30.).to(device)

        elif backbone == 'mlp':
            self.template_encoder = \
                BaseFCBlock(1,
                        8 ,
                        hidden_layers,
                        hidden_features,
                        outermost_linear=False,
                        nonlinearity='relu',
                        weight_init=None).to(device)


    def implicit_template(self, coords):
        encoded_template = self.template_encoder(coords.index_select(-1, torch.tensor([self.in_features-1,]).to(coords.device)))
        encoded_template_input = torch.cat((encoded_template, coords.index_select(-1, torch.tensor(list(range(self.in_features-1))).to(coords.device))), dim=-1)
        y = self.net_parts['initial'](encoded_template_input)
        return y

    def infer(self, coords, attributes):
        y_displacement = torch.zeros_like(coords).to(self.device)

        for ith_name, ith_data in attributes.items():
            current_attribute = ith_data[:, None]
            current_attribute = current_attribute.repeat(1, coords.shape[1], 1).float()
            current_coords_with_attri = torch.cat((coords, current_attribute), dim=-1).clone().detach().requires_grad_(True)
            '''
            current vector field
            '''
            current_delta_coords = self.net_parts[ith_name](current_coords_with_attri)#[:, :, slt_index])
            current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
            y_displacement += current_delta_coords
        y = self.implicit_template(coords - y_displacement)
        return y

    def template(self, coords):
        y = self.implicit_template(coords)
        return y



    def forward(self, coords, attributes, template_coords=None):
        coords_init = coords.clone().detach().requires_grad_(True)
        template_coords_init = template_coords.clone().detach().requires_grad_(True)        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        '''
        get input array of the network
        '''

        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
            template_enc_coords = self.pos_encoder(template_coords_init)
        else:
            enc_coords = coords_init
            template_enc_coords = template_coords_init

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
        y_init = self.implicit_template(template_enc_coords)

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
            current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
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
            if self.pos_enc:
                coords_current = self.pos_encoder(coords_current)
            model_map[self.attributes[ith_key]] = self.implicit_template(coords_current)
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
        y_displaced = coords_init - y_displacement
        if self.pos_enc:
            y_displaced = self.pos_encoder(y_displaced)
        y = self.implicit_template(y_displaced)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))
        '''
        gradient map
        '''
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':coords_init,
                'template_in': template_coords_init,
                'coords_with_disp': coords_init - y_displacement,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords






class NAIVF_fixed(nn.Module):
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


        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=5, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features

        if backbone == 'mlp':
            self.net_parts['initial'] = BaseFCBlock(self.in_features -1 + 8, #8
                                                    out_features,
                                                    6,
                                                    hidden_features,
                                                    outermost_linear=True,
                                                    nonlinearity='relu',
                                                   weight_init=None).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(self.in_features -1 + 8, #8
                                              hidden_features,
                                              1,
                                              out_features,
                                              self.composer,
                                              outermost_linear=True,
                                              first_omega_0=30.,
                                             hidden_omega_0=30.).to(device)


        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)
            if backbone == 'mlp':
                self.net_parts[ith_attri] = BaseFCBlock(1,
                                                        8,
                                                        hidden_layers,
                                                        hidden_features,
                                                        outermost_linear=True,
                                                        nonlinearity='relu',
                                                        weight_init=None).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(1,
                                                hidden_features,
                                                hidden_layers,
                                                8,
                                                  self.composer,
                                                outermost_linear=True,
                                                first_omega_0=30.,
                                                hidden_omega_0=30.).to(device)

        if backbone == 'siren':
            self.vf_decoder = \
                Siren(8+8+2,
                      hidden_features,
                      hidden_layers,
                      2,
                      self.composer,
                      outermost_linear=True,
                      first_omega_0=30.,
                      hidden_omega_0=30.).to(device)

        elif backbone == 'mlp':
            self.vf_decoder = \
                BaseFCBlock(8+8+2,
                            2,
                            hidden_layers,
                            hidden_features,
                            outermost_linear=False,
                            nonlinearity='relu',
                            weight_init=None).to(device)

        if backbone == 'siren':
            self.template_encoder =\
            Siren(1,
            hidden_features,
            hidden_layers,
            8,
            self.composer,
            outermost_linear=True,
            first_omega_0=30.,
            hidden_omega_0=30.).to(device)

        elif backbone == 'mlp':
            self.template_encoder = \
                BaseFCBlock(1,
                        8 ,
                        hidden_layers,
                        hidden_features,
                        outermost_linear=False,
                        nonlinearity='relu',
                        weight_init=None).to(device)


    def implicit_template(self, coords):
        encoded_template = self.template_encoder(coords.index_select(-1, torch.tensor([self.in_features-1,]).to(coords.device)))
        encoded_template_input = torch.cat((encoded_template, coords.index_select(-1, torch.tensor(list(range(self.in_features-1))).to(coords.device))), dim=-1)
        y = self.net_parts['initial'](encoded_template_input)
        return y

    def infer(self, coords, attributes):
        y_displacement = torch.zeros_like(coords).to(self.device)

        for ith_name, ith_data in attributes.items():
            current_attribute = ith_data[:, None]
            current_attribute = current_attribute.repeat(1, coords.shape[1], 1).float()
            current_coords_with_attri = torch.cat((coords, current_attribute), dim=-1).clone().detach().requires_grad_(True)
            '''
            current vector field
            '''
            current_delta_coords = self.net_parts[ith_name](current_coords_with_attri)#[:, :, slt_index])
            current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
            y_displacement += current_delta_coords
        y = self.implicit_template(coords - y_displacement)
        return y

    def template(self, coords):
        y = self.implicit_template(coords)
        return y

    def vf(self, attribute_name, attribute_data, encoded_depth, xy):
        encoded_attribute = self.net_parts[attribute_name](attribute_data)
        encoded_x = torch.cat((encoded_attribute, encoded_depth, xy), dim=-1)
        y_disp = self.vf_decoder(encoded_x)
        return y_disp


    def forward(self, coords, attributes, template_coords=None):
        coords_init = coords.clone().detach().requires_grad_(True)
        template_coords_init = template_coords.clone().detach().requires_grad_(True)        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        '''
        get input array of the network
        '''

        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
            template_enc_coords = self.pos_encoder(template_coords_init)
        else:
            enc_coords = coords_init
            template_enc_coords = template_coords_init

        '''
        get input array of the network
        '''
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        input = torch.cat([enc_coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)

        arr_depth = input.index_select(-1, torch.tensor([self.in_features - 1, ]).to(coords.device))
        arr_xy = input.index_select(-1, torch.tensor(list(range(self.in_features-1))).to(coords.device))
        encoded_depth = self.template_encoder(arr_depth)

        '''
        template shape
        '''
        #y_coords = input.index_select(-1, torch.arange(coords.shape[-1]).to(input.device))
        y_init = self.implicit_template(template_enc_coords)

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
            current_delta_coords = self.vf(self.attributes[ith_key], input.index_select(-1, torch.tensor([self.in_features -1 + ith_key,]).to(self.device)), encoded_depth, arr_xy) #self.net_parts[self.attributes[ith_key]](input.index_select(-1, slt_index))#[:, :, slt_index])
            current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
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
            if self.pos_enc:
                coords_current = self.pos_encoder(coords_current)
            model_map[self.attributes[ith_key]] = self.implicit_template(coords_current)
            '''
            regularize for template
            '''
            #current_template_vec = input.index_select(-1, slt_index).clone()
            #current_template_vec[:, :, -1] = self.template_attributes[self.attributes[ith_key]]
            #vec_fields[self.attributes[ith_key] + '_templ'] = self.net_parts[self.attributes[ith_key]](current_template_vec)
            #model_map[self.attributes[ith_key] + '_velocity_field'] = current_delta_coords.clone()
            hypo_params += list(self.net_parts[self.attributes[ith_key]].parameters())

        '''
        final output of the implicit function
        '''
        y_displaced = coords_init - y_displacement
        if self.pos_enc:
            y_displaced = self.pos_encoder(y_displaced)
        y = self.implicit_template(y_displaced)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))
        '''
        gradient map
        '''
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':coords_init,
                'template_in': template_coords_init,
                'coords_with_disp': coords_init - y_displacement,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords





class NAIVF_autotempl(nn.Module):
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


        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=5, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features

        if backbone == 'mlp':
            self.net_parts['initial'] = BaseFCBlock(self.in_features -1 + 8, #8
                                                    out_features,
                                                    6,
                                                    hidden_features,
                                                    outermost_linear=True,
                                                    nonlinearity='relu',
                                                   weight_init=None).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(self.in_features -1 + 8, #8
                                              hidden_features,
                                              1,
                                              out_features,
                                              self.composer,
                                              outermost_linear=True,
                                              first_omega_0=30.,
                                             hidden_omega_0=30.).to(device)


        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)
            if backbone == 'mlp':
                self.net_parts[ith_attri] = BaseFCBlock(self.in_features + 1,
                                                        in_features -1,
                                                        hidden_layers,
                                                        hidden_features,
                                                        outermost_linear=True,
                                                        nonlinearity='relu',
                                                        weight_init=None).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(self.in_features+1,
                                                hidden_features,
                                                hidden_layers,
                                                in_features-1,
                                                  self.composer,
                                                outermost_linear=True,
                                                first_omega_0=30.,
                                                hidden_omega_0=30.).to(device)

        if backbone == 'siren':
            self.template_encoder =\
            Siren(1,
            hidden_features,
            hidden_layers,
            8,
            self.composer,
            outermost_linear=True,
            first_omega_0=30.,
            hidden_omega_0=30.).to(device)

        elif backbone == 'mlp':
            self.template_encoder = \
                BaseFCBlock(1,
                        8 ,
                        hidden_layers,
                        hidden_features,
                        outermost_linear=False,
                        nonlinearity='relu',
                        weight_init=None).to(device)


    def implicit_template(self, coords):
        encoded_template = self.template_encoder(coords.index_select(-1, torch.tensor([self.in_features-1,]).to(coords.device)))
        encoded_template_input = torch.cat((encoded_template, coords.index_select(-1, torch.tensor(list(range(self.in_features-1))).to(coords.device))), dim=-1)
        y = self.net_parts['initial'](encoded_template_input)
        return y

    def infer(self, coords, attributes):
        y_displacement = torch.zeros_like(coords).to(self.device)

        for ith_name, ith_data in attributes.items():
            current_attribute = ith_data[:, None]
            current_attribute = current_attribute.repeat(1, coords.shape[1], 1).float()
            current_coords_with_attri = torch.cat((coords, current_attribute), dim=-1).clone().detach().requires_grad_(True)
            '''
            current vector field
            '''
            current_delta_coords = self.net_parts[ith_name](current_coords_with_attri)#[:, :, slt_index])
            current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
            y_displacement += current_delta_coords
        y = self.implicit_template(coords - y_displacement)
        return y

    def forward(self, coords, attributes, template=None):
        coords_init = coords.clone().detach().requires_grad_(True)       #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
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
        y_init = self.implicit_template(enc_coords)

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
            current_delta_coords = torch.cat((current_delta_coords, torch.zeros_like(current_delta_coords[..., [0]])), dim=-1)
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
            if self.pos_enc:
                coords_current = self.pos_encoder(coords_current)
            model_map[self.attributes[ith_key]] = self.implicit_template(coords_current)
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
        y_displaced = coords_init - y_displacement
        if self.pos_enc:
            y_displaced = self.pos_encoder(y_displaced)
        y = self.implicit_template(y_displaced)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))
        '''
        gradient map
        '''
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':coords_init,
                'template_in': coords_init,
                'coords_with_disp': coords_init - y_displacement,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords




class LipNAIVF_withtempl(nn.Module):
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
                 pos_enc=False):
        super().__init__()
        self.net_parts = nn.ModuleDict({})
        print(template_attributes)
        self.template_attributes = template_attributes
        self.device = device
        self.composer = torch.nn.Parameter(data=torch.tensor(10.0).float(), requires_grad=True)

        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=5, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features


        #self.net_parts['initial'] = Siren(in_features,
        #                                  hidden_features,
        #                                  hidden_layers,
        #                                  out_features,
        #                                  outermost_linear=True,
        #                                  first_omega_0=30,
        #                                 hidden_omega_0=30.).to(device)
        if backbone == 'mlp':
            self.net_parts['initial'] = LipMLPBlock(in_features,
                                                    out_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=False).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(in_features,
                                              hidden_features,
                                              hidden_layers,
                                              out_features,
                                              self.composer,
                                              outermost_linear=False,
                                              first_omega_0=10,
                                             hidden_omega_0=10.).to(device)


        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)
            if backbone == 'mlp':
                self.net_parts[ith_attri] = LipMLPBlock(self.in_features + 1,
                                                        in_features,
                                                        hidden_layers,
                                                        hidden_features,
                                                        outermost_linear=True).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(in_features+1,
                                                hidden_features,
                                                hidden_layers,
                                                in_features,
                                                  self.composer,
                                                outermost_linear=True,
                                                first_omega_0=10,
                                                hidden_omega_0=10.).to(device)

        if backbone == 'siren':
            self.aggregator = Siren(hidden_features,
                                    hidden_features,
                                    hidden_layers,
                                    in_features,
                                    self.composer,
                                    outermost_linear=True,
                                    first_omega_0=10,
                                    hidden_omega_0=10.).to(device)
        elif backbone == 'mlp':
            self.aggregator =LipMLPBlock(in_features + 1,
                                                    hidden_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=False).to(device)


    def forward(self, coords, attributes):
        coords_init = coords.clone().detach().requires_grad_(True)

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
        loss_lip_initial = 1

        '''
        template shape
        '''
        y_init, loss_lip_initial = self.net_parts['initial']((coords, loss_lip_initial))

        model_map = {'initial': y_init}
        vec_fields = {'overall': torch.zeros_like(coords).to(self.device)}
        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())

        '''
        fit shape
        '''

        loss_lip_attr = {}
        y_displacement = torch.zeros_like(coords).to(self.device)
        for ith_key in range(len(self.attributes)):
            '''
            input of current vector field
            '''
            slt_index =torch.tensor(list(range(enc_coords.shape[-1])) + [ith_key + enc_coords.shape[-1]]).to(input.device)
            '''
            current vector field
            '''
            current_delta_coords, loss_lip_attr[self.attributes[ith_key]] = \
                self.net_parts[self.attributes[ith_key]]((input.index_select(-1, slt_index), torch.tensor(1.0).to(self.device)))#[:, :, slt_index])

            #loss_lip += current_loss_lip
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
            current_evolution, _ = self.net_parts['initial']((coords_current, torch.tensor(1.0).to(coords_current.device)))
            model_map[self.attributes[ith_key]] = current_evolution
            '''
            regularize for template
            '''
            current_template_vec = input.index_select(-1, slt_index).clone()
            current_template_vec[:, :, -1] = self.template_attributes[self.attributes[ith_key]]
            vec_fields[self.attributes[ith_key] + '_templ'], _ = self.net_parts[self.attributes[ith_key]]((current_template_vec,
                                                                                                           torch.tensor(1.0).to(coords_current.device)))
            #model_map[self.attributes[ith_key] + '_velocity_field'] = current_delta_coords.clone()
            hypo_params += list(self.net_parts[self.attributes[ith_key]].parameters())

        vf_loss_lip = 0
        for i_attri in range(len(self.attributes)):
            vf_loss_lip += loss_lip_attr[self.attributes[ith_key]]
        '''
        final output of the implicit functino
        '''
        y, loss_lip = self.net_parts['initial']((coords_init - y_displacement, vf_loss_lip/2)) #torch.tensor(1.0).to(coords_current.device)))#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))

        '''
        gradient map
        '''
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':coords_init,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map,
                'loss_lip': loss_lip + vf_loss_lip,
                'loss_lip_initial': loss_lip_initial
            }
        return model_output#, coords




class NAIVF(nn.Module):
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
        self.net_parts = nn.ModuleDict({})
        print(attibutes)
        self.attributes = attibutes
        self.in_features = in_features
        self.device = device
        #self.net_parts['initial'] = Siren(in_features,
        #                                  hidden_features,
        #                                  hidden_layers,
        #                                  out_features,
        #                                  outermost_linear=True,
        #                                  first_omega_0=30,
        #                                 hidden_omega_0=30.).to(device)
        self.net_parts['initial'] = BaseFCBlock(in_features,
                                                out_features,
                                                hidden_layers,
                                                hidden_features,
                                                outermost_linear=True,
                                                nonlinearity='relu',
                                                weight_init=None).to(device)
        for ith_attri in attibutes:
            #self.net_parts[ith_attri] = Siren_per_attri(in_features,
            #                                            hidden_features,
            #                                            hidden_layers,
            #                                            in_features,
            #                                            outermost_linear=True,
            #                                            first_omega_0=30,
            #                                            hidden_omega_0=30.).to(device)
            self.net_parts[ith_attri] = BaseFCBlock(in_features + 1,
                                                    in_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=True,
                                                    nonlinearity='relu',
                                                    weight_init=None).to(device)
    def forward(self, coords, attributes):
        coords_init = coords.clone().detach().requires_grad_(True)
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        input = torch.cat([coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)


        y_init = self.net_parts['initial'](input[:, :, 0:coords.shape[-1]].clone())
        model_map = {'initial': y_init}
        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())

        y_coords = input.index_select(-1, torch.arange(self.in_features).to(self.device))#[:, :, 0:coords.shape[-1]]
        for ith_key in range(len(self.attributes)):
            slt_index =torch.tensor(list(range(coords.shape[-1])) + [ith_key + coords.shape[-1]]).to(input.device)
            current_delta_coords = self.net_parts[self.attributes[ith_key]](input.index_select(-1, slt_index))#[:, :, slt_index])
            y_coords = y_coords + current_delta_coords

            coords_current = coords_init + current_delta_coords
            model_map[self.attributes[ith_key]] = self.net_parts['initial'](coords_current.clone())
            #model_map[self.attributes[ith_key] + '_velocity_field'] = current_delta_coords.clone()
            hypo_params += list(self.net_parts[self.attributes[ith_key]].parameters())


        y = self.net_parts['initial'](y_coords)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))

        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient


        model_output = \
            {   'all_input': input,
                'model_in':coords_init,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords

class BaselineVF(nn.Module):
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
                 pos_enc=False,):
        super().__init__()

        self.net_parts = nn.ModuleDict({})
        self.template_attributes = template_attributes
        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)

        self.composer = torch.nn.Parameter(data=torch.tensor(5.0).float(), requires_grad=True)

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

            self.net_parts['overall'] = BaseFCBlock(in_features + len(self.attributes),
                                                  in_features,
                                                  hidden_layers,
                                                  hidden_features,
                                                  outermost_linear=True,
                                                  nonlinearity='tanh',
                                                  weight_init=None).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(in_features,
                                              hidden_features,
                                              hidden_layers,
                                              out_features,
                                              composer=self.composer,
                                              outermost_linear=False,
                                              first_omega_0=5.,
                                              hidden_omega_0=5.).to(device)

            self.net_parts['overall'] = Siren(in_features+len(self.attributes),
                                          hidden_features,
                                          hidden_layers,
                                          in_features,
                                          outermost_linear=True,
                                          composer=self.composer,
                                          first_omega_0=5.,
                                          hidden_omega_0=5.).to(device)
    def forward(self, coords, attributes):


        coords_init = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        '''
        get input array of the network
        '''

        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init

        vec_fields = {}
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        input = torch.cat([enc_coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)

        y_init = self.net_parts['initial'](enc_coords)
        model_map = {'initial': y_init}
        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())

        current_delta_coords = self.net_parts['overall'](input)
        y_coords = coords_init - current_delta_coords
        vec_fields['overall'] = current_delta_coords
        hypo_params += list(self.net_parts['overall'].parameters())
        y = self.net_parts['initial'](y_coords)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))


        template_vec = input.clone()
        for ith_key in range(len(self.attributes)):
            template_vec[:, :, enc_coords.shape[-1]+ith_key] = self.template_attributes[self.attributes[ith_key]]

        vec_fields['_templ'] = self.net_parts['overall'](template_vec)
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient


        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':enc_coords,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords



class SirenlatentVF(nn.Module):
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
                 hidden_omega_0=30.):
        super().__init__()

        self.net_parts = nn.ModuleDict({})
        self.template_attributes = template_attributes
        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)

        self.composer = torch.nn.Parameter(data=torch.tensor(10.0).float(), requires_grad=True)

        if backbone == 'mlp':
            self.net_parts['initial'] = BaseFCBlock(in_features,
                                                    out_features,
                                                    hidden_layers,
                                                    hidden_features,
                                                    outermost_linear=False,
                                                    nonlinearity='relu',
                                                    weight_init=None).to(device)
            self.net_parts['attri'] = BaseFCBlock(in_features + len(self.attributes),
                                                  16,
                                                  hidden_layers,
                                                  hidden_features,
                                                  outermost_linear=False,
                                                  nonlinearity='tanh',
                                                  weight_init=None).to(device)
        else:
            self.net_parts['initial'] = Siren(in_features,
                                              hidden_features,
                                              hidden_layers,
                                              out_features,
                                              composer=self.composer,
                                              outermost_linear=True,
                                              first_omega_0=10,
                                              hidden_omega_0=10.).to(device)
            self.net_parts['attri'] =  Siren(in_features+len(self.attributes),
                                          hidden_features,
                                          hidden_layers,
                                          16,
                                          outermost_linear=True,
                                        composer=self.composer,
                                          first_omega_0=10,
                                          hidden_omega_0=10.).to(device)
        self.f_0 = BaseFCBlock(in_features=len(self.attributes),
                                            out_features=in_features,
                                            num_hidden_layers=1,
                                            hidden_features=16,
                                            outermost_linear=True,
                                            nonlinearity='relu',
                                            weight_init=None).to(device)
        if backbone == 'siren':
            self.aggregator = Siren(in_features=16,
                                    hidden_features=16,
                                    hidden_layers=1,
                                    out_features=in_features,
                                    outermost_linear=True,
                                    composer=self.composer,
                                    first_omega_0=30,
                                    hidden_omega_0=30.).to(device)



        elif backbone == 'mlp':
            self.aggregator = BaseFCBlock(in_features=16,
                                            out_features=in_features,
                                            num_hidden_layers=1,
                                            hidden_features=16,
                                            outermost_linear=True,
                                            nonlinearity='relu',
                                            weight_init=None).to(device)

    def forward(self, coords, attributes):


        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        vec_fields = {}
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        input = torch.cat([coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)

        y_init = self.net_parts['initial'](input[:, :, 0:coords.shape[-1]])
        model_map = {'initial': y_init}
        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())

        y_delta_coords = input.index_select(-1, torch.tensor(torch.arange(coords.shape[-1])).to(input.device))
        current_delta_coords = self.aggregator(self.net_parts['attri'](input))  #[:, :, slt_index])\
        #current_delta_coords -=  torch.mean(current_delta_coords, dim=-1, keepdim=True)
        y_coords = (y_delta_coords - current_delta_coords)* self.f_0(input[:, :, torch.arange(coords.shape[-1])])
        vec_fields['overall'] = current_delta_coords
        hypo_params += list(self.net_parts['attri'].parameters())
        y = self.net_parts['initial'](y_coords)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))


        template_vec = input.clone()
        for ith_key in range(len(self.attributes)):
            template_vec[:, :, coords.shape[-1]+ith_key] = self.template_attributes[self.attributes[ith_key]]

        vec_fields['_templ'] = self.net_parts['attri'](template_vec)
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient


        model_output = \
            {   'all_input': input,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in':coords,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords





class HyperSirenBaseline(nn.Module):
    def __init__(self,
                 attributes,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 device,
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()
        self.net_parts = nn.ModuleDict({})
        self.net = NeuralProcessImplicitHypernet(
                         in_features=in_features,
                         out_features=out_features,
                         attribute_dim=len(attributes),
                         hidden_features=hidden_features,
                         hidden_layers=hidden_layers,
                         encoder_nl='sine',
                         device=device)

    def forward(self, coords, attributes):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        hypo_params = self.net.parameters()
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)

        y = self.net(coords, arr_attri.float())['model_out']
        model_output = {'model_in':coords,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params}
        return model_output#, coords


class SirenBaseline(nn.Module):
    def __init__(self, attributes, in_features, hidden_features, hidden_layers, out_features, device, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.attributes = attributes
        in_features = in_features + len(attributes)
        self.net = BaseFCBlock(in_features,
                               out_features,
                               hidden_layers,
                               hidden_features,
                                outermost_linear=True,
                               nonlinearity='relu',
                               weight_init=None).to(device)

    def forward(self, coords, attributes):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        #print(coords.shape)
        #print(arr_attri.shape)
        input = torch.cat([coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)
        output = self.net(input)
        #get map
        model_map = {}
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(output, input)[:, :,[i_attri+coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient
        model_output = {'model_in':coords,
                        'all_input': input,
                        'model_out':output,
                        'latent_vec': attributes,
                        'model_map': model_map}

        return model_output#, coords



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



class Baseline(nn.Module):
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
                 pos_enc=False):
        super().__init__()
        print(template_attributes)
        self.template_attributes = template_attributes
        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
        self.device = device
        self.composer = torch.nn.Parameter(data=torch.tensor(5.0).float(), requires_grad=True)

        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=5, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features

        self.in_features = self.in_features + len(self.template_attributes)

        if backbone == 'mlp':
            self.net = BaseFCBlock(self.in_features,
                                   out_features,
                                    hidden_layers,
                                    hidden_features,
                                    outermost_linear=True,
                                    nonlinearity='relu',
                                    weight_init=None).to(device)
        elif backbone == 'siren':
            self.net = Siren(self.in_features,
                              hidden_features,
                              hidden_layers,
                              out_features,
                              self.composer,
                              outermost_linear=True,
                              first_omega_0=10.,
                             hidden_omega_0=10.).to(device)


    def forward(self, coords, attributes):
        coords_init = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        '''
        get input array of the network
        '''

        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init


        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        #print(coords.shape)
        #print(arr_attri.shape)
        input = torch.cat([enc_coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)
        output = self.net(input)
        #get map
        model_map = {}
        for i_attri in range(len(self.template_attributes)):
            gradient = diff_operators.gradient(output, input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient
        model_output = {'model_in':coords_init,
                        'all_input': input,
                        'model_out':output,
                        'latent_vec': attributes,
                        'model_map': model_map}

        return model_output#, coords



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





class FCBaseline(nn.Module):
    '''
    for pediatric airway atlas
    '''
    def __init__(self,
                 attributes,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 device,
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()
        self.attributes = attributes
        self.in_features = in_features + len(attributes)
        self.net = BaseFCBlock(self.in_features,
                               out_features,
                               hidden_layers,
                               hidden_features,
                               outermost_linear=True,
                               nonlinearity='sine',
                               weight_init=None).to(device)

    def forward(self, coords, attributes):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        #arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        input = torch.cat([coords, arr_attri], dim=-1).float().clone().detach().requires_grad_(True)
        output = self.net(input)

        '''
        get map
        '''
        model_map = {}
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(output, input)[:, [i_attri+coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        model_output = {'model_in':coords,
                        'all_input': input,
                        'model_out':output,
                        'latent_vec': attributes,
                        'model_map': model_map}

        return model_output#, coords



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





class NAIlatentVF_withtempl(nn.Module):
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
                 pos_enc=False):
        super().__init__()
        self.net_parts = nn.ModuleDict({})
        print(template_attributes)
        self.template_attributes = template_attributes
        self.composer = torch.nn.Parameter(data=torch.tensor(10.0).float(), requires_grad=True)
        self.device = device
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
            self.net_parts['initial'] = BaseFCBlock(in_features=in_features,
                                                    out_features=out_features,
                                                    num_hidden_layers=hidden_layers,
                                                    hidden_features=hidden_features,
                                                    outermost_linear=False,
                                                    nonlinearity='relu',
                                                    weight_init=None).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(in_features=in_features,
                                              hidden_features=hidden_features,
                                              hidden_layers=hidden_layers,
                                              out_features=out_features,
                                              composer=self.composer,
                                              outermost_linear=False,
                                              first_omega_0=5,
                                             hidden_omega_0=5.,
                                              zero_init_last_layer=True,
                                              ).to(device)
        '''
        self.net_parts['initial'] = Siren(in_features=in_features,
                                          hidden_features=hidden_features,
                                          hidden_layers=hidden_layers,
                                          out_features=out_features,
                                          composer=self.composer,
                                          outermost_linear=False,
                                          first_omega_0=5.0,
                                          hidden_omega_0=5.0,
                                          zero_init_last_layer=True,
                                          ).to(device)
        '''


        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)
            if backbone == 'mlp':
                self.net_parts[ith_attri] = BaseFCBlock(in_features=self.in_features + 1,
                                                        out_features=16,
                                                        num_hidden_layers=hidden_layers,
                                                        hidden_features=hidden_features,
                                                        outermost_linear=True,
                                                        nonlinearity='relu',
                                                        weight_init=None).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(in_features=self.in_features+1,
                                                hidden_features=hidden_features,
                                                hidden_layers=hidden_layers,
                                                out_features=16,
                                                composer=self.composer,
                                                outermost_linear=True,
                                                first_omega_0=5.,
                                                hidden_omega_0=5.,
                                                zero_init_last_layer=False,
                                                  ).to(device)
        if backbone == 'siren':
            self.aggregator = Siren(in_features=16,
                                    hidden_features=16,
                                    hidden_layers=1,
                                    out_features=in_features,
                                    composer=self.composer,
                                    outermost_linear=True,
                                    first_omega_0=5.0,
                                    hidden_omega_0=5.,
                                    zero_init_last_layer=False).to(device)
        elif backbone == 'mlp':
            self.aggregator = BaseFCBlock(in_features=16,
                                            out_features=in_features,
                                            num_hidden_layers=1,
                                            hidden_features=16,
                                            outermost_linear=True,
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

        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        input = torch.cat([enc_coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)

        '''
        template shape
        '''



        vec_fields = {'overall': torch.zeros((coords.shape[0], coords.shape[1], 16)).to(self.device)}
        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())

        latent_y = torch.zeros((coords.shape[0], coords.shape[1], 16)).to(self.device)
        latent_y_templ = torch.zeros((coords.shape[0], coords.shape[1], 16)).to(self.device)
        '''
        fit shape
        '''
        #y_coords = input.index_select(-1, torch.arange(coords.shape[-1]).to(input.device))#[:, :, 0:coords.shape[-1]]
        y_init = self.net_parts['initial'](coords_init)
        model_map = {'initial': y_init}
        for ith_key in range(len(self.attributes)):
            '''
            input of current vector field
            '''
            slt_index =torch.tensor(list(range(enc_coords.shape[-1])) + [ith_key + enc_coords.shape[-1]]).to(input.device)
            '''
            current vector field
            '''
            current_delta_coords = self.net_parts[self.attributes[ith_key]](input.index_select(-1, slt_index))#[:, :, slt_index])
            vec_fields[self.attributes[ith_key]] = self.aggregator(current_delta_coords) # explicit displacement
            vec_fields[self.attributes[ith_key] + '_implicit'] = current_delta_coords # implicit displacement
            '''
            current deformation
            '''
            latent_y = latent_y + current_delta_coords
            '''
            individual evolution
            '''
            model_map[self.attributes[ith_key]] = self.net_parts['initial'](coords_init - self.aggregator(current_delta_coords))
            '''
            regularize for template
            '''
            current_template_vec = input.index_select(-1, slt_index).clone()
            current_template_vec[:, :, -1] = self.template_attributes[self.attributes[ith_key]]
            #model_map[self.attributes[ith_key] + '_velocity_field'] = current_delta_coords.clone()
            hypo_params += list(self.net_parts[self.attributes[ith_key]].parameters())

            latent_y_templ += self.net_parts[self.attributes[ith_key]](current_template_vec)
            vec_fields[self.attributes[ith_key] + '_template'] = self.aggregator(self.net_parts[self.attributes[ith_key]](current_template_vec)) # explicit displacement
            vec_fields[self.attributes[ith_key] + '_template_implicit'] = self.net_parts[self.attributes[ith_key]](current_template_vec) # implicit displacement
        '''
        template vector field
        '''
        vec_fields['template_implicit'] = latent_y_templ
        vec_fields['template'] = self.aggregator(latent_y_templ)
        '''
        overall vector field
        '''
        vec_fields['overall_implicit'] = latent_y
        vec_fields['overall'] = self.aggregator(latent_y)
        '''
        final output of the implicit function
        '''
        y = self.net_parts['initial'](coords_init - self.aggregator(latent_y))#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))
        '''
        gradient map
        '''
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, input)[:, :,[i_attri+enc_coords.shape[-1]]]
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


class LIPNAIlatentVF_withtempl(nn.Module):
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
                 hidden_omega_0=30.):
        super().__init__()
        self.net_parts = nn.ModuleDict({})
        print(template_attributes)
        self.template_attributes = template_attributes
        self.composer = torch.nn.Parameter(data=torch.tensor(10.0).float(), requires_grad=True)
        self.device = device
        #self.net_parts['initial'] = Siren(in_features,
        #                                  hidden_features,
        #                                  hidden_layers,
        #                                  out_features,
        #                                  outermost_linear=True,
        #                                  first_omega_0=30,
        #                                 hidden_omega_0=30.).to(device)

        if backbone == 'mlp':
            self.net_parts['initial'] = LipMLPBlock(in_features=in_features,
                                                    out_features=out_features,
                                                    num_hidden_layers=hidden_layers,
                                                    hidden_features=hidden_features,
                                                    outermost_linear=False).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(in_features=in_features,
                                              hidden_features=hidden_features,
                                              hidden_layers=hidden_layers,
                                              out_features=out_features,
                                              composer=self.composer,
                                              outermost_linear=False,
                                              first_omega_0=10,
                                             hidden_omega_0=10.,
                                              ).to(device)
        '''
        self.net_parts['initial'] = Siren(in_features=in_features,
                                          hidden_features=hidden_features,
                                          hidden_layers=hidden_layers,
                                          out_features=out_features,
                                          composer=self.composer,
                                          outermost_linear=False,
                                          first_omega_0=10,
                                          hidden_omega_0=10.,
                                          ).to(device)
        '''

        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)
            if backbone == 'mlp':
                self.net_parts[ith_attri] = LipMLPBlock(in_features=in_features + 1,
                                                        out_features=16,
                                                        num_hidden_layers=hidden_layers,
                                                        hidden_features=hidden_features,
                                                        outermost_linear=True).to(device)
            elif backbone == 'siren':
                self.net_parts[ith_attri] = Siren(in_features=in_features+1,
                                                hidden_features=hidden_features,
                                                hidden_layers=hidden_layers,
                                                out_features=16,
                                                composer=self.composer,
                                                outermost_linear=True,
                                                first_omega_0=10,
                                                hidden_omega_0=10.,).to(device)
        if backbone == 'siren':
            self.aggregator = Siren(in_features=16,
                                    hidden_features=16,
                                    hidden_layers=1,
                                    out_features=in_features,
                                    composer=self.composer,
                                    outermost_linear=False,
                                    first_omega_0=10,
                                    hidden_omega_0=10.).to(device)
        elif backbone == 'mlp':
            self.aggregator = LipMLPBlock(in_features=16,
                                            out_features=in_features,
                                            num_hidden_layers=1,
                                            hidden_features=16,
                                            outermost_linear=True).to(device)

    def forward(self, coords, attributes):
        coords_init = coords.clone().detach().requires_grad_(True)
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        '''
        get input array of the network
        '''
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        input = torch.cat([coords, arr_attri], dim=-1).clone().detach().requires_grad_(True)

        loss_lip_initial = 1
        '''
        template shape
        '''
        y_init, loss_lip_initial = self.net_parts['initial']((coords_init, loss_lip_initial))
        model_map = {'initial': y_init}

        vec_fields = {'overall': torch.zeros((coords.shape[0], coords.shape[1], 16)).to(self.device)}
        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())

        latent_y = torch.zeros((coords.shape[0], coords.shape[1], 16)).to(self.device)
        latent_y_templ = torch.zeros((coords.shape[0], coords.shape[1], 16)).to(self.device)

        '''
        fit shape
        '''
        loss_lip_attr = {}
        y_coords = input.index_select(-1, torch.arange(coords.shape[-1]).to(input.device))#[:, :, 0:coords.shape[-1]]


        for ith_key in range(len(self.attributes)):
            '''
            input of current vector field
            '''
            slt_index =torch.tensor(list(range(coords.shape[-1])) + [ith_key + coords.shape[-1]]).to(input.device)
            '''
            current vector field
            '''
            current_delta_coords, current_lip_loss = \
                self.net_parts[self.attributes[ith_key]]((input.index_select(-1, slt_index), 1))  # [:, :, slt_index])

            vec_fields[self.attributes[ith_key]], loss_lip_attr[self.attributes[ith_key]] = self.aggregator(current_delta_coords, current_lip_loss)
            '''
            current deformation
            '''
            latent_y = latent_y + current_delta_coords
            '''
            individual evolution
            '''
            current_evolution, loss_lip_attr[self.attributes[ith_key]] = self.net_parts['initial']((y_coords - self.aggregator(current_delta_coords)),
                                                                                                   torch.tensor(1.0).to(self.device))
            model_map[self.attributes[ith_key]] = current_evolution
            '''
            regularize for template
            '''
            current_template_vec = input.index_select(-1, slt_index).clone()
            current_template_vec[:, :, -1] = self.template_attributes[self.attributes[ith_key]]
            #model_map[self.attributes[ith_key] + '_velocity_field'] = current_delta_coords.clone()
            hypo_params += list(self.net_parts[self.attributes[ith_key]].parameters())

            current_latent_y_templ, _= self.net_parts[self.attributes[ith_key]]((current_template_vec, torch.tensor(1.0).to(self.device)))
            latent_y_templ += current_latent_y_templ
            vec_fields[self.attributes[ith_key] + '_template'] = current_latent_y_templ

        vec_fields['template'], _ = self.aggregator((latent_y_templ, torch.tensor(1.0).to(self.device)))
        '''
        overall vector field
        '''
        vec_fields['overall'],  = self.aggregator((latent_y, torch.tensor(1.0).to(self.device) ))
        '''
        final output of the implicit function
        '''
        u_of_y, loss_lip = self.aggregator((latent_y, torch.tensor(1.0).to(self.device)))
        y = self.net_parts['initial']((y_coords -  u_of_y, loss_lip))#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))
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
                'model_map': model_map,
                'loss_lip': vf_loss_lip + loss_lip + loss_lip_initial,
            }
        return model_output#, coords


class ICVF(nn.Module):
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
                 pos_enc=False,):
        super().__init__()

        self.net_parts = nn.ModuleDict({})
        self.device = device
        self.template_attributes = template_attributes
        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            self.template_attributes[ith_attri] = torch.tensor(self.template_attributes[ith_attri]).to(device)

        self.composer = torch.nn.Parameter(data=torch.tensor(5.0).float(), requires_grad=True)

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

            self.net_parts['overall'] = BaseFCBlock(in_features + len(self.attributes) * 2,
                                                  in_features,
                                                  hidden_layers,
                                                  hidden_features,
                                                  outermost_linear=True,
                                                  nonlinearity='tanh',
                                                  weight_init=None).to(device)
        elif backbone == 'siren':
            self.net_parts['initial'] = Siren(in_features,
                                              hidden_features,
                                              hidden_layers,
                                              out_features,
                                              composer=self.composer,
                                              outermost_linear=False,
                                              first_omega_0=30.,
                                              hidden_omega_0=30.).to(device)

            self.net_parts['overall'] = Siren(in_features+len(self.attributes) * 2,
                                          hidden_features,
                                          hidden_layers,
                                          in_features,
                                          outermost_linear=True,
                                          composer=self.composer,
                                          first_omega_0=30.,
                                          hidden_omega_0=30.).to(device)
    def forward(self, coords, attributes):

        coords_init = coords.clone().detach().requires_grad_(True)
        '''
        get input array of the network
        '''

        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init

        '''
        
        '''
        vec_fields = {}
        list_attributes = []
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()

        arr_attri_templ = torch.zeros_like(arr_attri).to(self.device).clone().detach().requires_grad_(True)
        # target -> template
        forward_input = torch.cat([enc_coords, arr_attri, arr_attri_templ], dim=-1).clone().detach().requires_grad_(True) #\


        y_init = self.net_parts['initial'](enc_coords)

        model_map = {'initial': y_init}

        # from target to template
        current_delta_coords = self.net_parts['overall'](forward_input)
        y_coords_template = coords_init - current_delta_coords

        # template -> target
        inverse_input = torch.cat([y_coords_template, arr_attri_templ, arr_attri], dim=-1)

        # from template to target
        inverse_current_delta_coords = self.net_parts['overall'](inverse_input)
        y_coords_inverse = y_coords_template + inverse_current_delta_coords

        vec_fields['overall'] = current_delta_coords

        #
        y = self.net_parts['initial'](y_coords_template)#(torch.index_select(input, dim=-1, index=torch.tensor([0,1]).to(input.device)))

        template_vec = forward_input.clone()
        for ith_key in range(len(self.attributes)):
            template_vec[:, :, enc_coords.shape[-1]+ith_key] = self.template_attributes[self.attributes[ith_key]]

        vec_fields['_templ'] = self.net_parts['overall'](template_vec)
        for i_attri in range(len(self.attributes)):
            gradient = diff_operators.gradient(y, forward_input)[:, :,[i_attri+enc_coords.shape[-1]]]
            model_map[self.attributes[i_attri] + '_grad'] = gradient

        hypo_params = []
        hypo_params += list(self.net_parts['initial'].parameters())
        hypo_params += list(self.net_parts['overall'].parameters())
        model_output = \
            {   'all_input': forward_input,
                'inverse_to_target': y_coords_inverse,
                'template': y_init,
                'vec_fields': vec_fields,
                'model_in': enc_coords,
                'model_out':y,
                'latent_vec': attributes,
                'hypo_params':hypo_params,
                'model_map': model_map
            }
        return model_output#, coords

