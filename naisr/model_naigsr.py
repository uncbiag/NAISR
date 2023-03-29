import torch
from torch import nn
from collections import OrderedDict
from naisr.modules import SineLayer, Siren, BaseFCBlock, LipMLPBlock, LipBoundedPosEnc, BaseDeepSDF, \
    BaseDeepSDFSiren  # ,HyperNetwork, NeuralProcessImplicitHypernet
import numpy as np

import naisr.diff_operators as diff_operators


class DeepNAIGSR(nn.Module):
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
        self.net_direction = nn.ModuleDict({})
        self.net_amplitude = nn.ModuleDict({})

        self.latent_size = latent_size
        self.template_attributes = template_attributes
        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)

        self.device = device
        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=10, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features

        if backbone == 'mlp':
            self.template = BaseFCBlock(3,
                                        out_features,
                                        hidden_layers,
                                        hidden_features,
                                        outermost_linear=False,
                                        nonlinearity='relu',
                                        weight_init=None).to(device)

        elif backbone == 'siren':
            self.template = Siren(3,
                                  hidden_features,
                                  hidden_layers,
                                  out_features,
                                  3,
                                  outermost_linear=False,
                                  first_omega_0=30.,
                                  hidden_omega_0=30.).to(device)

        self.attributes = []
        for ith_attri in self.template_attributes.keys():
            self.attributes.append(ith_attri)
            if backbone == 'mlp':
                self.net_direction[ith_attri] = BaseDeepSDF(in_features=self.in_features,
                                                            latent_size=self.latent_size + 1,
                                                            out_features=3,
                                                            num_hidden_layers=hidden_layers,
                                                            hidden_features=hidden_features,
                                                            latent_in=[hidden_layers // 2],
                                                            outermost_linear=True,
                                                            nonlinearity='relu',
                                                            weight_init=None).to(device)
            elif backbone == 'siren':

                self.net_direction[ith_attri] = BaseDeepSDFSiren(in_features=self.in_features,
                                                                 latent_size=latent_size + 1,
                                                                 hidden_features=hidden_features,
                                                                 hidden_layers=hidden_layers,
                                                                 out_features=3,
                                                                 latent_in=[hidden_layers // 2],
                                                                 outermost_linear=True,
                                                                 first_omega_0=30.,
                                                                 hidden_omega_0=30.).to(device)



    def get_displacement_per_covariate(self, enc_coords, embedding, covaraite, which_covariate, whether_inv=False, training=True):

        embedding_with_attri = torch.cat((covaraite, embedding), dim=-1)
        if training:
            embedding_with_zeros = torch.cat((torch.randn_like(covaraite)*1e-3, embedding), dim=-1)
        else:
            embedding_with_zeros = torch.cat((torch.zeros_like(covaraite) * 1e-3, embedding), dim=-1)

        arr_disp = self.net_direction[which_covariate](embedding_with_attri, enc_coords) - self.net_direction[which_covariate](embedding_with_zeros, enc_coords)
        if not whether_inv:
            return arr_disp
        else:
            return -arr_disp

    '''
    def get_displacement_per_covariate_wo_embd(self, enc_coords, embedding, covaraite, which_covariate, whether_inv=False):

        embedding_with_attri = torch.cat((covaraite, embedding), dim=-1)
        embedding_with_zeros = torch.cat((torch.zeros_like(covaraite), embedding), dim=-1)
        arr_disp = self.net_direction[which_covariate](embedding_with_attri, enc_coords) - self.net_direction[which_covariate](embedding_with_zeros, enc_coords)
        if not whether_inv:
            return arr_disp
        else:
            return -arr_disp
    '''

    def evolution(self, coords, attributes, embedding, which_attribute):
        coords_init = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        for ith_attri in attributes.keys():
            if ith_attri == which_attribute:
                arr_attri = attributes[ith_attri][:, None]
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float().requires_grad_(True)

        '''
        get input array of the network
        '''
        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init

        '''
        use embedding as prior
        '''
        vf = torch.zeros((coords.shape[0], coords.shape[1], 3)).to(self.device)
        dict_vf = {}

        current_vf = self.get_displacement_per_covariate(enc_coords, embedding, arr_attri,
                                                         which_attribute)

        dict_vf[which_attribute] = current_vf
        vf += current_vf

        dict_vf['overall'] = vf
        output = self.template(vf + coords_init)
        # get map
        model_output = {'model_in': coords_init,
                        'all_input': coords_init,
                        'model_out': output,
                        'vec_fields': dict_vf,
                        'covariates': arr_attri,
                        }
        return model_output

    def forward(self, coords, attributes_1, embedding, testing=False, training=True):

        attributes = {}

        if testing:
            for name, value in attributes_1.items():
                attributes[name] = torch.tanh(attributes_1[name])
        else:

            attributes = attributes_1

        coords_init = coords.requires_grad_(True)  # allows to take derivative w.r.t. input

        list_attributes = []

        # divider = 0.0
        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
            # divider += torch.exp(self.factor[ith_attri])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        #arr_attri = arr_attri.requires_grad_(True)

        '''
        get input array of the network
        '''
        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init


        '''
        use embedding as prior
        '''
        vf = torch.zeros((coords.shape[0], coords.shape[1], coords_init.shape[-1])).to(self.device)
        dict_vf = {}
        dict_disentangle = {}
        for ith_key in range(len(self.attributes)):
            current_vf = self.get_displacement_per_covariate(enc_coords, embedding, arr_attri[..., [ith_key]],
                                                                      self.attributes[ith_key], training=training)
            current_vf_inv = self.get_displacement_per_covariate(enc_coords+current_vf, embedding, arr_attri[..., [ith_key]],
                                                                      self.attributes[ith_key], whether_inv=True, training=training)

            dict_vf[self.attributes[ith_key]] = current_vf
            dict_vf[self.attributes[ith_key] + '_inv'] = current_vf_inv

            vf += current_vf
        dict_vf['overall'] = vf
        output = self.template(vf + coords_init)

        '''
        wihtout embedding as prior
        '''
        vf_wo_z = torch.zeros((coords.shape[0], coords.shape[1], 3)).to(self.device)
        zero_padding = torch.zeros_like(embedding)
        for ith_key in range(len(self.attributes)):
            current_vf = self.get_displacement_per_covariate(enc_coords, zero_padding, arr_attri[..., [ith_key]],
                                                                      self.attributes[ith_key], training=training)
            current_vf_inv = self.get_displacement_per_covariate(enc_coords + current_vf, zero_padding,
                                                                 arr_attri[..., [ith_key]],
                                                                 self.attributes[ith_key], whether_inv=True, training=training)

            dict_vf[self.attributes[ith_key] + '_z_padding'] = current_vf
            dict_vf[self.attributes[ith_key] + '_z_padding' + '_inv'] = current_vf_inv
            vf_wo_z += current_vf

        dict_vf['overall_z_padding'] = vf_wo_z
        output_z_padding = self.template( vf_wo_z + coords_init)

        # get map
        model_output = {'model_in': coords_init,
                        'all_input': coords_init,
                        'model_out': output,
                        'vec_fields': dict_vf,
                        'model_out_z_padding': output_z_padding,
                        'disentangle': dict_disentangle,
                        'covariates': arr_attri,
                        'embedding': embedding,
                        'template': self.template(coords_init)
                        }

        return model_output  # , coords

    def template_sdf(self, coords):
        return self.template(coords)

    def inv_transorm(self, coords, attributes_1, embedding, whether_inv=True):
        attributes = {}
        attributes = attributes_1


        coords_init = coords.requires_grad_(True)  # allows to take derivative w.r.t. input

        list_attributes = []

        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()
        # arr_attri = arr_attri.requires_grad_(True)

        '''
        get input array of the network
        '''
        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init

        '''
        use embedding as prior
        '''

        svf = torch.zeros_like(coords).to(self.device)
        dict_vf = {}
        dict_disentangle = {}
        for ith_key in range(len(self.attributes)):
            current_svf = self.get_displacement_per_covariate(enc_coords, embedding,  arr_attri[..., [ith_key]], self.attributes[ith_key], whether_inv=True)
            dict_vf[self.attributes[ith_key]] = current_svf
            svf += current_svf

        transformed_p = svf + coords_init
        return transformed_p


    def template_transfer(self, coords, attributes_1,  embedding, new_template_type, sex, testing=False):

        '''
        get attributes
        '''
        attributes = {}
        if testing:
            for name, value in attributes_1.items():
                attributes[name] = torch.tanh(attributes_1[name])
        else:

            attributes = attributes_1
        coords_init = coords.requires_grad_(True)  # allows to take derivative w.r.t. input
        list_attributes = []

        for ith_attri in attributes.keys():
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()

        '''
        
        '''
        if new_template_type == 'mean':
            dict_attributes_new_temp = {'weight': -1,'age': -1, 'sex': sex}
        elif new_template_type == 'min':
            dict_attributes_new_temp = {'weight': -1, 'age': -1, 'sex': sex}
        else:
            dict_attributes_new_temp = {'weight': 0, 'age': 0, 'sex': 0}

        list_attributes_new_template = []
        for ith_attri in attributes.keys():
            list_attributes_new_template.append(torch.zeros_like(attributes[ith_attri][:, None]) + dict_attributes_new_temp[ith_attri])

        arr_attri_new_template = torch.cat(list_attributes_new_template, dim=-1)
        arr_attri_new_template = arr_attri_new_template.repeat(1, coords.shape[1], 1).float()


        '''
        get input array of the network
        '''
        if self.pos_enc:
            enc_coords = self.pos_encoder(coords_init)
        else:
            enc_coords = coords_init



        '''
        use embedding as prior
        '''
        vf = torch.zeros((coords.shape[0], coords.shape[1], coords_init.shape[-1])).to(self.device)
        dict_vf = {}
        dict_disentangle = {}
        for ith_key in range(len(self.attributes)):
            current_vf = self.get_displacement_per_covariate_temp_transfer(enc_coords, embedding,
                                                                           arr_attri[..., [ith_key]],
                                                                           arr_attri_new_template[..., [ith_key]],
                                                                           self.attributes[ith_key])
            dict_vf[self.attributes[ith_key]] = current_vf
            vf += current_vf
        dict_vf['overall'] = vf
        enc_coords_on_new_temp = vf + enc_coords

        vf = torch.zeros((coords.shape[0], coords.shape[1], coords_init.shape[-1])).to(self.device)
        for ith_key in range(len(self.attributes)):
            current_vf = self.get_displacement_per_covariate(enc_coords_on_new_temp, embedding,
                                                                           arr_attri_new_template[..., [ith_key]],
                                                                           self.attributes[ith_key])
            dict_vf[self.attributes[ith_key]] += current_vf
            vf += current_vf
        dict_vf['overall'] += vf
        output = self.template(vf + enc_coords_on_new_temp)

        return output

    def get_displacement_per_covariate_temp_transfer(self, enc_coords, embedding, covaraite, template_covariate, which_covariate, whether_inv=False, training=True):

        embedding_with_attri = torch.cat((covaraite, embedding), dim=-1)
        if training:
            embedding_with_template = torch.cat((torch.randn_like(covaraite)*1e-3 + template_covariate, embedding), dim=-1)
        else:
            embedding_with_template = torch.cat((torch.zeros_like(covaraite) + template_covariate, embedding), dim=-1)

        arr_disp = self.net_direction[which_covariate](embedding_with_attri, enc_coords) - self.net_direction[which_covariate](embedding_with_template, enc_coords)
        if not whether_inv:
            return arr_disp
        else:
            return -arr_disp