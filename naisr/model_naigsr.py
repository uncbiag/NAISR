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
                 latent_size=256,
                 whether_sampling=False):
        super().__init__()
        self.net_direction = nn.ModuleDict({})
        self.net_amplitude = nn.ModuleDict({})
        self.whether_sampling=whether_sampling
        self.latent_size = latent_size
        self.template_attributes = template_attributes
        self.attributes_names = list(self.template_attributes.keys())
        self.attributes_names.sort()

        self.device = device
        self.pos_enc = pos_enc
        self.ori_in_features = in_features
        if pos_enc:
            self.pos_encoder = LipBoundedPosEnc(in_features, n_freq=15, cat_inp=True)
            self.in_features = self.pos_encoder.out_dim
        else:
            self.in_features = in_features

        if backbone == 'mlp':
            self.template = BaseFCBlock(in_features,
                                        out_features,
                                        hidden_layers,
                                        hidden_features,
                                        outermost_linear=False,
                                        nonlinearity='relu',
                                        weight_init=None)#.cuda()#.to(device)

        elif backbone == 'siren':
            self.template = Siren(in_features,# 2 or 3
                                  hidden_features,
                                  hidden_layers,
                                  out_features,
                                  in_features, # 2 or 3
                                  outermost_linear=False,
                                  first_omega_0=30.,
                                  hidden_omega_0=30.)#.to(device)

        self.attributes = []
        for ith_attri in self.attributes_names:
            self.attributes.append(ith_attri)
            if backbone == 'mlp':
                self.net_direction[ith_attri] = BaseDeepSDF(in_features=self.in_features,
                                                            latent_size=self.latent_size + 1,
                                                            out_features=in_features, # 2 or 3
                                                            num_hidden_layers=hidden_layers,
                                                            hidden_features=hidden_features,
                                                            latent_in=[hidden_layers // 2],
                                                            outermost_linear=True,
                                                            nonlinearity='relu',
                                                            weight_init=None)#.to(device)
            elif backbone == 'siren':

                self.net_direction[ith_attri] = BaseDeepSDFSiren(in_features=self.in_features,
                                                                 latent_size=latent_size + 1,
                                                                 hidden_features=hidden_features,
                                                                 hidden_layers=hidden_layers,
                                                                 out_features=in_features, # 2 or 3
                                                                 latent_in=[hidden_layers // 2],
                                                                 outermost_linear=True,
                                                                 first_omega_0=30.,
                                                                 hidden_omega_0=30.)#.to(device)



    def get_displacement_per_covariate(self,
                                       enc_coords,
                                       embedding,
                                       covaraite,
                                       which_covariate,
                                       whether_inv=False,
                                       whether_augment=False,
                                       training=True):

        embedding_with_attri = torch.cat((covaraite, embedding), dim=-1)
        #neg_embedding = -embedding
        #neg_embedding_attri = torch.cat((covaraite, neg_embedding), dim=-1)
        if training:
            embedding_with_random = torch.cat((torch.randn_like(covaraite), embedding), dim=-1)
            embedding_with_zeros = torch.cat((torch.zeros_like(covaraite), embedding), dim=-1)
        else:
            embedding_with_zeros = torch.cat((torch.zeros_like(covaraite), embedding), dim=-1)

        if whether_augment:
            if not whether_inv:
                arr_disp_step1 = self.net_direction[which_covariate](embedding_with_attri, enc_coords) - self.net_direction[
                    which_covariate](embedding_with_random, enc_coords)

                arr_disp_step2 = self.net_direction[which_covariate](embedding_with_random, enc_coords+arr_disp_step1) - self.net_direction[
                    which_covariate](embedding_with_zeros, enc_coords+arr_disp_step1)
                arr_disp = arr_disp_step1 + arr_disp_step2
            else:
                arr_disp = self.net_direction[which_covariate](embedding_with_zeros, enc_coords) - self.net_direction[
                    which_covariate](embedding_with_attri, enc_coords)


        else:
            if not whether_inv:
                arr_disp = self.net_direction[which_covariate](embedding_with_attri, enc_coords) - self.net_direction[which_covariate](embedding_with_zeros, enc_coords)
            else:
                arr_disp = self.net_direction[which_covariate](embedding_with_zeros, enc_coords) - self.net_direction[which_covariate](embedding_with_attri, enc_coords)
        return arr_disp




    def get_displacement_per_covariate_sampling_inv(self, enc_coords, embedding, covaraite, which_covariate, whether_augment=True):

        embedding_with_attri = torch.cat((covaraite, embedding), dim=-1)


        embedding_with_random1 = torch.cat((torch.randn_like(covaraite), embedding), dim=-1)
        embedding_with_zeros = torch.cat((torch.zeros_like(covaraite), embedding), dim=-1)

        if self.pos_enc:
            arr_disp = self.net_direction[which_covariate](embedding_with_attri, self.pos_encoder(enc_coords)) - self.net_direction[
                which_covariate](embedding_with_random1, self.pos_encoder(enc_coords))
            arr_disp_inv = self.net_direction[which_covariate](embedding_with_random1, self.pos_encoder(enc_coords + arr_disp)) - \
                           self.net_direction[which_covariate](embedding_with_attri, self.pos_encoder(enc_coords + arr_disp))

        else:
            if whether_augment:
                arr_disp = self.net_direction[which_covariate](embedding_with_attri, enc_coords) - self.net_direction[which_covariate](embedding_with_random1, enc_coords)
                arr_disp_inv = self.net_direction[which_covariate](embedding_with_random1, enc_coords + arr_disp) - \
                               self.net_direction[which_covariate](embedding_with_attri, enc_coords + arr_disp)
            else:
                arr_disp = self.net_direction[which_covariate](embedding_with_attri, enc_coords) - self.net_direction[which_covariate](embedding_with_zeros, enc_coords)
                arr_disp_inv = self.net_direction[which_covariate](embedding_with_zeros, enc_coords + arr_disp) - self.net_direction[which_covariate](embedding_with_attri, enc_coords + arr_disp)


        return arr_disp, arr_disp_inv



    def evolution(self, coords, attributes, embedding, which_attribute):
        coords_init = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        for ith_attri in self.attributes_names:
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
        vf = torch.zeros_like(coords)
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

        attributes = attributes_1

        coords_init = coords.requires_grad_(True)  # allows to take derivative w.r.t. input

        list_attributes = []

        # divider = 0.0
        for ith_attri in self.attributes_names:
            list_attributes.append(attributes[ith_attri][:, None])
        arr_attri = torch.cat(list_attributes, dim=-1)
        arr_attri = arr_attri.repeat(1, coords.shape[1], 1).float()

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
        vf = torch.zeros_like(coords)
        dict_vf = {}
        dict_disentangle = {}
        for ith_key in range(len(self.attributes)):
            current_vf = self.get_displacement_per_covariate(enc_coords, embedding, arr_attri[..., [ith_key]],
                                                                      self.attributes[ith_key], training=training)

            # if self.pos_enc:
            #     current_vf_inv = self.get_displacement_per_covariate(self.pos_encoder(coords_init+current_vf), embedding, arr_attri[..., [ith_key]],
            #                                                               self.attributes[ith_key], whether_inv=True, training=training)
            # else:
            #     current_vf_inv = self.get_displacement_per_covariate(coords_init+current_vf, embedding, arr_attri[..., [ith_key]],
            #                                                               self.attributes[ith_key], whether_inv=True, training=training)


            dict_vf[self.attributes[ith_key]] = current_vf
            #dict_vf[self.attributes[ith_key] + '_inv'] = current_vf_inv
            vf += current_vf

            if self.whether_sampling:
                # inverse consistency regularization
                dict_vf[self.attributes[ith_key] + '_sampled_inv_forward'], dict_vf[self.attributes[ith_key] + '_sampled_inv_backward']\
                    = self.get_displacement_per_covariate_sampling_inv(coords_init, embedding, arr_attri[..., [ith_key]], self.attributes[ith_key])


        dict_vf['overall'] = vf
        if self.pos_enc:
            output = self.template(vf + coords_init)
        else:
            pts_on_template = vf + coords_init
            output = self.template(vf + coords_init)

        '''
        wihtout embedding as prior
        '''

        vf_wo_z = torch.zeros_like(coords)
        zero_padding = torch.zeros_like(embedding)
        for ith_key in range(len(self.attributes)):
            current_vf = self.get_displacement_per_covariate(enc_coords, zero_padding, arr_attri[..., [ith_key]],
                                                                      self.attributes[ith_key], training=training)
            # if self.pos_enc:
            #     current_vf_inv = self.get_displacement_per_covariate(self.pos_encoder(coords_init + current_vf), zero_padding,
            #                                                          arr_attri[..., [ith_key]],
            #                                                          self.attributes[ith_key], whether_inv=True, training=training)
            # else:
            #     current_vf_inv = self.get_displacement_per_covariate(coords_init + current_vf, zero_padding,
            #                                                          arr_attri[..., [ith_key]],
            #                                                          self.attributes[ith_key], whether_inv=True, training=training)

            dict_vf[self.attributes[ith_key] + '_z_padding'] = current_vf
            #dict_vf[self.attributes[ith_key] + '_z_padding' + '_inv'] = current_vf_inv
            vf_wo_z += current_vf

            # # inverse consistency regularization
            # dict_vf[self.attributes[ith_key] + '_z_padding' + '_sampled_inv_forward'], dict_vf[self.attributes[ith_key] + '_z_padding' + '_sampled_inv_backward']\
            #     = self.get_displacement_per_covariate_sampling_inv(coords_init, zero_padding, arr_attri[..., [ith_key]], self.attributes[ith_key])


        dict_vf['overall_z_padding'] = vf_wo_z
        if self.pos_enc:
            output_z_padding = self.template(vf_wo_z + coords_init)
        else:
            output_z_padding = self.template(vf_wo_z + coords_init)

        # get map
        model_output = {'model_in': coords_init,
                        'all_input': coords_init,
                        'model_out': output,
                        'vec_fields': dict_vf,
                        'model_out_z_padding': output_z_padding,
                        'disentangle': dict_disentangle,
                        'covariates': arr_attri,
                        'embedding': embedding,
                        'template': self.template_sdf(coords_init),
                        "pts_on_template": pts_on_template,
                        }

        return model_output  # , coords


    def template_sdf(self, coords):
        if self.pos_enc:
            return self.template(coords)
        else:
            return self.template(coords)


    def inv_transorm(self, coords, attributes_1, embedding, whether_inv=True):
        attributes = {}
        attributes = attributes_1


        coords_init = coords.requires_grad_(True)  # allows to take derivative w.r.t. input

        list_attributes = []

        for ith_attri in self.attributes_names:
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

        svf = torch.zeros_like(coords)
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

        for ith_attri in self.attributes_names:
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
        for ith_attri in self.attributes_names:
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
        vf = torch.zeros(coords)
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

        vf = torch.zeros(coords)
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