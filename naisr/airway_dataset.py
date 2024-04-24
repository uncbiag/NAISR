#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pandas as pd
import pyvista as pv
import progressbar as pb
from multiprocessing import *
import blosc


class PediatricAirwayCSAValueDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes # ['weight', 'age', 'height', 'sex', 'pos']

        self.split = self.load_yaml_as_dict(filename_split)[split]
        self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.valid_pos, self.valid_features, self.valid_csa_values, self.valid_ids = self.read_data(self.split)
        self.train_valid_pos, self.train_valid_features, self.train_valid_csa_values, self.train_valid_ids = self.read_data(self.train_split)

        self.mean = torch.tensor(self.train_valid_csa_values).mean().float()
        self.std = torch.tensor(self.train_valid_csa_values).std().float()
    def __len__(self):
        return len(self.valid_features)

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].isin(split)]
        #df_data_split = df_data_split.loc[df_data_split['pos']>0.3372]
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)

        features = np.array(list_attributes).T
        csa_values = np.array(df_data_split['csa'])
        pos_values = np.array(df_data_split['pos'])
        id_values = np.array(df_data_split['id'])
        valid_csa_values = csa_values[~np.isnan(csa_values)]
        valid_pos_values = pos_values[~np.isnan(csa_values)]
        valid_features = features[~np.isnan(csa_values)]
        valid_ids = id_values[~np.isnan(csa_values)]
        for ith_col in range(len(self.attributes)):
            #mean_v = valid_features[:, ith_col][~np.isnan(valid_features[:, ith_col])].mean()
            #valid_features[:, ith_col][np.isnan(valid_features[:, ith_col])] = mean_v
            valid_features = valid_features[~np.isnan(valid_features[:, ith_col])]

        return valid_pos_values[:, None], valid_features, valid_csa_values, valid_ids


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        attributes = {}
        for ith_attri in range(len(self.attributes)):
            #if self.attributes[ith_attri] != 'sex':
            #    a = torch.tensor(self.valid_features[idx][ith_attri])- self.train_valid_features[:, ith_attri].mean()
            #    attributes[self.attributes[ith_attri]] = (a / self.train_valid_features[:, ith_attri].std()).float()
            #else:
            attributes[self.attributes[ith_attri]] = torch.tensor(self.valid_features[idx][ith_attri])
        sdf = torch.tensor(self.valid_csa_values[idx]).float()

        sdf = (sdf - torch.tensor(self.train_valid_csa_values).mean()).float() / torch.tensor(self.train_valid_csa_values).std().float()
        gt = {'sdf': sdf, 'id': self.valid_ids[idx]}
        return torch.from_numpy(self.valid_pos[idx]).float(), attributes, gt


class PediatricAirwayCSAValueDatasetwithNAN(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes # ['weight', 'age', 'height', 'sex', 'pos']

        self.split = self.load_yaml_as_dict(filename_split)[split]
        #self.split = ['1035']
        self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.valid_pos, self.valid_features, self.valid_csa_values, self.valid_ids = self.read_data(self.split)
        self.train_valid_pos, self.train_valid_features, self.train_valid_csa_values, self.train_valid_ids = self.read_data(self.train_split)

        self.mean = torch.tensor(self.train_valid_csa_values).mean().float()
        self.std = torch.tensor(self.train_valid_csa_values).std().float()
    def __len__(self):
        return len(self.valid_features)

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].isin(split)]
        #df_data_split = df_data_split.loc[df_data_split['pos']>0.3372]
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)

        features = np.array(list_attributes).T
        csa_values = np.array(df_data_split['csa'])
        pos_values = np.array(df_data_split['pos'])
        id_values = np.array(df_data_split['id'])
        valid_csa_values = csa_values[~np.isnan(csa_values)]
        valid_pos_values = pos_values[~np.isnan(csa_values)]
        valid_features = features[~np.isnan(csa_values)]
        valid_ids = id_values[~np.isnan(csa_values)]
        for ith_col in range(len(self.attributes)):
            #mean_v = valid_features[:, ith_col][~np.isnan(valid_features[:, ith_col])].mean()
            #valid_features[:, ith_col][np.isnan(valid_features[:, ith_col])] = mean_v
            valid_features = valid_features[~np.isnan(valid_features[:, ith_col])]

        return valid_pos_values[:, None], valid_features, valid_csa_values, valid_ids


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        attributes = {}
        for ith_attri in range(len(self.attributes)):
            #if self.attributes[ith_attri] != 'sex':
            #    a = torch.tensor(self.valid_features[idx][ith_attri])- self.train_valid_features[:, ith_attri].mean()
            #    attributes[self.attributes[ith_attri]] = (a / self.train_valid_features[:, ith_attri].std()).float()
            #else:
            attributes[self.attributes[ith_attri]] = torch.tensor(self.valid_features[idx][ith_attri])
        sdf = torch.tensor(self.valid_csa_values[idx]).float()

        sdf = (sdf - torch.tensor(self.train_valid_csa_values).mean()).float() / torch.tensor(self.train_valid_csa_values).std().float()
        gt = {'sdf': sdf, 'id': self.valid_ids[idx]}
        return torch.from_numpy(self.valid_pos[idx]).float(), attributes, gt



class PediatricAirway2DCrossSectionDatasetwithNAN(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes + ['depth'] # ['weight', 'age', 'height', 'sex', 'pos']

        self.split = self.load_yaml_as_dict(filename_split)[split]
        self.split = ['1032', '1035', '1036', '1041', '1042', '1043', '1045', '1047', '1050', '1057']
        self.template_split = ['1032', ]
        #self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.csa2d, self.ids, self.ctl = self.read_data(self.split)
        self.template_covariates, self.template_csa2d, self.template_ids, self.template_ctl = self.read_data(self.template_split)
        #self.train_covariates, self.train_csa2d, self.train_ids, self.train_ctl = self.read_data(self.train_split)


    def __len__(self):
        return len(self.ids)

    def get_depthwise_template(self, depth):
        depths = np.array(self.template_covariates[:, -1])#.numpy()

        current_cov = self.template_covariates[np.abs(depths - depth)<0.003][:, 0:-1]
        try:
            current_csa2d = self.template_csa2d[np.abs(depths - depth)<0.003][0]
        except:
            print(np.abs(depth - depths).min())
            #print(self.template_csa2d)
        current_id = self.template_ids[np.abs(depths - depth)<0.003][0]
        current_ctl = self.template_ctl[np.abs(depths - depth)<0.003][0]

        return current_cov, current_csa2d, current_id, current_ctl

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.3372]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.49]

        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        csa2d_values = np.array(df_data_split['2dcsa'])
        id_values = np.array(df_data_split['id'])
        ctl_values = np.array(df_data_split['ctl'])
        return features, csa2d_values, id_values, ctl_values


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        #idx = 100
        #print(idx)
        #print(self.covariates[idx])
        attributes = {}
        for ith_attri in range(len(self.attributes)):
            if self.attributes[ith_attri] != 'depth':
                attributes[self.attributes[ith_attri]] = torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].min()) /  (self.covariates[:, ith_attri].max() - self.covariates[:, ith_attri].min() )]) #torch.tensor([self.covariates[idx][ith_attri]]) #torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].mean()) / self.covariates[:, ith_attri].std()])
                #attributes[self.attributes[ith_attri]] = torch.tensor([0.])
            else:
                attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]])

        #torch.tensor([self.covariates[idx][ith_attri]])
        DEPTH = float(attributes['depth'][0])
        #
        arr_2dcsa = np.load(self.csa2d[idx])[:, [0, 1]] / 10
        arr_normals = np.load(self.csa2d[idx])[:, [3, 4]]


        sampled_idx = np.random.randint(0, len(arr_2dcsa), 256)
        arr_2dcsa = torch.from_numpy(arr_2dcsa[sampled_idx]).float()
        local_surface = arr_2dcsa #+ 0.01 * torch.randn_like(arr_2dcsa)
        noise = torch.randn_like(arr_2dcsa)
        #noise[:, -1] = 0
        global_samples = 1*(noise * 2) + arr_2dcsa
        samples = torch.cat((local_surface, global_samples), dim=-2)

        arr_normals = torch.from_numpy(arr_normals[sampled_idx]).float()
        global_normals = torch.ones_like(arr_normals).float()
        arr_normals = torch.cat((arr_normals, global_normals), dim=-2)

        arr_depth = (attributes['depth'][None, :] - 0.5) * 2
        arr_depth = arr_depth.repeat(512, 1).float()
        samples = torch.cat((samples, arr_depth), axis=-1)
        attributes.pop('depth', None)

        sdf_local = torch.zeros((local_surface.shape[0], 1))
        sdf_global = torch.ones((global_samples.shape[0], 1)) * (-1)
        sdf = torch.cat((sdf_local, sdf_global), dim=-2)



        gt = {'sdf': sdf, 'id': self.ids[idx], 'normal': arr_normals, 'ctl_path': self.ctl[idx]}

        #print(np.isnan(arr_depth))
        #return torch.from_numpy(self.depth[idx]).float(), attributes, arr_2dcsa


        '''
        '''
        tmeplate_cov, template_csa2d, template_id, template_ctl = self.get_depthwise_template(DEPTH)
        arr_template_2dcsa = np.load(template_csa2d)[:, [0, 1]] / 10
        arr_template_normals = np.load(template_csa2d)[:, [3, 4]]


        sampled_idx = np.random.randint(0, len(arr_template_2dcsa), 256)
        arr_template_2dcsa = torch.from_numpy(arr_template_2dcsa[sampled_idx]).float()
        local_surface = arr_template_2dcsa #+ 0.01 * torch.randn_like(arr_2dcsa)
        #global_samples = 4 * (torch.rand_like(arr_template_2dcsa) - 0.5) + arr_template_2dcsa

        noise = torch.randn_like(arr_template_2dcsa)
        #noise[:, -1] = 0
        global_samples = 1*(noise * 2) + arr_template_2dcsa

        template_samples = torch.cat((local_surface, global_samples), dim=-2)

        arr_template_normals = torch.from_numpy(arr_template_normals[sampled_idx]).float()
        global_normals = torch.ones_like(arr_template_normals).float()
        arr_template_normals = torch.cat((arr_template_normals, global_normals), dim=-2)

        arr_template_depth = arr_depth
        template_samples = torch.cat((template_samples, arr_template_depth), axis=-1)

        sdf_local = torch.zeros((local_surface.shape[0], 1))
        sdf_global = torch.ones((global_samples.shape[0], 1)) * (-1)
        template_sdf = torch.cat((sdf_local, sdf_global), dim=-2)

        template_gt = {'template_sdf': template_sdf, 'template_id': self.template_ids[0], 'template_normal': arr_template_normals, 'template_ctl_path': template_ctl}
        gt.update(template_gt)
        return samples.float(), template_samples.float(), attributes, gt


class PediatricAirway2DCSADataset_testing(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes + ['depth'] # ['weight', 'age', 'height', 'sex', 'pos']

        self.split = self.load_yaml_as_dict(filename_split)[split]
        #self.split = ['1032',]
        #self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.csa2d, self.ids, self.ctl = self.read_data(self.split)
        #self.train_covariates, self.train_csa2d, self.train_ids, self.train_ctl = self.read_data(self.train_split)
        #self.unique_ids = np.unqiue(self.ids)

    def __len__(self):
        return len(self.ids)

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.49]
        id_values = np.array(df_data_split['id'])
        unique_id_values = np.unique(id_values)

        list_unique_features = []
        list_unique_csa2d_values = {}
        list_unique_ctl_values = []
        for ith_id in unique_id_values:
            list_unique_csa2d_values[ith_id] = []
            current_id_data =  df_data_split.loc[df_data_split['id'].astype('str').isin([str(ith_id),])]

            # read covariates
            list_attributes = []
            for ith_attribute in self.attributes:
                arr_current_attribute = np.array(current_id_data.iloc[0][ith_attribute])
                list_attributes.append(arr_current_attribute)
            current_features = np.array(list_attributes).T
            list_unique_features.append(current_features)

            # read target samples of the shape
            list_unique_ctl_values.append(current_id_data.iloc[0]['ctl'])
            for i in range(len(current_id_data)):
                list_unique_csa2d_values[ith_id].append(current_id_data.iloc[i]['2dcsa'])

        unique_features = np.array(list_unique_features)
        #unique_csa2d_values = np.array(list_unique_csa2d_values)
        return unique_features, list_unique_csa2d_values, unique_id_values, list_unique_ctl_values


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        #idx = 100
        #print(idx)
        #print(self.covariates[idx])
        current_id = self.ids[idx]
        list_current_csa2d = self.csa2d[current_id]
        attributes = {}
        #for ith_attri in range(len(self.attributes)):
        #    attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]]) #torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].mean()) / self.covariates[:, ith_attri].std()])
        #torch.tensor([self.covariates[idx][ith_attri]])
        for ith_attri in range(len(self.attributes)):
            if self.attributes[ith_attri] != 'depth':
                attributes[self.attributes[ith_attri]] = torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].min()) /  (self.covariates[:, ith_attri].max() - self.covariates[:, ith_attri].min() )]) #torch.tensor([self.covariates[idx][ith_attri]]) #
                #attributes[self.attributes[ith_attri]] = torch.tensor([0.])
            else:
                attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]])


        '''
        list_arr_2dcsa = []
        for ith_2dcsa in list_current_csa2d:                                                               #
            list_arr_2dcsa.append(np.load(list_current_csa2d[ith_2dcsa])[:, [0, 1]])
        samples = torch.from_numpy(np.concatenate(list_arr_2dcsa, axis=0))
        '''

        gt = {'csa2d': list_current_csa2d, 'id': current_id,  'ctl_path': self.ctl[idx]}

        #print(np.isnan(arr_depth))
        #return torch.from_numpy(self.depth[idx]).float(), attributes, arr_2dcsa
        return attributes, gt


class PediatricAirway2DCSATemplateDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes + ['depth'] # ['weight', 'age', 'height', 'sex', 'pos']

        self.split = self.load_yaml_as_dict(filename_split)[split]
        self.split = ['1032',]
        self.template_split = ['1032', ]
        #self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.csa2d, self.ids, self.ctl = self.read_data(self.split)
        self.template_covariates, self.template_csa2d, self.template_ids, self.template_ctl = self.read_data(self.template_split)
        #self.train_covariates, self.train_csa2d, self.train_ids, self.train_ctl = self.read_data(self.train_split)


    def __len__(self):
        return len(self.template_ids)

    def get_depthwise_template(self, depth):
        depths = np.array(self.template_covariates[:, -1])#.numpy()

        current_cov = self.template_covariates[np.abs(depths - depth)<0.003][:, 0:-1]
        try:
            current_csa2d = self.template_csa2d[np.abs(depths - depth)<0.003][0]
        except:
            print(np.abs(depth - depths).min())
            #print(self.template_csa2d)
        current_id = self.template_ids[np.abs(depths - depth)<0.003][0]
        current_ctl = self.template_ctl[np.abs(depths - depth)<0.003][0]

        return current_cov, current_csa2d, current_id, current_ctl

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.3372]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.49]

        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        csa2d_values = np.array(df_data_split['2dcsa'])
        id_values = np.array(df_data_split['id'])
        ctl_values = np.array(df_data_split['ctl'])
        return features, csa2d_values, id_values, ctl_values


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        #idx = 100
        #print(idx)
        #print(self.covariates[idx])
        attributes = {}
        for ith_attri in range(len(self.attributes)):
            if self.attributes[ith_attri] != 'depth':
                attributes[self.attributes[ith_attri]] = torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].min()) /  (self.covariates[:, ith_attri].max() - self.covariates[:, ith_attri].min() )]) #torch.tensor([self.covariates[idx][ith_attri]]) #torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].mean()) / self.covariates[:, ith_attri].std()])
                #attributes[self.attributes[ith_attri]] = torch.tensor([0.])
            else:
                attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]])

        #torch.tensor([self.covariates[idx][ith_attri]])
        DEPTH = float(attributes['depth'][0])
        #
        arr_2dcsa = np.load(self.csa2d[idx])[:, [0, 1]] / 10
        arr_normals = np.load(self.csa2d[idx])[:, [3, 4]]


        sampled_idx = np.random.randint(0, len(arr_2dcsa), 256)
        arr_2dcsa = torch.from_numpy(arr_2dcsa[sampled_idx]).float()
        local_surface = arr_2dcsa #+ 0.01 * torch.randn_like(arr_2dcsa)
        noise = torch.rand_like(arr_2dcsa)
        #noise[:, -1] = 0
        global_samples = 1*(noise - 0.5) + arr_2dcsa
        samples = torch.cat((local_surface, global_samples), dim=-2)

        arr_normals = torch.from_numpy(arr_normals[sampled_idx]).float()
        global_normals = torch.ones_like(arr_normals).float()
        arr_normals = torch.cat((arr_normals, global_normals), dim=-2)

        arr_depth = (attributes['depth'][None, :] - 0.5) * 2
        arr_depth = arr_depth.repeat(512, 1).float()
        samples = torch.cat((samples, arr_depth), axis=-1)
        attributes.pop('depth', None)

        sdf_local = torch.zeros((local_surface.shape[0], 1))
        sdf_global = torch.ones((global_samples.shape[0], 1)) * (-1)
        sdf = torch.cat((sdf_local, sdf_global), dim=-2)



        gt = {'sdf': sdf, 'id': self.ids[idx], 'normal': arr_normals, 'ctl_path': self.ctl[idx]}

        #print(np.isnan(arr_depth))
        #return torch.from_numpy(self.depth[idx]).float(), attributes, arr_2dcsa


        '''
        '''
        tmeplate_cov, template_csa2d, template_id, template_ctl = self.get_depthwise_template(DEPTH)
        arr_template_2dcsa = np.load(template_csa2d)[:, [0, 1]] / 10
        arr_template_normals = np.load(template_csa2d)[:, [3, 4]]


        sampled_idx = np.random.randint(0, len(arr_template_2dcsa), 256)
        arr_template_2dcsa = torch.from_numpy(arr_template_2dcsa[sampled_idx]).float()
        local_surface = arr_template_2dcsa #+ 0.01 * torch.randn_like(arr_2dcsa)
        #global_samples = 4 * (torch.rand_like(arr_template_2dcsa) - 0.5) + arr_template_2dcsa

        noise = torch.rand_like(arr_template_2dcsa)
        #noise[:, -1] = 0
        global_samples = 1*(noise - 0.5) + arr_template_2dcsa

        template_samples = torch.cat((local_surface, global_samples), dim=-2)

        arr_template_normals = torch.from_numpy(arr_template_normals[sampled_idx]).float()
        global_normals = torch.ones_like(arr_template_normals).float()
        arr_template_normals = torch.cat((arr_template_normals, global_normals), dim=-2)

        arr_template_depth = arr_depth
        template_samples = torch.cat((template_samples, arr_template_depth), axis=-1)

        sdf_local = torch.zeros((local_surface.shape[0], 1))
        sdf_global = torch.ones((global_samples.shape[0], 1)) * (-1)
        template_sdf = torch.cat((sdf_local, sdf_global), dim=-2)

        template_gt = {'template_sdf': template_sdf, 'template_id': self.template_ids[0], 'template_normal': arr_template_normals, 'template_ctl_path': template_ctl}
        gt.update(template_gt)
        return samples.float(), template_samples.float(), attributes, gt





class PediatricAirway2DCSADataset_1(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes + ['depth'] # ['weight', 'age', 'height', 'sex', 'pos']

        self.split = self.load_yaml_as_dict(filename_split)[split]
        #self.split = ['1032',]
        self.template_split = ['1032', ]
        #self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.csa2d, self.ids, self.ctl = self.read_data(self.split)
        self.template_covariates, self.template_csa2d, self.template_ids, self.template_ctl = self.read_data(self.template_split)
        #self.train_covariates, self.train_csa2d, self.train_ids, self.train_ctl = self.read_data(self.train_split)


    def __len__(self):
        return len(self.ids)

    def get_depthwise_template(self, depth):
        depths = np.array(self.template_covariates[:, -1])#.numpy()

        current_cov = self.template_covariates[np.abs(depths - depth)<0.003][:, 0:-1]
        try:
            current_csa2d = self.template_csa2d[np.abs(depths - depth)<0.003][0]
        except:
            print(np.abs(depth - depths).min())
            #print(self.template_csa2d)
        current_id = self.template_ids[np.abs(depths - depth)<0.003][0]
        current_ctl = self.template_ctl[np.abs(depths - depth)<0.003][0]

        return current_cov, current_csa2d, current_id, current_ctl

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.3372]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.49]

        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        csa2d_values = np.array(df_data_split['2dcsa'])
        id_values = np.array(df_data_split['id'])
        ctl_values = np.array(df_data_split['ctl'])
        return features, csa2d_values, id_values, ctl_values


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        #idx = 100
        #print(idx)
        #print(self.covariates[idx])
        attributes = {}
        for ith_attri in range(len(self.attributes)):
            if self.attributes[ith_attri] != 'depth':
                attributes[self.attributes[ith_attri]] = torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].min()) /  (self.covariates[:, ith_attri].max() - self.covariates[:, ith_attri].min() )]) #torch.tensor([self.covariates[idx][ith_attri]]) #torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].mean()) / self.covariates[:, ith_attri].std()])
                #attributes[self.attributes[ith_attri]] = torch.tensor([0.])
            else:
                attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]])

        #torch.tensor([self.covariates[idx][ith_attri]])
        DEPTH = float(attributes['depth'][0])
        #
        arr_2dcsa = np.load(self.csa2d[idx])[:, [0, 1]] / 10
        arr_normals = np.load(self.csa2d[idx])[:, [3, 4]]
        arr_normals = arr_normals / np.linalg.norm(arr_normals, axis=-1)[:, None]


        sampled_idx = np.random.randint(0, len(arr_2dcsa), 256)
        arr_2dcsa = torch.from_numpy(arr_2dcsa[sampled_idx]).float()
        arr_normals = torch.from_numpy(arr_normals[sampled_idx]).float()
        local_surface = arr_2dcsa #+ 0.01 * torch.randn_like(arr_2dcsa)
        noise = torch.rand((arr_2dcsa.shape[0], 1,))
        noise= 1*(noise - 0.5)



        #noise[:, -1] = 0
        global_samples = noise * arr_normals + arr_2dcsa
        samples = global_samples #torch.cat((local_surface, global_samples), dim=-2)
        #global_normals = torch.ones_like(arr_normals).float()
        #arr_normals = torch.cat((arr_normals, global_normals), dim=-2)

        arr_depth = (attributes['depth'][None, :] - 0.5) * 2
        arr_depth = arr_depth.repeat(256, 1).float()
        samples = torch.cat((samples, arr_depth), axis=-1)
        samples = torch.cat((samples, noise), axis=-1)
        attributes.pop('depth', None)

        sdf_local = torch.zeros((local_surface.shape[0], 1))
        sdf_global = torch.ones((global_samples.shape[0], 1)) * (-1)
        sdf = sdf_global #torch.cat((sdf_local, sdf_global), dim=-2)



        gt = {'sdf': sdf, 'id': self.ids[idx], 'normal': arr_normals, 'ctl_path': self.ctl[idx]}

        #print(np.isnan(arr_depth))
        #return torch.from_numpy(self.depth[idx]).float(), attributes, arr_2dcsa


        '''
        
        tmeplate_cov, template_csa2d, template_id, template_ctl = self.get_depthwise_template(DEPTH)
        arr_template_2dcsa = np.load(template_csa2d)[:, [0, 1]] / 10
        arr_template_normals = np.load(template_csa2d)[:, [3, 4]]


        sampled_idx = np.random.randint(0, len(arr_template_2dcsa), 256)
        arr_template_2dcsa = torch.from_numpy(arr_template_2dcsa[sampled_idx]).float()
        local_surface = arr_template_2dcsa #+ 0.01 * torch.randn_like(arr_2dcsa)
        #global_samples = 4 * (torch.rand_like(arr_template_2dcsa) - 0.5) + arr_template_2dcsa

        noise = torch.rand_like(arr_template_2dcsa)
        #noise[:, -1] = 0
        global_samples = 1*(noise - 0.5) + arr_template_2dcsa

        template_samples = torch.cat((local_surface, global_samples), dim=-2)

        arr_template_normals = torch.from_numpy(arr_template_normals[sampled_idx]).float()
        global_normals = torch.ones_like(arr_template_normals).float()
        arr_template_normals = torch.cat((arr_template_normals, global_normals), dim=-2)

        arr_template_depth = arr_depth
        template_samples = torch.cat((template_samples, arr_template_depth), axis=-1)

        sdf_local = torch.zeros((local_surface.shape[0], 1))
        sdf_global = torch.ones((global_samples.shape[0], 1)) * (-1)
        template_sdf = torch.cat((sdf_local, sdf_global), dim=-2)

        template_gt = {'template_sdf': template_sdf, 'template_id': self.template_ids[0], 'template_normal': arr_template_normals, 'template_ctl_path': template_ctl}
        gt.update(template_gt)
        
        return samples.float(), template_samples.float(), attributes, gt
        '''
        return samples.float(), attributes, gt, idx


class PediatricAirway2DCSADataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes + ['depth'] # ['weight', 'age', 'height', 'sex', 'pos']

        self.split = self.load_yaml_as_dict(filename_split)[split]
        #self.split = ['1032', '1035', ] #'1036', '1041', '1042']
        #self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.csa2d, self.ids, self.ctl = self.read_data(self.split)
        #self.train_covariates, self.train_csa2d, self.train_ids, self.train_ctl = self.read_data(self.train_split)
        #self.unique_ids = np.unqiue(self.ids)

    def __len__(self):
        return len(self.ids)

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        #df_data_split = df_data_split.loc[df_data_split['depth'] >0.5]
        id_values = np.array(df_data_split['id'])
        unique_id_values = np.unique(id_values)

        list_unique_features = []
        list_unique_csa2d_values = {}
        list_unique_ctl_values = []
        for ith_id in unique_id_values:
            list_unique_csa2d_values[ith_id] = []
            current_id_data =  df_data_split.loc[df_data_split['id'].astype('str').isin([str(ith_id),])]

            # read covariates
            list_attributes = []
            for ith_attribute in self.attributes:
                arr_current_attribute = np.array(current_id_data.iloc[0][ith_attribute])
                list_attributes.append(arr_current_attribute)
            current_features = np.array(list_attributes).T
            list_unique_features.append(current_features)

            # read target samples of the shape
            list_unique_ctl_values.append(current_id_data.iloc[0]['ctl'])
            for i in range(len(current_id_data)):
                list_unique_csa2d_values[ith_id].append(current_id_data.iloc[i]['2dcsa'])

        unique_features = np.array(list_unique_features)
        #unique_csa2d_values = np.array(list_unique_csa2d_values)
        return unique_features, list_unique_csa2d_values, unique_id_values, list_unique_ctl_values


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        #idx = 100
        #print(idx)
        #print(self.covariates[idx])
        current_id = self.ids[idx]
        list_current_csa2d = self.csa2d[current_id]
        attributes = {}
        #for ith_attri in range(len(self.attributes)):
        #    attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]]) #torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].mean()) / self.covariates[:, ith_attri].std()])
        #torch.tensor([self.covariates[idx][ith_attri]])
        for ith_attri in range(len(self.attributes)):
            if self.attributes[ith_attri] != 'depth':
                attributes[self.attributes[ith_attri]] = torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].min()) /  (self.covariates[:, ith_attri].max() - self.covariates[:, ith_attri].min() )]) #torch.tensor([self.covariates[idx][ith_attri]]) #
                #attributes[self.attributes[ith_attri]] = torch.tensor([0.])
            else:
                attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]])



        list_arr_2dcsa = []
        list_arr_normal = []
        for ith_2dcsa in range(len(list_current_csa2d)):                                                               #

            arr_2dcsa = torch.from_numpy(np.load(list_current_csa2d[ith_2dcsa])[:, [0, 1]] / 10)
            arr_depth = (attributes['depth'][None, :] - 0.5) * 2
            arr_depth = arr_depth.repeat(arr_2dcsa.shape[0], 1).float()
            arr_2dcsa = torch.cat((arr_2dcsa, arr_depth), axis=-1)

            list_arr_2dcsa.append(arr_2dcsa)
            list_arr_normal.append(np.load(list_current_csa2d[ith_2dcsa])[:, [3, 4]])

        arr_samples = torch.from_numpy(np.concatenate(list_arr_2dcsa, axis=0))
        arr_normals = torch.from_numpy(np.concatenate(list_arr_normal, axis=0))
        arr_normals = arr_normals / np.linalg.norm(arr_normals, axis=-1)[:, None]

        sampled_idx = np.random.randint(0, len(arr_samples), 10000)
        arr_2dcsa = arr_samples[sampled_idx].float()
        arr_normals = arr_normals[sampled_idx].float()
        arr_normals = torch.cat((arr_normals, torch.zeros((arr_normals.shape[0], 1))), dim=-1)
        noise = torch.randn((arr_2dcsa.shape[0], 1,)) * arr_2dcsa[:, [0,1]].std()
        #noise= 1*(noise - 0.5)

        samples = noise * arr_normals + arr_2dcsa


        samples = torch.cat((samples, noise), axis=-1)
        attributes.pop('depth', None)
        sdf = torch.ones((samples.shape[0], 1)) * (-1)

        gt = {'sdf': sdf, 'id': self.ids[idx], 'normal': arr_normals, 'ctl_path': self.ctl[idx]}

        return samples.float(), attributes, gt, idx



'''
class PediatricAirway3DShapeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes

        self.split = self.load_yaml_as_dict(filename_split)[split]
        #self.split = ['1032', '1035', '1036', '1041', '1042', '1043', '1045', '1047', '1050', '1057']
        self.template_split = ['1032', ]
        #self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.shape3d, self.sdf3d, self.ids, self.ctl = self.read_data(self.split)

    def __len__(self):
        return len(self.ids)


    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.3372]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.49]

        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        shape3d_values = np.array(df_data_split['3dshape'])
        sdf3d_values = np.array(df_data_split['3dsdf'])
        id_values = np.array(df_data_split['id'])
        ctl_values = np.array(df_data_split['ctl'])
        return features, shape3d_values, sdf3d_values, id_values, ctl_values


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        attributes = {}
        for ith_attri in range(len(self.attributes)):
            if self.attributes[ith_attri] != 'depth':
                attributes[self.attributes[ith_attri]] = torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].min()) /  (self.covariates[:, ith_attri].max() - self.covariates[:, ith_attri].min() )]) #torch.tensor([self.covariates[idx][ith_attri]]) #torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].mean()) / self.covariates[:, ith_attri].std()])
                #attributes[self.attributes[ith_attri]] = torch.tensor([0.])
            else:
                attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]])

        #
        arr_3dsdf = np.load(self.sdf3d[idx]) #[:, [0, 1, 2]] / 10
        sampled_idx = np.random.randint(0, arr_3dsdf.shape[0], 10000)
        arr_3dsdf = torch.from_numpy(arr_3dsdf[sampled_idx]).float()
        arr_3dsdf_points = arr_3dsdf[..., [0, 1, 2]]
        arr_3dsdf_normals = torch.zeros_like((arr_3dsdf_points))
        arr_3dsdf_sdf = arr_3dsdf[..., [3]]

        pv_3dshape = pv.read(self.shape3d[idx]) #[:, [0, 1]] / 10
        sampled_idx = np.random.randint(0, np.array(pv_3dshape.points).shape[0], 10000)
        arr_3dshape_normals = torch.from_numpy(np.array(pv_3dshape.point_normals[sampled_idx])) #np.load(self.csa2d[idx])[:, [3, 4]]
        arr_3dshape_points = torch.from_numpy(np.array(pv_3dshape.points[sampled_idx]))
        arr_3dshape_sdf = torch.zeros_like(arr_3dsdf_sdf)


        arr_points = torch.cat((arr_3dshape_points, arr_3dsdf_points), dim=0)
        arr_normals = torch.cat((arr_3dshape_normals, arr_3dsdf_normals), dim=0)
        arr_sdf = torch.cat((arr_3dshape_sdf, arr_3dsdf_sdf), dim=0)

        samples = torch.cat((arr_3dsdf_points, arr_3dsdf_sdf), dim=-1)
        #samples[:, [0, 1, 2]] = (samples[:, [0, 1, 2]] - torch.mean(samples[:, [0, 1, 2]], dim=0))
        #samples[:, [3]] /= 5
        gt = {'sdf': arr_sdf, 'id': self.ids[idx], 'normal': arr_normals, 'ctl_path': self.ctl[idx]}

        return samples.float(), attributes, gt, idx
'''

'''
class PediatricAirway3DShapeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes=['weight', 'age', 'sex', ],
            split='train',
    ):
        self.attributes = attributes
        self.num_of_workers = 8
        self.split = self.load_yaml_as_dict(filename_split)[split]
        self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.shape3d, self.sdf3d, self.ids, self.ctl = self.read_data(self.split)
        self.train_covariates, self.train_shape3d, self.train_sdf3d, self.train_ids, self.train_ctl = self.read_data(
            self.train_split)
        self.list_cases = []

    def __len__(self):
        return len(self.ids)

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]

        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        shape3d_values = np.array(df_data_split['3dshape_npy'])
        sdf3d_values = np.array(df_data_split['3dsdf'])
        id_values = np.array(df_data_split['id'].astype('str'))
        ctl_values = np.array(df_data_split['ctl'])
        return features, shape3d_values, sdf3d_values, id_values, ctl_values

    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):
        attributes = {}
        for ith_attri in range(len(self.attributes)):
            if self.attributes[ith_attri] != 'depth':
                attributes[self.attributes[ith_attri]] = (torch.tensor(
                    [self.covariates[idx][ith_attri]]) - self.train_covariates[:, ith_attri].min()) / (
                                                                     self.train_covariates[:,
                                                                     ith_attri].max() - self.train_covariates[:,
                                                                                        ith_attri].min())  # torch.tensor([self.covariates[idx][ith_attri]]) #torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].mean()) / self.covariates[:, ith_attri].std()])
                attributes[self.attributes[ith_attri]] = attributes[self.attributes[ith_attri]] * 2 - 1
            else:
                attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]])

        pv_3dshape = np.load(self.shape3d[idx])  # [:, [0, 1]] / 10
        sampled_idx = np.random.randint(0, np.array(pv_3dshape[:, [0, 1, 2]]).shape[0], 2000)  # 500)
        arr_3dshape_normals = torch.from_numpy(np.array(pv_3dshape[:, [3, 4, 5]][sampled_idx])) * (-1)
        arr_3dshape_points = torch.from_numpy(np.array(pv_3dshape[:, [0, 1, 2]][sampled_idx])).float() / 60

        #
        arr_3dsdf = np.load(self.sdf3d[idx])  # [:, [0, 1, 2]] / 10
        arr_3dsdf_sdf = arr_3dsdf[..., 3]
        arr_3dsdf_pos = arr_3dsdf[arr_3dsdf_sdf >= 0.05]
        # arr_3dsdf_neg = arr_3dsdf[arr_3dsdf_sdf < 0]
        sampled_idx_pos = np.random.randint(0, arr_3dsdf_pos.shape[0], 1000)  # 250)
        # sampled_idx_neg = np.random.randint(0, arr_3dsdf_neg.shape[0], 500)

        arr_3dsdf = arr_3dsdf_pos[
            sampled_idx_pos]  # np.concatenate((arr_3dsdf_pos[sampled_idx_pos], arr_3dsdf_neg[sampled_idx_neg]), axis=0)
        arr_3dsdf_points = torch.from_numpy(arr_3dsdf[..., [0, 1, 2]]).float() / 60
        # arr_3dsdf_points /= scale
        arr_3dsdf_sdf = torch.from_numpy(arr_3dsdf[..., [3]]) / 60
        arr_3dsdf_normals = torch.zeros_like((arr_3dsdf_points))
        arr_samples = torch.cat((arr_3dsdf_points, arr_3dshape_points), dim=-2)
        arr_normals = torch.cat((arr_3dsdf_normals, arr_3dshape_normals), dim=-2)

        sdf_local = torch.zeros((arr_3dshape_points.shape[0], 1))
        sdf = torch.cat((arr_3dsdf_sdf, sdf_local), dim=-2)

        gt = {'sdf': sdf, 'id': self.ids[idx], 'normal': arr_normals.float(), 'ctl_path': self.ctl[idx],
              'gt_path': self.shape3d[idx]}

        return arr_samples.float(), attributes, gt, idx
'''

class PediatricAirway3DShapeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes
        self.num_of_workers = 8

        self.split = self.load_yaml_as_dict(filename_split)[split]
        self.train_split = self.load_yaml_as_dict(filename_split)['train']
        #self.split = ['1067']
        self.filename_datasource = filename_datasource
        self.covariates, self.shape3d, self.sdf3d, self.ids, self.ctl = self.read_data(self.split)
        self.train_covariates, self.train_shape3d, self.train_sdf3d, self.train_ids, self.train_ctl = self.read_data(self.train_split)
        self.list_cases = []
        self.init_img_pool()

    def __len__(self):
        return len(self.ids)



    def split_ids(self, ids, split_num):
        ids_split = np.array_split(np.arange(len(ids)), split_num)
        return ids_split


    def init_img_pool(self):
        manager = Manager()
        pts_dic = manager.dict()

        split_ids = self.split_ids(self.split, self.num_of_workers)
        procs = []
        for i in range(self.num_of_workers):
            p = Process(target=self.read_data_into_zipnp, args=(split_ids[i], pts_dic))
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()
        print("the loading phase finished, total {} shape have been loaded".format(len(pts_dic)))
        for idx in self.ids:
            self.list_cases.append([pts_dic[idx]['points_on_surface'], pts_dic[idx]['points_off_surface']])


    def get_covariates_for_one_case(self, idx):
        covariates = {}
        # for ith_attri in range(len(self.attributes)):
        #     covariates[self.attributes[ith_attri]] = (torch.tensor([self.covariates[idx][ith_attri]]) - self.train_covariates[:, ith_attri].min()) / (self.train_covariates[:, ith_attri].max() - self.train_covariates[:,ith_attri].min())
        #     covariates[self.attributes[ith_attri]] = covariates[self.attributes[ith_attri]] * 2 - 1
        for ith_attri in range(len(self.attributes)):
            covariates[self.attributes[ith_attri]] = (torch.tensor([self.covariates[idx][ith_attri]]) - self.train_covariates[:, ith_attri].mean()) / self.train_covariates[:, ith_attri].std()

        return covariates

    def read_data_into_zipnp(self, ids, img_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(ids)).start()
        count = 0
        for idx in ids:
            dict_data_case = {}

            # get points on surface
            pv_3dshape = np.load(self.shape3d[idx])
            # get points off surface
            arr_3dsdf = np.load(self.sdf3d[idx])

            dict_data_case['points_on_surface'] = blosc.pack_array(pv_3dshape)
            dict_data_case['points_off_surface'] = blosc.pack_array(arr_3dsdf)

            img_dic[self.ids[idx]] = dict_data_case
            count += 1
            pbar.update(count)
        pbar.finish()

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        shape3d_values = np.array(df_data_split['3dshape_npy'])
        sdf3d_values = np.array(df_data_split['3dsdf'])
        id_values = np.array(df_data_split['id'].astype('str'))
        ctl_values = np.array(df_data_split['ctl'])
        return features, shape3d_values, sdf3d_values, id_values, ctl_values


    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):

        list_items = [blosc.unpack_array(item) for item in self.list_cases[idx]]
        arr_points_on_surface = torch.from_numpy(list_items[0]).float()
        arr_points_off_surface = torch.from_numpy(list_items[1]).float()


        sampled_idx = np.random.randint(0, np.array(arr_points_on_surface[:, [0, 1, 2]]).shape[0],500) #2000) #500
        arr_3dshape_normals = np.array(arr_points_on_surface[:, [3, 4, 5]][sampled_idx]) * (-1)
        arr_3dshape_points = np.array(arr_points_on_surface[:, [0, 1, 2]][sampled_idx]) / 60

        arr_3dsdf_sdf = arr_points_off_surface[..., 3]
        arr_3dsdf_off = arr_points_off_surface[np.abs(arr_3dsdf_sdf) >= 2.]
        sampled_idx_off = np.random.randint(0, arr_3dsdf_off.shape[0], 250) #1000)  # 250)#

        arr_3dsdf = arr_3dsdf_off[sampled_idx_off]
        arr_3dsdf_points = arr_3dsdf[..., [0, 1, 2]] / 60
        arr_3dsdf_sdf = arr_3dsdf[..., [3]] / 60
        arr_3dsdf_normals = np.zeros_like((arr_3dsdf_points))
        arr_samples = np.concatenate((arr_3dsdf_points, arr_3dshape_points), axis=-2)
        arr_normals = np.concatenate((arr_3dsdf_normals, arr_3dshape_normals), axis=-2)

        sdf_local = np.zeros((arr_3dshape_points.shape[0], 1))
        sdf = np.concatenate((arr_3dsdf_sdf, sdf_local), axis=-2)
        

        #gt = self.list_cases[idx][1][0]
        #covariates = self.list_cases[1][1]
        gt = {'id': self.ids[idx], 'ctl_path': self.ctl[idx], 'gt_path': self.shape3d[idx]}

        gt.update({'sdf': sdf})
        gt.update({'normal': arr_normals})
        # get covariates
        covariates = self.get_covariates_for_one_case(idx)

        return arr_samples, covariates, gt, idx


class PediatricAirway3DShapeNormalizedScale1Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes=['weight', 'age', 'sex', ],
            split='train',
    ):
        self.attributes = attributes
        self.num_of_workers = 8

        self.split = self.load_yaml_as_dict(filename_split)[split]
        self.train_split = self.load_yaml_as_dict(filename_split)['train']
        #self.split = ['1067']
        self.filename_datasource = filename_datasource
        self.covariates, self.shape3d, self.sdf3d, self.ids, self.ctl = self.read_data(self.split)
        self.train_covariates, self.train_shape3d, self.train_sdf3d, self.train_ids, self.train_ctl = self.read_data(
            self.train_split)
        self.list_cases = []
        self.init_img_pool()

    def __len__(self):
        return len(self.ids)

    def split_ids(self, ids, split_num):
        ids_split = np.array_split(np.arange(len(ids)), split_num)
        return ids_split

    def init_img_pool(self):
        manager = Manager()
        pts_dic = manager.dict()

        split_ids = self.split_ids(self.split, self.num_of_workers)
        procs = []
        for i in range(self.num_of_workers):
            p = Process(target=self.read_data_into_zipnp, args=(split_ids[i], pts_dic))
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()
        print("the loading phase finished, total {} shape have been loaded".format(len(pts_dic)))
        for idx in self.ids:
            self.list_cases.append([pts_dic[idx]['points_on_surface'], pts_dic[idx]['points_off_surface']])

    def get_covariates_for_one_case(self, idx):
        covariates = {}
        # for ith_attri in range(len(self.attributes)):
        #     covariates[self.attributes[ith_attri]] = (torch.tensor([self.covariates[idx][ith_attri]]) - self.train_covariates[:, ith_attri].min()) / (self.train_covariates[:, ith_attri].max() - self.train_covariates[:,ith_attri].min())
        #     covariates[self.attributes[ith_attri]] = covariates[self.attributes[ith_attri]] * 2 - 1
        for ith_attri in range(len(self.attributes)):
            covariates[self.attributes[ith_attri]] = (torch.tensor(
                [self.covariates[idx][ith_attri]]) - self.train_covariates[:,
                                                     ith_attri].mean()) / self.train_covariates[:, ith_attri].std()

        return covariates

    def read_data_into_zipnp(self, ids, img_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(ids)).start()
        count = 0
        for idx in ids:
            dict_data_case = {}

            # get points on surface
            pv_3dshape = np.load(self.shape3d[idx])
            # get points off surface
            arr_3dsdf = np.load(self.sdf3d[idx])

            dict_data_case['points_on_surface'] = blosc.pack_array(pv_3dshape)
            dict_data_case['points_off_surface'] = blosc.pack_array(arr_3dsdf)

            img_dic[self.ids[idx]] = dict_data_case
            count += 1
            pbar.update(count)
        pbar.finish()

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        shape3d_values = np.array(df_data_split['3dshape_npy'])
        sdf3d_values = np.array(df_data_split['3dsdf'])
        id_values = np.array(df_data_split['id'].astype('str'))
        ctl_values = np.array(df_data_split['ctl'])
        return features, shape3d_values, sdf3d_values, id_values, ctl_values

    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):

        list_items = [blosc.unpack_array(item) for item in self.list_cases[idx]]
        arr_points_on_surface = torch.from_numpy(list_items[0]).float()
        arr_points_off_surface = torch.from_numpy(list_items[1]).float()

        sampled_idx = np.random.randint(0, np.array(arr_points_on_surface[:, [0, 1, 2]]).shape[0], 500)  # 2000) #500
        arr_3dshape_normals = np.array(arr_points_on_surface[:, [3, 4, 5]][sampled_idx]) * (-1)
        arr_3dshape_points = np.array(arr_points_on_surface[:, [0, 1, 2]][sampled_idx]) / 60

        arr_3dsdf_sdf = arr_points_off_surface[..., 3]
        arr_3dsdf_off = arr_points_off_surface[np.abs(arr_3dsdf_sdf) >= 2.]
        sampled_idx_off = np.random.randint(0, arr_3dsdf_off.shape[0], 250)  # 1000)  # 250)#

        arr_3dsdf = arr_3dsdf_off[sampled_idx_off]
        arr_3dsdf_points = arr_3dsdf[..., [0, 1, 2]] / 60
        arr_3dsdf_sdf = arr_3dsdf[..., [3]] / 60
        arr_3dsdf_normals = np.zeros_like((arr_3dsdf_points))
        arr_samples = np.concatenate((arr_3dsdf_points, arr_3dshape_points), axis=-2)
        arr_normals = np.concatenate((arr_3dsdf_normals, arr_3dshape_normals), axis=-2)

        sdf_local = np.zeros((arr_3dshape_points.shape[0], 1))
        sdf = np.concatenate((arr_3dsdf_sdf, sdf_local), axis=-2)

        # gt = self.list_cases[idx][1][0]
        # covariates = self.list_cases[1][1]
        gt = {'id': self.ids[idx], 'ctl_path': self.ctl[idx], 'gt_path': self.shape3d[idx]}

        gt.update({'sdf': sdf})
        gt.update({'normal': arr_normals})
        # get covariates
        covariates = self.get_covariates_for_one_case(idx)

        return arr_samples, covariates, gt, idx



'''
def load_yaml_as_dict(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict
'''
def load_yaml_as_dict(yaml_path):
    import yaml
    with open(yaml_path, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict

def get_airway_ids(filename_split, split='train'):
    split = load_yaml_as_dict(filename_split)[split]
    return split


def get_youngest_ids(filename_split, split='train'):
    split = load_yaml_as_dict(filename_split)[split]
    return split



def get_airways_for_transport(filename_datasource, filename_split, split='test_multiple'):
    timelines = load_yaml_as_dict(filename_split)[split]
    df_data = pd.read_csv(filename_datasource, header=0)

    list_scans = []
    list_patient_scans = []
    for patient in timelines:
        #df_data_split = df_data.loc[df_data['PID'].astype('str') == patient['name']]
        #ages = np.array(df_data_split['age'].values)
        #youngest_scan = df_data_split.loc[df_data_split['age'] == ages.min()]
        #if  youngest_scan['id'].values[0]== 1181:
        list_scans += patient['value']
        df_data_split = df_data.loc[df_data['PID'].astype('str') == patient['name']]
        ages = np.array(df_data_split['age'].values)
        youngest_scan = df_data_split.loc[df_data_split['age'] == ages.min()]
        other_scans = df_data_split.loc[df_data_split['age'] > ages.min()]
        other_scans = other_scans['id'].values[np.argsort(other_scans['age'])]
        #print(youngest_scan['id'].values[0] )
        #if youngest_scan['id'].values[0] == 1366 or youngest_scan['id'].values[0] == 1369:
        current_dict = {'patient': patient['name'],
                        'youngest_scan': youngest_scan['id'].values[0],
                        'other_scans': other_scans}
        list_patient_scans.append(current_dict)
    return list_patient_scans



def read_data(split, df_data, attributes):
    #df_data = pd.read_csv(filename_datasource, header=0)
    df_data_split = df_data.loc[df_data['id'].astype('str').isin(split)]

    # read covariates
    list_attributes = []
    for ith_attribute in attributes:
        arr_current_attribute = np.array(df_data_split[ith_attribute])
        list_attributes.append(arr_current_attribute)
    features = np.array(list_attributes).T

    # read target samples of the shape
    shape3d_values = np.array(df_data_split['3dshape_npy'])
    pvshape3d_values = np.array(df_data_split['3dshape'])
    sdf3d_values = np.array(df_data_split['3dsdf'])
    id_values = np.array(df_data_split['id'].astype('str'))
    ctl_values = np.array(df_data_split['ctl'])
    return features, shape3d_values, sdf3d_values, id_values, ctl_values, pvshape3d_values


def get_mean_normals_of_covariates(filename_dataset, train_split, attributes_names):
    df_data = filename_dataset
    train_covariates, train_shape3d, train_sdf3d, train_ids, train_ctl, _ = read_data(train_split,
                                                                                      df_data,
                                                                                      attributes_names)

    attributes_mean = {}
    attributes_std = {}

    for ith_attri in range(len(attributes_names)):
        attributes_mean[attributes_names[ith_attri]] = train_covariates[:, ith_attri].mean()
        attributes_std[attributes_names[ith_attri]] = train_covariates[:, ith_attri].std()

    return attributes_mean, attributes_std




def get_airway_data_for_id(test_idx, filename_dataset, train_split, attributes_names, stage='test', normalization='normal',  **kwargs):
    #train_split = get_ids(filename_split, split='train')
    import pandas as pd
    df_data = filename_dataset#pd.read_csv(filename_dataset, header=0)

    train_covariates, train_shape3d, train_sdf3d, train_ids, train_ctl, _ = read_data(train_split, df_data, attributes_names)
    case_covariates, case_shape3d, case_sdf3d, case_ids, case_ctl, case_pvshape = read_data([str(test_idx)], df_data, attributes_names)

    print(case_ids + '-----')
    attributes = {}
    ori_attributes = {}


    if normalization == 'normal':
        for ith_attri in range(len(attributes_names)):
            ori_attributes[attributes_names[ith_attri]]  = case_covariates[0][ith_attri]
            attributes[attributes_names[ith_attri]] = (torch.tensor([case_covariates[0][ith_attri]]) - train_covariates[:, ith_attri].mean()) / train_covariates[:,ith_attri].std()
            attributes[attributes_names[ith_attri]] = attributes[attributes_names[ith_attri]].float()[None, :]
    elif normalization == 'min_max':
        for ith_attri in range(len(attributes_names)):
            ori_attributes[attributes_names[ith_attri]]  = case_covariates[0][ith_attri]
            attributes[attributes_names[ith_attri]] = (torch.tensor([case_covariates[0][ith_attri]]) - train_covariates[:, ith_attri].min()) / (train_covariates[:,ith_attri].max()-train_covariates[:, ith_attri].min())
            attributes[attributes_names[ith_attri]] = attributes[attributes_names[ith_attri]].float()[None, :]
            attributes[attributes_names[ith_attri]] = attributes[attributes_names[ith_attri]]* 2 - 1

    pv_3dshape_points = np.array(np.load(case_shape3d[0])) # [:, [0, 1]] / 10
    pv_3dshape_normals = np.array(np.load(case_shape3d[0]))
    sampled_idx = np.random.randint(0, np.array(pv_3dshape_points[:, [0, 1, 2]]).shape[0], 20000)
    arr_3dshape_normals = torch.from_numpy(np.array(pv_3dshape_normals[:, [3, 4, 5]][sampled_idx])) * (-1)
    arr_3dshape_points = torch.from_numpy(np.array(pv_3dshape_points[:, [0, 1, 2]][sampled_idx])).float() / 60

    #
    arr_3dsdf = np.load(case_sdf3d[0])[0:250000]  # [:, [0, 1, 2]] / 10
    arr_3dsdf_sdf = arr_3dsdf[..., 3]
    #arr_3dsdf_pos = arr_3dsdf[arr_3dsdf_sdf >= 2.]
    arr_3dsdf_pos = arr_3dsdf[np.abs(arr_3dsdf_sdf) >= 2.]
    sampled_idx_pos = np.random.randint(0, arr_3dsdf_pos.shape[0], 10000)

    arr_3dsdf = arr_3dsdf_pos[sampled_idx_pos]  # np.concatenate((arr_3dsdf_pos[sampled_idx_pos], arr_3dsdf_neg[sampled_idx_neg]), axis=0)
    arr_3dsdf_points = torch.from_numpy(arr_3dsdf[..., [0, 1, 2]]).float() / 60
    arr_3dsdf_sdf = torch.from_numpy(arr_3dsdf[..., [3]]) / 60
    arr_3dsdf_normals = torch.zeros_like((arr_3dsdf_points))
    arr_samples = torch.cat((arr_3dsdf_points, arr_3dshape_points), dim=-2)[None, :, :]
    arr_normals = torch.cat((arr_3dsdf_normals, arr_3dshape_normals), dim=-2)[None, :, :]

    sdf_local = torch.zeros((arr_3dshape_points.shape[0], 1))
    sdf = torch.cat((arr_3dsdf_sdf, sdf_local), dim=-2)

    gt = {'sdf': sdf[None, :],
          'id': [case_ids[0]],
          'normal': arr_normals.float(),
          'ctl_path': [case_ctl[0]],
          'gt_path': [case_shape3d[0]],
          'vis_path': case_pvshape[0],
          'covariates': ori_attributes}

    return arr_samples.float(), attributes, gt


class PediatricAirway3DShapeSubset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes=['weight', 'age', 'sex', ],
            split='train',
    ):
        self.attributes = attributes
        self.num_of_workers = 8

        self.split = self.load_yaml_as_dict(filename_split)[split][0:50]
        self.train_split = self.load_yaml_as_dict(filename_split)['train'][0:50]
        #self.split = ['1067']
        self.filename_datasource = filename_datasource
        self.covariates, self.shape3d, self.sdf3d, self.ids, self.ctl = self.read_data(self.split)
        self.train_covariates, self.train_shape3d, self.train_sdf3d, self.train_ids, self.train_ctl = self.read_data(
            self.train_split)
        self.list_cases = []
        self.init_img_pool()

    def __len__(self):
        return len(self.ids)

    def split_ids(self, ids, split_num):
        ids_split = np.array_split(np.arange(len(ids)), split_num)
        return ids_split

    def init_img_pool(self):
        manager = Manager()
        pts_dic = manager.dict()

        split_ids = self.split_ids(self.split, self.num_of_workers)
        procs = []
        for i in range(self.num_of_workers):
            p = Process(target=self.read_data_into_zipnp, args=(split_ids[i], pts_dic))
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()
        print("the loading phase finished, total {} shape have been loaded".format(len(pts_dic)))
        for idx in self.ids:
            self.list_cases.append([pts_dic[idx]['points_on_surface'], pts_dic[idx]['points_off_surface']])

    def get_covariates_for_one_case(self, idx):
        covariates = {}
        # for ith_attri in range(len(self.attributes)):
        #     covariates[self.attributes[ith_attri]] = (torch.tensor([self.covariates[idx][ith_attri]]) - self.train_covariates[:, ith_attri].min()) / (self.train_covariates[:, ith_attri].max() - self.train_covariates[:,ith_attri].min())
        #     covariates[self.attributes[ith_attri]] = covariates[self.attributes[ith_attri]] * 2 - 1
        for ith_attri in range(len(self.attributes)):
            covariates[self.attributes[ith_attri]] = (torch.tensor(
                [self.covariates[idx][ith_attri]]) - self.train_covariates[:,
                                                     ith_attri].mean()) / self.train_covariates[:, ith_attri].std()

        return covariates

    def read_data_into_zipnp(self, ids, img_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(ids)).start()
        count = 0
        for idx in ids:
            dict_data_case = {}

            # get points on surface
            pv_3dshape = np.load(self.shape3d[idx])
            # get points off surface
            arr_3dsdf = np.load(self.sdf3d[idx])

            dict_data_case['points_on_surface'] = blosc.pack_array(pv_3dshape)
            dict_data_case['points_off_surface'] = blosc.pack_array(arr_3dsdf)

            img_dic[self.ids[idx]] = dict_data_case
            count += 1
            pbar.update(count)
        pbar.finish()

    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        shape3d_values = np.array(df_data_split['3dshape_npy'])
        sdf3d_values = np.array(df_data_split['3dsdf'])
        id_values = np.array(df_data_split['id'].astype('str'))
        ctl_values = np.array(df_data_split['ctl'])
        return features, shape3d_values, sdf3d_values, id_values, ctl_values

    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):

        list_items = [blosc.unpack_array(item) for item in self.list_cases[idx]]
        arr_points_on_surface = torch.from_numpy(list_items[0]).float()
        arr_points_off_surface = torch.from_numpy(list_items[1]).float()

        sampled_idx = np.random.randint(0, np.array(arr_points_on_surface[:, [0, 1, 2]]).shape[0], 500)  # 2000) #500
        arr_3dshape_normals = np.array(arr_points_on_surface[:, [3, 4, 5]][sampled_idx]) * (-1)
        arr_3dshape_points = np.array(arr_points_on_surface[:, [0, 1, 2]][sampled_idx]) / 60

        arr_3dsdf_sdf = arr_points_off_surface[..., 3]
        arr_3dsdf_off = arr_points_off_surface[np.abs(arr_3dsdf_sdf) >= 2.]
        sampled_idx_off = np.random.randint(0, arr_3dsdf_off.shape[0], 250)  # 1000)  # 250)#

        arr_3dsdf = arr_3dsdf_off[sampled_idx_off]
        arr_3dsdf_points = arr_3dsdf[..., [0, 1, 2]] / 60
        arr_3dsdf_sdf = arr_3dsdf[..., [3]] / 60
        arr_3dsdf_normals = np.zeros_like((arr_3dsdf_points))
        arr_samples = np.concatenate((arr_3dsdf_points, arr_3dshape_points), axis=-2)
        arr_normals = np.concatenate((arr_3dsdf_normals, arr_3dshape_normals), axis=-2)

        sdf_local = np.zeros((arr_3dshape_points.shape[0], 1))
        sdf = np.concatenate((arr_3dsdf_sdf, sdf_local), axis=-2)

        # gt = self.list_cases[idx][1][0]
        # covariates = self.list_cases[1][1]
        gt = {'id': self.ids[idx], 'ctl_path': self.ctl[idx], 'gt_path': self.shape3d[idx]}

        gt.update({'sdf': sdf})
        gt.update({'normal': arr_normals})
        # get covariates
        covariates = self.get_covariates_for_one_case(idx)

        return arr_samples, covariates, gt, idx


