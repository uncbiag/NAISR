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



class Starman2DShapeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split=None,
            attributes=['cov_1', 'cov_2', 'cov_3', 'cov_4',],
            split='train',
    ):
        self.attributes = attributes
        self.num_of_workers = 8
        self.filename_datasource = filename_datasource
        self.covariates, self.shape3d, self.sdf3d, self.ids  = self.read_data(split)
        self.train_covariates, self.train_shape3d, self.train_sdf3d, self.train_ids = self.read_data('train')

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

        split_ids = self.split_ids(self.ids, self.num_of_workers)
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
        for ith_attri in range(len(self.attributes)):
            covariates[self.attributes[ith_attri]] = (torch.tensor(
                [self.covariates[idx][ith_attri]]) - self.train_covariates[:, ith_attri].min()) / (
                                                                 self.train_covariates[:,
                                                                 ith_attri].max() - self.train_covariates[:,
                                                                                    ith_attri].min())
            covariates[self.attributes[ith_attri]] = covariates[self.attributes[ith_attri]] * 2 - 1
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
        self.df_data = pd.read_csv(self.filename_datasource[split], header=0)
        df_data_split = self.df_data.copy()
        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        shape3d_values = np.array(df_data_split['2dshape'])
        sdf3d_values = np.array(df_data_split['2dsdf'])
        id_values = np.array(df_data_split['id'].astype('str'))
        return features, shape3d_values, sdf3d_values, id_values

    def load_yaml_as_dict(self, yaml_path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def __getitem__(self, idx):

        list_items = [blosc.unpack_array(item) for item in self.list_cases[idx]]
        arr_points_on_surface = torch.from_numpy(list_items[0]).float()
        arr_points_off_surface = torch.from_numpy(list_items[1]).float()

        sampled_idx = np.random.randint(0, np.array(arr_points_on_surface).shape[0], 500)  # 2000) #500
        arr_3dshape_normals = np.array(arr_points_on_surface[:, [2, 3]][sampled_idx]) #* (-1)
        arr_3dshape_points = np.array(arr_points_on_surface[:, [0, 1]][sampled_idx]) #/ 60

        arr_3dsdf_sdf = arr_points_off_surface[..., 2]
        arr_3dsdf_off = arr_points_off_surface#[np.abs(arr_3dsdf_sdf) >= 0.1]
        sampled_idx_off = np.random.randint(0, arr_3dsdf_off.shape[0], 250)  # 1000)  # 250)#

        arr_3dsdf = arr_3dsdf_off[sampled_idx_off]
        arr_3dsdf_points = arr_3dsdf[..., [0, 1]] #/ 60
        arr_3dsdf_sdf = arr_3dsdf[..., [2]] #/ 60
        arr_3dsdf_normals = np.zeros_like((arr_3dsdf_points))
        arr_samples = np.concatenate((arr_3dsdf_points, arr_3dshape_points), axis=-2)
        arr_normals = np.concatenate((arr_3dsdf_normals, arr_3dshape_normals), axis=-2)

        sdf_local = np.zeros((arr_3dshape_points.shape[0], 1))
        sdf = np.concatenate((arr_3dsdf_sdf, sdf_local), axis=-2)

        # gt = self.list_cases[idx][1][0]
        # covariates = self.list_cases[1][1]
        gt = {'id': self.ids[idx], 'gt_path': self.shape3d[idx]}

        gt.update({'sdf': sdf})
        gt.update({'normal': arr_normals})
        # get covariates
        covariates = self.get_covariates_for_one_case(idx)

        return arr_samples, covariates, gt, idx



def load_yaml_as_dict(yaml_path):
    import yaml
    with open(yaml_path, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def get_starman_ids(dict_filename_dataset, split='train'):
    filename = dict_filename_dataset[split]
    pd_dataset = pd.read_csv(filename, header=0)
    #split = load_yaml_as_dict(filename_split)[split]
    ids = pd_dataset['id'].values
    return ids


# def get_youngest_ids(filename_split, split='train'):
#     split = load_yaml_as_dict(filename_split)[split]
#     return split
#
#
# def get_patients_for_transport(filename_datasource, filename_split, split='test_multiple'):
#     timelines = load_yaml_as_dict(filename_split)[split]
#     df_data = pd.read_csv(filename_datasource, header=0)
#
#     list_scans = []
#     list_patient_scans = []
#     for patient in timelines:
#         # df_data_split = df_data.loc[df_data['PID'].astype('str') == patient['name']]
#         # ages = np.array(df_data_split['age'].values)
#         # youngest_scan = df_data_split.loc[df_data_split['age'] == ages.min()]
#         # if  youngest_scan['id'].values[0]== 1181:
#         list_scans += patient['value']
#         df_data_split = df_data.loc[df_data['PID'].astype('str') == patient['name']]
#         ages = np.array(df_data_split['age'].values)
#         youngest_scan = df_data_split.loc[df_data_split['age'] == ages.min()]
#         other_scans = df_data_split.loc[df_data_split['age'] > ages.min()]
#         other_scans = other_scans['id'].values[np.argsort(other_scans['age'])]
#         # print(youngest_scan['id'].values[0] )
#         # if youngest_scan['id'].values[0] == 1366 or youngest_scan['id'].values[0] == 1369:
#         current_dict = {'patient': patient['name'],
#                         'youngest_scan': youngest_scan['id'].values[0],
#                         'other_scans': other_scans}
#         list_patient_scans.append(current_dict)
#     return list_patient_scans


def read_starman_data(split, df_data, attributes):
    # df_data = pd.read_csv(filename_datasource, header=0)
    df_data_split = df_data.loc[df_data['id'].astype('str').isin(split)]

    # read covariates
    list_attributes = []
    for ith_attribute in attributes:
        arr_current_attribute = np.array(df_data_split[ith_attribute])
        list_attributes.append(arr_current_attribute)
    features = np.array(list_attributes).T

    # read target samples of the shape
    shape3d_values = np.array(df_data_split['2dshape'])
    sdf3d_values = np.array(df_data_split['2dsdf'])
    id_values = np.array(df_data_split['id'].astype('str'))

    template_control_values = np.array(df_data_split['temp_control'].astype('str'))
    template_contour_values = np.array(df_data_split['temp_contour'].astype('str'))
    template_values = {'mean_control': template_control_values, 'mean_contour': template_contour_values}

    return features, shape3d_values, sdf3d_values, id_values, template_values


def get_mean_normals_of_covariates(filename_dataset, train_split, attributes_names):

    attributes_mean = {}
    attributes_std = {}

    for ith_attri in range(len(attributes_names)):
        attributes_mean[attributes_names[ith_attri]] = 0
        attributes_std[attributes_names[ith_attri]] = 1

    return attributes_mean, attributes_std


def get_starman_data_for_id(test_idx, filename_dataset, train_split, attributes_names, stage, **kwargs):
    # train_split = get_ids(filename_split, split='train')

    df_test_data = filename_dataset #pd.read_csv(filename_dataset[stage])

    case_covariates, case_shape2d, case_sdf2d, case_ids, case_template = read_starman_data([str(test_idx)], df_test_data, attributes_names)

    print(case_ids + '-----')
    attributes = {}
    ori_attributes = {}
    for ith_attri in range(len(attributes_names)):
        ori_attributes[attributes_names[ith_attri]] = case_covariates[0][ith_attri]
        attributes[attributes_names[ith_attri]] = torch.tensor([case_covariates[0][ith_attri]])
        attributes[attributes_names[ith_attri]] = attributes[attributes_names[ith_attri]].float()[None, :]

    arr_2dshape_pts_with_normals = np.array(np.load(case_shape2d[0]))
    sampled_idx = np.random.randint(0, arr_2dshape_pts_with_normals.shape[0], 2000)

    arr_2dshape_normals = torch.from_numpy(np.array(arr_2dshape_pts_with_normals[:, [2, 3]][sampled_idx])).float()
    arr_2dshape_points = torch.from_numpy(np.array(arr_2dshape_pts_with_normals[:, [0, 1]][sampled_idx])).float()

    #
    arr_2dsdf = np.load(case_sdf2d[0])
    arr_2dsdf_pos = arr_2dsdf#[arr_2dsdf_sdf >= 2.]
    sampled_idx_pos = np.random.randint(0, arr_2dsdf_pos.shape[0], 1000)

    arr_2dsdf = arr_2dsdf_pos[sampled_idx_pos]  # np.concatenate((arr_3dsdf_pos[sampled_idx_pos], arr_3dsdf_neg[sampled_idx_neg]), axis=0)
    arr_2dsdf_points = torch.from_numpy(arr_2dsdf[..., [0, 1]]).float() #/ 60
    arr_2dsdf_sdf = torch.from_numpy(arr_2dsdf[..., [2]]) #/ 60
    arr_2dsdf_normals = torch.zeros_like((arr_2dsdf_points))

    arr_samples = torch.cat((arr_2dsdf_points, arr_2dshape_points), dim=-2)[None, :, :]
    arr_normals = torch.cat((arr_2dsdf_normals, arr_2dshape_normals), dim=-2)[None, :, :]

    sdf_local = torch.zeros((arr_2dshape_points.shape[0], 1))
    sdf = torch.cat((arr_2dsdf_sdf, sdf_local), dim=-2)

    # READING TEMPLATES

    pv_contour = pv.read(case_template['mean_contour'][0])
    arr_control = np.load(case_template['mean_control'][0])

    gt = {'sdf': sdf[None, :],
          'id': [case_ids[0]],
          'normal': arr_normals.float(),
          'gt_path': [case_shape2d[0]],
          'covariates': ori_attributes,
          'mean_control': arr_control,
          'mean_contour': pv_contour}

    return arr_samples.float(), attributes, gt


def get_starmans_for_transport(filename_datasource, split='test'):
    df_data = pd.read_csv(filename_datasource[split], header=0)

    list_subjects = np.unique(df_data['PID'].values)
    list_patient_scans = []
    for ith_subj in list_subjects:
        #ith_subj = '{0:04}'.format(ith_subj)
        subjname = str('{0:04}'.format(ith_subj))
        first_scan = subjname + '_0'
        scans = df_data[df_data['PID'] == ith_subj]
        other_scans = list(scans[scans['id'] != first_scan]['id'].values)

        current_dict = {'patient': subjname,
                        'youngest_scan': first_scan,
                        'other_scans': other_scans}
        list_patient_scans.append(current_dict)

    return list_patient_scans