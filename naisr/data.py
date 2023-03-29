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

def get_toy_instance_filenames(data_source, shapetype,  split):
    filename_split = os.path.join(data_source, shapetype, split + '_split_' + str(shapetype) + '.npz')
    #filename_split = os.path.join(data_source, shapetype, 'split_' + str(shapetype) + '.npz')
    split_data = np.load(filename_split, allow_pickle=True)[split]

    npzfiles = []
    for current_attri in split_data:

        current_npz_name = os.path.join(data_source, shapetype, 'npz', shapetype)
        for ith_attri in current_attri.keys():
            current_npz_name = current_npz_name + str('_') + ith_attri + '_' + str(current_attri[ith_attri])
        current_npz_name = current_npz_name + '.npz'
        npzfiles.append(current_npz_name)
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]



def unpack_toydata(filename, subsample=2000):
    npz = np.load(filename, allow_pickle=True)
    if subsample is None:
        return npz
    pts_tensor = remove_nans(torch.from_numpy(npz["pts"]))
    norms_tensor = torch.from_numpy(npz["norms"])

    # split the sample into half
    subsample= int(subsample)
    random_pos = (torch.rand(subsample) * pts_tensor.shape[0]).long()
    samples = torch.index_select(pts_tensor, 0, random_pos)[:, 0:3]
    normals = torch.index_select(norms_tensor, 0, random_pos)
    normals = normals / (torch.sum(normals**2, dim=1)**0.5)[:, None]
    covariates = npz["covariates"]

    samples = (samples - samples.min()) / (samples.max() - samples.min())
    samples -= 0.5
    samples *= 2.

    return samples, covariates, normals

class ToySamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        shapetype,
        print_filename=False,
        articulation=False,
        num_atc_parts=1,
    ):
        self.subsample = subsample

        self.data_source = data_source

        self.npyfiles = get_toy_instance_filenames(data_source, shapetype, split)
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        coords, covariates, normals = unpack_toydata(filename, subsample=self.subsample)
        gt = {'normals': normals.float(), 'sdf': torch.zeros((len(coords), 1)).float()}
        return coords.float(), covariates.item(), gt #unpack_sdf_samples(filename, self.subsample, self.articualtion, self.num_atc_parts), idx

class ToyEllipsoidDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            attributes,
            shapetype,
            split='train',
            articulation=False,
            num_atc_parts=1,
    ):
        self.shapetype = shapetype
        self.data_source = data_source
        self.split = split

        self.npyfiles = get_toy_instance_filenames(data_source, shapetype, split='train')
        self.attributes = attributes
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        #if self.shapetype == 'ellipsoid:':
        if self.split == 'train':
            a= torch.rand(1) * 0.7 + 0.1
            b = torch.rand(1) * 0.7 + 0.1
            c = torch.rand(1) * 0.7 + 0.1
        elif self.split == 'test':
            a = torch.rand(1) * 0.2 + 0.8
            b = torch.rand(1) * 0.2 + 0.8
            c = 0.5


        x = torch.rand((2000, 1)) * 2 - 1#* a + a
        y = torch.rand((2000, 1)) * 2 - 1#* b + b
        z = torch.rand((2000, 1)) * 2 - 1#* c + c

        coords = torch.cat([x, y, z], dim=-1)
        s = x**2 / a**2 + y**2 / b**2 + z**2 / c**2  - 1

        normals = torch.cat((2 * x / a**2, 2 * y / b**2, 2 * z / c**2 ), dim=-1)

        covariates = {}
        #for ith_attri in self.attributes:
        covariates[self.attributes[0]] = a
        covariates[self.attributes[1]] = b

        a_temp = 0.4
        b_temp = 0.4
        coords_templ = torch.cat([x, y, z], dim=-1)
        s_templ = torch.tanh(x**2 / a_temp**2 + y**2 /b_temp**2 + z**2/c**2 - 1)
        normals_templ = torch.cat((2 * x / a_temp ** 2, 2 * y / b_temp ** 2), dim=-1)



        gt = {'normals': normals.float(),
              'sdf': s.float(),
              'templ_normals': normals_templ.float(),
              'templ_sdf': s_templ.float(),
              }
        return coords.float(), covariates, gt




class ToyTorusDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            attributes,
            shapetype,
            split='train',
            articulation=False,
            num_atc_parts=1,
    ):
        self.shapetype = shapetype
        self.data_source = data_source
        self.split = split

        self.npyfiles = get_toy_instance_filenames(data_source, shapetype, split='train')
        self.attributes = attributes
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts

        if self.split == 'print_train':
            self.attribute_pair = []
            rs = np.arange(start=0.15, stop=0.36, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r+1e-5) // 0.05 + 1, stop=(1 -  r + 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})
        elif self.split == 'print_test':
            self.attribute_pair = []
            rs = np.arange(start=0.4, stop=0.46, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r+1e-5) // 0.05 + 1, stop=(1 -  r + 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})

            rs = np.arange(start=0.05, stop=0.11, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r+1e-5) // 0.05 + 1, stop=(1 - r+ 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})



    def __len__(self):
        if 'print' in self.split:
           return len(self.attribute_pair)
        else:
            return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        '''
        if self.split == 'train':
            R = torch.rand(1) * 0.8
            r = torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))
        elif self.split == 'test':

            R = torch.rand(1) * 0.2 + 0.8
            r = torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))
        else:
            R = torch.rand(1) * 0.7 + 0.1
            r = torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))

        #theta = torch.deg2rad(torch.rand((2000, 1)) * torch.tensor(360))
        #fi = torch.deg2rad(torch.rand((2000, 1)) * torch.tensor(360))

        x = torch.rand((2000, 1)) * 2 - 1#* a + a
        y = torch.rand((2000, 1)) * 2 - 1#* b + b
        z = torch.rand((2000, 1)) * 2 - 1#* c + c
        '''

        if self.split == 'train':
            r = torch.rand(1) * 0.2 + 0.15
            R = torch.rand(1) * (1 - 2 * r) + r #torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))
        elif self.split == 'test':
            r1 = torch.rand(1) * 0.1 + 0.35
            r2 = torch.rand(1) * 0.1 + 0.05
            if np.random.rand(1) > 0.5:
                r=r1
            else:
                r=r2
            R = torch.rand(1) * (1 - 2 * r) + r
        elif self.split == 'print_train':
            r = self.attribute_pair[idx]['r']
            R = self.attribute_pair[idx]['R']
        elif self.split == 'print_test':
            r = self.attribute_pair[idx]['r']
            R = self.attribute_pair[idx]['R']

        x = (torch.rand((2000, 1)) * 2 - 1) * 1.2
        y = (torch.rand((2000, 1)) * 2 - 1) * 1.2
        z = (torch.rand((2000, 1)) * 2 - 1) * 1.2

        coords = torch.cat([x, y, z], dim=-1)
        s = torch.tanh(torch.square(torch.sqrt(x**2 + y**2) - R)  + z**2  - r**2)

        diff_x = 2 * (torch.sqrt(x ** 2 + y ** 2) - R) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * x
        diff_y = 2 * (torch.sqrt(x ** 2 + y ** 2) - R) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * y
        diff_z = 2 * z
        normals = torch.cat((diff_x, diff_y, diff_z), dim=-1)

        covariates = {}
        #for ith_attri in self.attributes:
        covariates[self.attributes[0]] = R + r
        covariates[self.attributes[1]] =  R-r




        #gt = {'normals': normals.float(), 'sdf': s.float()}
        #return coords.float(), covariates, gt

        R_templ = 0.6 #(1 + 0.1)/2
        r_templ = 0.2#(1 - 0.1)/2
        #R_templ = 0.6 #(1 + 0.1)/2
        #r_templ =  0.4 #(1 - 0.1)/2
        #coords_templ = torch.cat([x, y], dim=-1)
        s_templ = torch.tanh(torch.square(torch.sqrt(x**2 + y**2) - R_templ) + z**2  - r_templ**2)
        diff_x_templ = 2 * (torch.sqrt(x ** 2 + y ** 2) - R_templ) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * x
        diff_y_templ = 2 * (torch.sqrt(x ** 2 + y ** 2) - R_templ) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * y
        normals_templ = torch.cat((diff_x_templ, diff_y_templ), dim=-1)




        '''
        
        theta = torch.rand((2000, 1)) * torch.pi * 2
        fi = torch.rand((2000, 1)) * torch.pi * 2
        #sign_y = (torch.rand((2000, 1)) > 0.5).float() * 2 - 1

        x = (R + r * torch.cos(theta)) * torch.cos(fi)
        y = (R + r * torch.cos(theta)) * torch.sin(fi)
        z = r * torch.sin(theta)    
        '''

        gt = {'normals': normals.float(),
              'sdf': s.float(),
              'templ_normals': normals_templ.float(),
              'templ_sdf': s_templ.float(),
              }
        return coords.float(), covariates, gt


class ToyTorusPCDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            attributes,
            shapetype,
            split='train',
            articulation=False,
            num_atc_parts=1,
    ):
        self.shapetype = shapetype
        self.data_source = data_source
        self.split = split

        self.npyfiles = get_toy_instance_filenames(data_source, shapetype, split='train')
        self.attributes = attributes
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts

        if self.split == 'print_train':
            self.attribute_pair = []
            rs = np.arange(start=0.15, stop=0.36, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r + 1e-5) // 0.05 + 1,
                                stop=(1 - r + 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})
        elif self.split == 'print_test':
            self.attribute_pair = []
            rs = np.arange(start=0.4, stop=0.46, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r + 1e-5) // 0.05 + 1,
                                stop=(1 - r + 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})

            rs = np.arange(start=0.05, stop=0.11, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r + 1e-5) // 0.05 + 1,
                                stop=(1 - r + 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})

    def __len__(self):
        if 'print' in self.split:
            return len(self.attribute_pair)
        else:
            return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        '''
        if self.split == 'train':
            R = torch.rand(1) * 0.8
            r = torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))
        elif self.split == 'test':

            R = torch.rand(1) * 0.2 + 0.8
            r = torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))
        else:
            R = torch.rand(1) * 0.7 + 0.1
            r = torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))

        #theta = torch.deg2rad(torch.rand((2000, 1)) * torch.tensor(360))
        #fi = torch.deg2rad(torch.rand((2000, 1)) * torch.tensor(360))

        x = torch.rand((2000, 1)) * 2 - 1#* a + a
        y = torch.rand((2000, 1)) * 2 - 1#* b + b
        z = torch.rand((2000, 1)) * 2 - 1#* c + c
        '''

        if self.split == 'train':
            r = torch.rand(1) * 0.2 + 0.15
            R = torch.rand(1) * (1 - 2 * r) + r  # torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))
        elif self.split == 'test':
            r1 = torch.rand(1) * 0.1 + 0.35
            r2 = torch.rand(1) * 0.1 + 0.05
            if np.random.rand(1) > 0.5:
                r = r1
            else:
                r = r2
            R = torch.rand(1) * (1 - 2 * r) + r
        elif self.split == 'print_train':
            r = self.attribute_pair[idx]['r']
            R = self.attribute_pair[idx]['R']
        elif self.split == 'print_test':
            r = self.attribute_pair[idx]['r']
            R = self.attribute_pair[idx]['R']

        '''
        point cloud
        '''
        theta = torch.rand((2000, 1)) * torch.pi * 2
        fi = torch.rand((2000, 1)) * torch.pi * 2
        # sign_y = (torch.rand((2000, 1)) > 0.5).float() * 2 - 1

        x = (R + r * torch.cos(theta)) * torch.cos(fi)
        y = (R + r * torch.cos(theta)) * torch.sin(fi)
        z = r * torch.sin(theta)

        coords = torch.cat([x, y, z], dim=-1)
        diff_x = 2 * (torch.sqrt(x ** 2 + y ** 2) - R) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * x
        diff_y = 2 * (torch.sqrt(x ** 2 + y ** 2) - R) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * y
        diff_z = 2 * z
        normals = torch.cat((diff_x, diff_y, diff_z), dim=-1)

        covariates = {}
        # for ith_attri in self.attributes:
        covariates[self.attributes[0]] = R + r
        covariates[self.attributes[1]] = R - r

        # gt = {'normals': normals.float(), 'sdf': s.float()}
        # return coords.float(), covariates, gt

        R_templ = 0.6  # (1 + 0.1)/2
        r_templ = 0.2  # (1 - 0.1)/2

        diff_x_templ = 2 * (torch.sqrt(x ** 2 + y ** 2) - R_templ) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * x
        diff_y_templ = 2 * (torch.sqrt(x ** 2 + y ** 2) - R_templ) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * y
        normals_templ = torch.cat((diff_x_templ, diff_y_templ), dim=-1)

        x_templ = (R_templ + r_templ * torch.cos(theta)) * torch.cos(fi)
        y_templ = (R_templ + r_templ * torch.cos(theta)) * torch.sin(fi)
        z_templ = r_templ * torch.sin(theta)
        coords_templ = torch.cat([x_templ, y_templ, z_templ], dim=-1)

        gt = {'normals': normals.float(),
              'templ_normals': normals_templ.float(),}

        return coords.float(), coords_templ.float(), covariates, gt




class Toy2DEllipsoidDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            attributes,
            shapetype,
            split='train',
            articulation=False,
            num_atc_parts=1,
    ):
        self.shapetype = shapetype
        self.data_source = data_source
        self.split = split

        self.npyfiles = get_toy_instance_filenames(data_source, shapetype, split='train')
        self.attributes = attributes
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        #if self.shapetype == 'ellipsoid:':
        if self.split == 'train':
            a = torch.rand(1) * 0.7 + 0.1
            b = torch.rand(1) * 0.7 + 0.1
        elif self.split == 'test':
            a = torch.rand(1) * 0.2 + 0.8
            b = torch.rand(1) * 0.2 + 0.8

        x = torch.rand((2000, 1)) * 2 - 1#* a + a
        y = torch.rand((2000, 1)) * 2 - 1#* b + b

        coords = torch.cat([x, y], dim=-1)
        s = torch.tanh(torch.sqrt(x**2 / a**2 + y**2 /b**2) - 1)

        normals = torch.cat((2 * x / a**2, 2 * y / b**2), dim=-1)

        covariates = {}
        #for ith_attri in self.attributes:
        covariates[self.attributes[0]] = a
        covariates[self.attributes[1]] = b


        a_temp = 0.4
        b_temp = 0.4
        coords_templ = torch.cat([x, y], dim=-1)
        s_templ = torch.tanh(torch.sqrt(x**2 / a_temp**2 + y**2 /b_temp**2) - 1)
        normals_templ = torch.cat((2 * x / a_temp ** 2, 2 * y / b_temp ** 2), dim=-1)



        gt = {'normals': normals.float(),
              'sdf': s.float(),
              'templ_normals': normals_templ.float(),
              'templ_sdf': s_templ.float(),
              }



        return coords.float(), covariates, gt



class Toy2DPointEllipsoidDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            attributes,
            subsample,
            shapetype,
            split='train',
            articulation=False,
            num_atc_parts=1,
    ):
        self.shapetype = shapetype
        self.data_source = data_source
        self.split = split

        self.npyfiles = get_toy_instance_filenames(data_source, shapetype, split='train')
        self.attributes = attributes
        self.subsample = subsample
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        #if self.shapetype == 'ellipsoid:':
        if self.split == 'train':
            a = torch.rand(1) * 0.7 + 0.1
            b = torch.rand(1) * 0.7 + 0.1
        elif self.split == 'test':
            a = torch.rand(1) * 0.2 + 0.8
            b = torch.rand(1) * 0.2 + 0.8

        x = torch.rand((2000, 1)) * 2 - 1#* a + a
        sign_y = (torch.rand((2000, 1)) > 0.5).float() * 2 - 1
        y = torch.sqrt((1 - x**2 / a**2) * b**2) * sign_y

        coords = torch.cat([x, y], dim=-1)
        s = torch.tanh(x**2 / a**2 + y**2 /b**2 - 1)

        normals = torch.cat((2 * x / a**2, 2 * y / b**2), dim=-1)

        covariates = {}
        #for ith_attri in self.attributes:
        covariates[self.attributes[0]] = a
        covariates[self.attributes[1]] = b

        a_temp = 0.4
        b_temp = 0.4
        sign_y = (torch.rand((2000, 1)) > 0.5).float() * 2 - 1
        y_temp = torch.sqrt((1 - x**2 / a_temp**2) * b_temp**2) * sign_y
        coords_templ = torch.cat([x, y_temp], dim=-1)

        s_templ = torch.tanh(x**2 / a_temp**2 + y_temp**2 /b_temp**2 - 1)
        normals_templ = torch.cat((2 * x / a_temp ** 2, 2 * y / b_temp ** 2), dim=-1)



        gt = {'normals': normals.float(),
              'sdf': s.float(),
              'templ_normals': normals_templ.float(),
              'templ_sdf': s_templ.float(),
              }



        return coords.float(), coords_templ.float(), covariates, gt

class Toy2DTorusDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            attributes,
            shapetype,
            split='train',
            articulation=False,
            num_atc_parts=1,
    ):
        self.shapetype = shapetype
        self.data_source = data_source
        self.split = split

        self.npyfiles = get_toy_instance_filenames(data_source, shapetype, split='train')
        self.attributes = attributes
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts

        if self.split == 'print_train':
            self.attribute_pair = []
            rs = np.arange(start=0.15, stop=0.4, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r+1e-5) // 0.05 + 1, stop=(1 -  r + 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})
        elif self.split == 'print_test':
            self.attribute_pair = []
            rs = np.arange(start=0.35, stop=0.55, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r+1e-5) // 0.05 + 1, stop=(1 -  r + 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})

            rs = np.arange(start=0.05, stop=0.2, step=0.05)
            for r in rs:
                Rs = (np.arange(start=(r+1e-5) // 0.05 + 1, stop=(1 - r+ 1e-5) // 0.05) * 0.05).tolist()  # torch.tensor(0.65)
                for R in Rs:
                    self.attribute_pair.append({'R': R, 'r': r})


    def __len__(self):
        if 'print' in self.split:
           return len(self.attribute_pair)
        else:
            return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        #if self.shapetype == 'ellipsoid:':
        #if self.split == 'train':
        #    R = torch.rand(1) * 0.8
        #    r = torch.rand(1) * torch.minimum(R, 1-R)  + 1e-3
        #elif self.split == 'test':
        #    R = torch.rand(1) * 0.2 + 0.8
        #    r = torch.rand(1) * torch.minimum(R, 1-R)  + 1e-3
        #else:
        #    R = torch.rand(1) * 0.8
        #    r = torch.rand(1) * torch.minimum(R, 1-R) + 1e-3


        '''
        if self.split == 'train':
            r = torch.rand(1) * 0.25 + 0.05
            R = torch.rand(1) * (1 - 2 * r) + r #torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))
        elif self.split == 'test':
            r = torch.rand(1) * 0.1 + 0.3
            R = torch.rand(1) * (1 - 2 * r) + r
        '''

        if self.split == 'train':
            r = torch.rand(1) * 0.2 + 0.15
            R = torch.rand(1) * (1 - 2 * r) + r #torch.maximum(torch.rand(1) * torch.minimum(R, 1 - R), torch.tensor(0.05))
        elif self.split == 'test':
            r1 = torch.rand(1) * 0.1 + 0.35
            r2 = torch.rand(1) * 0.1 + 0.05
            if np.random.rand(1) > 0.5:
                r=r1
            else:
                r=r2
            R = torch.rand(1) * (1 - 2 * r) + r
        elif self.split == 'print_train':
            r = self.attribute_pair[idx]['r']
            R = self.attribute_pair[idx]['R']
        elif self.split == 'print_test':
            r = self.attribute_pair[idx]['r']
            R = self.attribute_pair[idx]['R']

        x = (torch.rand((2000, 1)) * 2 - 1) * 1.2
        y = (torch.rand((2000, 1)) * 2 - 1) * 1.2

        coords = torch.cat([x, y,], dim=-1)
        s = torch.tanh(torch.abs(torch.sqrt(x**2 + y**2) - R) - r)

        diff_x = 2 * (torch.sqrt(x ** 2 + y ** 2) - R) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * x
        diff_y = 2 * (torch.sqrt(x ** 2 + y ** 2) - R) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * y

        normals = torch.cat((diff_x, diff_y), dim=-1)

        covariates = {}
        #for ith_attri in self.attributes:
        covariates[self.attributes[0]] = R+r
        covariates[self.attributes[1]] = R-r


        R_templ = 0.6 #(1.0 + 0.9) / 2 #0.6 #(1 + 0.1)/2
        r_templ =  0.2#(1.0 - 0.9) / 2 #0.2 #(1 - 0.1)/2
        #coords_templ = torch.cat([x, y], dim=-1)
        s_templ = torch.tanh(torch.abs(torch.sqrt(x**2 + y**2) - R_templ)  - r_templ)
        diff_x_templ = 2 * (torch.sqrt(x ** 2 + y ** 2) - R_templ) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * x
        diff_y_templ = 2 * (torch.sqrt(x ** 2 + y ** 2) - R_templ) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * y
        normals_templ = torch.cat((diff_x_templ, diff_y_templ), dim=-1)

        gt = {'normals': normals.float(),
              'sdf': s.float(),
              'templ_normals': normals_templ.float(),
              'templ_sdf': s_templ.float(),
              }
        return coords.float(), covariates, gt


class Toy2DPointTorusDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            attributes,
            subsample,
            shapetype,
            split='train',
            articulation=False,
            num_atc_parts=1,
    ):
        self.shapetype = shapetype
        self.data_source = data_source
        self.split = split

        self.npyfiles = get_toy_instance_filenames(data_source, shapetype, split='train')
        self.attributes = attributes
        self.subsample = subsample
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        #if self.shapetype == 'ellipsoid:':
        if self.split == 'train':
            R = torch.rand(1) * 0.8
            r = torch.rand(1) * torch.minimum(R, 1-R)  + 1e-3
        elif self.split == 'test':
            R = torch.rand(1) * 0.2 + 0.8
            r = torch.rand(1) * torch.minimum(R, 1-R)  + 1e-3
        else:
            R = torch.rand(1) * 0.8
            r = torch.rand(1) * torch.minimum(R, 1-R) + 1e-3

        theta = torch.rand((2000, 1)) * torch.pi * 2
        sign_y = (torch.rand((2000, 1)) > 0.5).float() * 2 - 1

        x = (R + r * sign_y) * torch.cos(theta)
        y = (R + r * sign_y) * torch.sin(theta)

        coords = torch.cat([x, y,], dim=-1)
        s = torch.tanh(torch.abs(torch.sqrt(x**2 + y**2) - R) - r)

        diff_x = 2 * (torch.sqrt(x ** 2 + y ** 2) - R) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * x
        diff_y = 2 * (torch.sqrt(x ** 2 + y ** 2) - R) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * y

        normals = torch.cat((diff_x, diff_y), dim=-1)

        covariates = {}
        #for ith_attri in self.attributes:
        covariates[self.attributes[0]] = R+r
        covariates[self.attributes[1]] = R-r


        R_templ = 0.55
        r_templ = 0.45
        x_templ = (R_templ + r_templ * sign_y) * torch.cos(theta)
        y_templ = (R_templ + r_templ * sign_y) * torch.sin(theta)
        coords_templ = torch.cat([x_templ, y_templ], dim=-1)

        s_templ = torch.tanh(torch.square(torch.sqrt(x_templ**2 + y_templ**2) - R_templ)  - r_templ**2)
        diff_x_templ = 2 * (torch.sqrt(x ** 2 + y ** 2) - R_templ) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * x
        diff_y_templ = 2 * (torch.sqrt(x ** 2 + y ** 2) - R_templ) * 0.5 * torch.pow(x ** 2 + y ** 2, -0.5) * 2 * y
        normals_templ = torch.cat((diff_x_templ, diff_y_templ), dim=-1)

        gt = {'normals': normals.float(),
              'sdf': s.float(),
              'templ_normals': normals_templ.float(),
              'templ_sdf': s_templ.float(),
              }
        return coords.float(), coords_templ.float(), covariates, gt

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
        for ith_attri in range(len(self.attributes)):
            covariates[self.attributes[ith_attri]] = (torch.tensor([self.covariates[idx][ith_attri]]) - self.train_covariates[:, ith_attri].min()) / (self.train_covariates[:, ith_attri].max() - self.train_covariates[:,ith_attri].min())
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
        '''
        #gt = self.list_cases[idx][1][0]
        #covariates = self.list_cases[1][1]
        gt = {'id': self.ids[idx], 'ctl_path': self.ctl[idx], 'gt_path': self.shape3d[idx]}
        sdf_local = np.zeros((arr_3dshape_points.shape[0], 1))
        gt.update({'sdf': sdf_local})
        gt.update({'normal': arr_3dshape_normals})
        # get covariates
        covariates = self.get_covariates_for_one_case(idx)

        return arr_3dshape_points, covariates, gt, idx
        '''


class PediatricAirway3DShapeDataset_puresdf(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes=['weight', 'age', 'sex', ],
            split='train',
    ):
        self.attributes = attributes

        self.split = self.load_yaml_as_dict(filename_split)[split]
        self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.shape3d, self.sdf3d, self.ids, self.ctl = self.read_data(self.split)
        self.train_covariates, self.train_shape3d, self.train_sdf3d, self.train_ids, self.train_ctl = self.read_data(
            self.train_split)

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

        #
        arr_3dsdf = np.load(self.sdf3d[idx])  # [:, [0, 1, 2]] / 10
        arr_3dsdf_sdf = arr_3dsdf[..., 3]
        arr_3dsdf_pos = arr_3dsdf[arr_3dsdf_sdf >= 0.]
        arr_3dsdf_neg = arr_3dsdf[arr_3dsdf_sdf < 0.]
        sampled_idx_pos = np.random.randint(0, arr_3dsdf_pos.shape[0], 2500)
        sampled_idx_neg = np.random.randint(0, arr_3dsdf_neg.shape[0], 2500)

        arr_3dsdf = np.concatenate((arr_3dsdf_pos[sampled_idx_pos], arr_3dsdf_neg[sampled_idx_neg]), axis=0)
        arr_3dsdf_points = torch.from_numpy(arr_3dsdf[..., [0, 1, 2]]).float() / 60
        # arr_3dsdf_points /= scale
        arr_3dsdf_sdf = torch.from_numpy(arr_3dsdf[..., [3]]) / 60
        arr_3dsdf_normals = torch.zeros_like((arr_3dsdf_points))


        gt = {'sdf': arr_3dsdf_sdf, 'id': self.ids[idx], 'ctl_path': self.ctl[idx],
              'gt_path': self.shape3d[idx]}

        return arr_3dsdf_points.float(), attributes, gt, idx





class PediatricAirway3DShapeDataset_testing(torch.utils.data.Dataset):
    def __init__(
            self,
            filename_datasource,
            filename_split,
            attributes =  ['weight', 'age', 'sex',],
            split='train',
    ):
        self.attributes = attributes

        self.split = self.load_yaml_as_dict(filename_split)[split]
        #self.split = ['1032', '1035','1036', '1041', '1042', '1043', '1045', '1047', '1048', '1049', '1050', '1052', '1053']

        self.template_split = ['1032', ]
        #self.train_split = self.load_yaml_as_dict(filename_split)['train']
        self.filename_datasource = filename_datasource
        self.covariates, self.shape3d, self.sdf3d, self.ids, self.ctl = self.read_data(self.split)
        self.template_covariates, self.template_shape3d, self.template_sdf3d, self.template_ids, self.template_ctl = self.read_data(self.template_split)

    def __len__(self):
        return len(self.ids)


    def read_data(self, split):
        self.df_data = pd.read_csv(self.filename_datasource, header=0)
        df_data_split = self.df_data.loc[self.df_data['id'].astype('str').isin(split)]
        df_data_split = df_data_split.loc[df_data_split['length'] >= 199]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.3372]
        #df_data_split = df_data_split.loc[df_data_split['depth'] > 0.49]

        # read covariates
        list_attributes = []
        for ith_attribute in self.attributes:
            arr_current_attribute = np.array(df_data_split[ith_attribute])
            list_attributes.append(arr_current_attribute)
        features = np.array(list_attributes).T

        # read target samples of the shape
        shape3d_values = np.array(df_data_split['3dshape_npy'])
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
                attributes[self.attributes[ith_attri]] = (torch.tensor([self.covariates[idx][ith_attri]]) - self.covariates[:, ith_attri].min()) /  (self.covariates[:, ith_attri].max() - self.covariates[:, ith_attri].min()) #torch.tensor([self.covariates[idx][ith_attri]]) #torch.tensor([(self.covariates[idx][ith_attri] - self.covariates[:, ith_attri].mean()) / self.covariates[:, ith_attri].std()])
                #attributes[self.attributes[ith_attri]] = torch.tensor([0.])
            else:
                attributes[self.attributes[ith_attri]] = torch.tensor([self.covariates[idx][ith_attri]])



        pv_3dshape = np.load(self.shape3d[idx]) #[:, [0, 1]] / 10
        sampled_idx = np.random.randint(0, np.array(pv_3dshape[:, [0, 1, 2]]).shape[0], 5000)
        arr_3dshape_normals = torch.from_numpy(np.array(pv_3dshape[:, [3, 4, 5]][sampled_idx]))  * (-1)#np.load(self.csa2d[idx])[:, [3, 4]]
        #arr_3dshape_mean_points = np.mean(pv_3dshape[:, [0, 1, 2]], axis=0) #(np.max(pv_3dshape[:, [0, 1, 2]], axis=0) - np.min(pv_3dshape[:, [0, 1, 2]], axis=0))/2
        #arr_3dshape_mean_points[-1] = (np.max(pv_3dshape[:,  2]) - np.min(pv_3dshape[:,2]))/2

        arr_3dshape_std_points = np.std(pv_3dshape[:, [0, 1, 2]], axis=0)
        arr_3dshape_points = torch.from_numpy(np.array(pv_3dshape[:, [0, 1, 2]][sampled_idx])).float() /60
        #scale = torch.max(torch.linalg.norm(arr_3dshape_points, dim=-1))
        #arr_3dshape_points /= scale

        #
        arr_3dsdf = np.load(self.sdf3d[idx]) #[:, [0, 1, 2]] / 10
        arr_3dsdf_sdf = arr_3dsdf[..., 3]
        arr_3dsdf_pos = arr_3dsdf[arr_3dsdf_sdf >= 0]
        arr_3dsdf_neg = arr_3dsdf[arr_3dsdf_sdf < 0]
        sampled_idx_pos = np.random.randint(0, arr_3dsdf_pos.shape[0],  2500)
        sampled_idx_neg = np.random.randint(0, arr_3dsdf_neg.shape[0], 2500)

        arr_3dsdf = np.concatenate((arr_3dsdf_pos[sampled_idx_pos], arr_3dsdf_neg[sampled_idx_neg]), axis=0)
        arr_3dsdf_points = torch.from_numpy(arr_3dsdf[..., [0, 1, 2]]).float() /60
        #arr_3dsdf_points /= scale
        arr_3dsdf_sdf = torch.from_numpy(arr_3dsdf[..., [3]]) /60
        #arr_3dsdf_points = (arr_3dsdf_points - torch.min(arr_3dsdf_points, dim=0).values) / (torch.max(arr_3dsdf_points, dim=0).values - torch.min(arr_3dsdf_points, dim=0).values)
        #arr_3dsdf_points = arr_3dsdf_points * (torch.max(arr_3dshape_points, dim=0).values - torch.min(arr_3dshape_points, dim=0).values) + torch.min(arr_3dshape_points, dim=0).values
        #arr_3dsdf_sdf = arr_3dsdf_sdf * torch.sum( (torch.max(arr_3dshape_points, dim=0).values - torch.min(arr_3dshape_points, dim=0).values) **2)**0.5
        arr_3dsdf_normals = torch.zeros_like((arr_3dsdf_points))




        noise = torch.randn_like(arr_3dshape_points) * torch.std(arr_3dshape_points, dim=0)
        #noise[:, -1] = 0
        global_samples = noise # 0.01*(noise - 0.5) + arr_3dshape_points
        arr_samples = torch.cat(( arr_3dsdf_points,  arr_3dshape_points), dim=-2)

        global_normals = torch.ones_like(arr_3dshape_normals).float()
        arr_normals = torch.cat((arr_3dsdf_normals,  arr_3dshape_normals ), dim=-2)


        sdf_local = torch.zeros((arr_3dshape_points.shape[0], 1))
        sdf_global = torch.ones((global_samples.shape[0], 1)) * (-10)
        sdf = torch.cat((arr_3dsdf_sdf, sdf_local), dim=-2)



        gt = {'sdf': sdf, 'id': self.ids[idx], 'normal': arr_normals.float(), 'ctl_path': self.ctl[idx], 'gt_path': self.sdf3d[idx]}

        return arr_samples.float(), attributes, gt, idx



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

def get_ids(filename_split, split='train'):
    split = load_yaml_as_dict(filename_split)[split]
    return split


def get_youngest_ids(filename_split, split='train'):
    split = load_yaml_as_dict(filename_split)[split]
    return split



def get_patients_for_transport(filename_datasource, filename_split, split='test_multiple'):
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


def get_data_for_id(test_idx, df_data, train_split, attributes_names):
    #train_split = get_ids(filename_split, split='train')

    train_covariates, train_shape3d, train_sdf3d, train_ids, train_ctl, _ = read_data(train_split, df_data, attributes_names)
    case_covariates, case_shape3d, case_sdf3d, case_ids, case_ctl, case_pvshape = read_data([str(test_idx)], df_data, attributes_names)

    print(case_ids + '-----')
    attributes = {}
    ori_attributes = {}
    for ith_attri in range(len(attributes_names)):
        ori_attributes[attributes_names[ith_attri]]  = case_covariates[0][ith_attri]
        if attributes_names[ith_attri] != 'depth':
            attributes[attributes_names[ith_attri]] = (torch.tensor([case_covariates[0][ith_attri]]) - train_covariates[:, ith_attri].min()) / (train_covariates[:,ith_attri].max() - train_covariates[:,ith_attri].min())
            attributes[attributes_names[ith_attri]] = attributes[attributes_names[ith_attri]] * 2 - 1
            attributes[attributes_names[ith_attri]] = attributes[attributes_names[ith_attri]].float()[None, :]
        else:
            attributes[attributes[ith_attri]] = torch.tensor([case_covariates[ith_attri]]).float()[None, :]




    pv_3dshape_points = np.array(np.load(case_shape3d[0])) # [:, [0, 1]] / 10
    pv_3dshape_normals = np.array(np.load(case_shape3d[0]))
    sampled_idx = np.random.randint(0, np.array(pv_3dshape_points[:, [0, 1, 2]]).shape[0], 20000)
    arr_3dshape_normals = torch.from_numpy(np.array(pv_3dshape_normals[:, [3, 4, 5]][sampled_idx])) * (-1)
    arr_3dshape_points = torch.from_numpy(np.array(pv_3dshape_points[:, [0, 1, 2]][sampled_idx])).float() / 60

    #
    arr_3dsdf = np.load(case_sdf3d[0])[0:250000]  # [:, [0, 1, 2]] / 10
    arr_3dsdf_sdf = arr_3dsdf[..., 3]
    arr_3dsdf_pos = arr_3dsdf[arr_3dsdf_sdf >= 2.]
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
          'pvgt_path': [case_pvshape[0]],
          'covariates': ori_attributes}

    return arr_samples.float(), attributes, gt
