import numpy as np

import pyvista as pv
from pyvista import examples
import argparse
import logging
import json
import numpy as np
import os

import pandas as pd
import trimesh
import csv

import naisr
import naisr.workspace as ws
import glob
import re
import torch.utils.data as data_utils
import torch
import time
import skimage.measure as measure


'''
vis gradients: covariant-wise landmark
'''


def vis_covariatewise_landmark(args, specs):
    '''
    read network setting and IO settings
    '''
    print(args)
    pos_enc = args.posenc
    backbone = args.backbone
    shapetype = args.shape
    prefix = args.prefix
    in_features = int(args.dimension)
    experiment_name = args.experiment_directory
    print(experiment_name)

    # get class and path
    # shapetype = specs["Class"]
    data_source = specs['DataSource']
    root_path = os.path.join(specs["LoggingRoot"], experiment_name)

    centers = []
    for i in range(5):
        for j in range(5):
            centers.append(np.array([-i, j, 0]) * 3)
    centers = np.array(centers).astype('float')

    list_ringradius = [0.1, 0.3, 0.5, 0.7, 0.9]
    list_crossectionradius = [0.05, 0.15, 0.25, 0.35, 0.45]

    list_attributes = []
    for i_ringradius in list_ringradius:
        for i_csr in list_crossectionradius:
            # if i_ringradius > i_csr:
            list_attributes.append({'ringradius': i_ringradius,
                                    'crosssectionradius': i_csr})

    rootdir_train = os.path.join(root_path, 'train')
    rootdir_test = os.path.join(root_path, 'test')

    list_train = os.listdir(rootdir_train)
    list_test = os.listdir(rootdir_test)

    list_attributes_with_set = []
    for attributes in list_attributes:
        current_attri = attributes
        filename = ''
        for ith_attri in attributes.keys():
            filename = filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri], 2))
        filename += '.stl'
        if filename in list_train:
            current_attri['set'] = "train"
        elif filename in list_test:
            current_attri['set'] = 'test'
        else:
            current_attri['set'] = 'missing'
        list_attributes_with_set.append(current_attri)

    color = {'train': 'lightblue', 'test': 'pink'}
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background = 'white'
    pv.global_theme.interactive = True

    for i in range(len(list_attributes_with_set)):
        print(list_attributes_with_set[i])
        attributes = list_attributes_with_set[i]
        filename = ''
        for ith_attri in ['ringradius', 'crosssectionradius']:
            filename = filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri], 2))





'''
vis importance
'''




def vis_importance(args, specs):
    '''
    read network setting and IO settings
    '''
    print(args)
    pos_enc = args.posenc
    backbone = args.backbone
    shapetype = args.shape
    prefix = args.prefix
    in_features = int(args.dimension)
    experiment_name = args.experiment_directory
    print(experiment_name)

    # get class and path
    #shapetype = specs["Class"]
    data_source = specs['DataSource']
    root_path = os.path.join(specs["LoggingRoot"], experiment_name)

    centers = []
    for i in range(5):
        for j in range(5):
            centers.append(np.array([-i, j, 0]) * 3)
    centers = np.array(centers).astype('float')





    list_ringradius = [0.1, 0.3, 0.5, 0.7, 0.9]
    list_crossectionradius = [0.05, 0.15, 0.25, 0.35, 0.45]

    list_attributes = []
    for i_ringradius in list_ringradius:
        for i_csr in list_crossectionradius:
            #if i_ringradius > i_csr:
            list_attributes.append({'ringradius': i_ringradius,
                                        'crosssectionradius': i_csr})


    rootdir_train = os.path.join(root_path, 'train')
    rootdir_test = os.path.join(root_path, 'test')

    list_train = os.listdir(rootdir_train)
    list_test = os.listdir(rootdir_test)

    list_attributes_with_set = []
    for attributes in list_attributes:
        current_attri = attributes
        filename = ''
        for ith_attri in attributes.keys():
            filename = filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri], 2))
        filename +=  '.stl'
        if filename in list_train:
            current_attri['set'] = "train"
        elif filename in list_test:
            current_attri['set'] = 'test'
        else:
            current_attri['set'] = 'missing'
        list_attributes_with_set.append(current_attri)

    color = {'train': 'lightblue','test': 'pink'}
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", off_screen=True,window_size=[1024, 1024])
    pv.global_theme.background = 'white'
    pv.global_theme.interactive = True

    for i in range(len(list_attributes_with_set)):
        print(list_attributes_with_set[i])
        attributes = list_attributes_with_set[i]
        filename = ''
        for ith_attri in ['ringradius', 'crosssectionradius']:
            filename = filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri], 2))












'''
vis evolution
'''


def vis_evolution(args, specs):
    '''
    read network setting and IO settings
    '''
    print(args)
    pos_enc = args.posenc
    backbone = args.backbone
    shapetype = args.shape
    prefix = args.prefix
    in_features = int(args.dimension)
    experiment_name = args.experiment_directory
    print(experiment_name)

    # get class and path
    #shapetype = specs["Class"]
    data_source = specs['DataSource']
    root_path = os.path.join(specs["LoggingRoot"], experiment_name)

    centers = []
    for i in range(5):
        for j in range(7):
            centers.append(np.array([-i, j, 0]) * 2)
    centers = np.array(centers).astype('float')

    list_ringradius = [0.1, 0.3, 0.5, 0.7, 0.85]
    list_crossectionradius = [0.05, 0.1, 0.15, 0.25, 0.35, 0.4, 0.45]

    list_attributes = []
    for i_ringradius in list_ringradius:
        for i_csr in list_crossectionradius:
            #if i_ringradius > i_csr:
            list_attributes.append({'ringradius': i_ringradius,
                                        'crosssectionradius': i_csr})

    rootdir_train = os.path.join(root_path, 'train')
    rootdir_test = os.path.join(root_path, 'test')

    list_train = os.listdir(rootdir_train)
    list_test = os.listdir(rootdir_test)

    list_attributes_with_set = []
    for attributes in list_attributes:
        current_attri = attributes
        filename = ''
        for ith_attri in attributes.keys():
            filename = filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri], 2))
        filename +=  '.stl'
        if filename in list_train:
            current_attri['set'] = "train"
        elif filename in list_test:
            current_attri['set'] = 'test'
        else:
            current_attri['set'] = 'missing'
        list_attributes_with_set.append(current_attri)

    color = {'train': 'lightblue','test': 'pink'}
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", off_screen=True,window_size=[1024, 1024])
    pv.global_theme.background = 'white'
    pv.global_theme.interactive = True

    for i in range(len(list_attributes_with_set)):
        print(list_attributes_with_set[i])
        attributes = list_attributes_with_set[i]
        filename = ''
        for ith_attri in ['ringradius', 'crosssectionradius']:
            filename = filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri], 2))


        reconstructed_mesh_filename = os.path.join(root_path, attributes['set'], filename + '.stl')

        logging.debug('reconstructed mesh is "' + reconstructed_mesh_filename + '"')

        if os.path.isfile(reconstructed_mesh_filename):
            reconstruction = pv.read(reconstructed_mesh_filename)
            reconstruction.points+= centers[i]
            p.add_mesh(reconstruction, color=color[attributes['set']], specular=1.0, specular_power=10, label=str(attributes))
        else:
            if attributes['ringradius']  > attributes['crosssectionradius'] and (attributes['ringradius']  + attributes['crosssectionradius'] <=0.95):
                p.add_point_labels(points=np.array([centers[i]]), labels=['missing'],  italic=True, font_size=20,
                            point_color='red', point_size=20,
                            render_points_as_spheres=True,
                            always_visible=True, shadow=True)
            continue
    p.view_vector((5.0, 2, 3))
    p.add_floor('-z', lighting=True, color='grey', pad=1.0)
    p.enable_shadows()
    p.screenshot(os.path.join(root_path, 'screenshot.png'))
    p.export_html(os.path.join(root_path, 'vis.html'), backend='panel')
    p.close()




if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")


    arg_parser.add_argument(
        "--networksetting",
        "-n",
        dest="networksetting",
        default='examples/toy/torus/baseline.json',
        #required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )

    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_siren_hinge",
        #required=True,
        help="experiment directory name",
    )

    arg_parser.add_argument(
        "--posenc",
        "-p",
        dest="posenc",
        help="position encoding",
        action='store_true'
    )

    arg_parser.add_argument(
        "--backbone",
        "-b",
        dest="backbone",
        default='siren',
        #required=True,
        help="mlp or siren",
    )

    arg_parser.add_argument(
        "--shape",
        "-s",
        dest="shape",
        default='torus',
        #required=True,
        help="shape ellipsoid or torus",
    )

    arg_parser.add_argument(
        "--dim",
        "-d",
        dest="dimension",
        default=3,
        #required=True,
        type=int,
        help="shape ellipsoid or torus",
    )

    arg_parser.add_argument(
        "--prefix",
        "-f",
        dest="prefix",
        default="NAIVF_TEM_VEC_AnalFit_1110",
        #required=True,
        help='prefix of experiment name',
    )


    #naisr.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    #naisr.configure_logging(args)
    pos_enc = False #args.posenc
    backbone = args.backbone
    shapetype = args.shape
    prefix = args.prefix
    in_features = int(args.dimension)
    experiment_name = args.experiment_directory

    specs_filename = args.networksetting
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    specs = json.load(open(specs_filename))
    print(args.experiment_directory)
    vis_evolution(args, specs,)
