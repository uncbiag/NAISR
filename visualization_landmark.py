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


def load_model(args, specs):
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
    template_attributes = specs["TemplateAttributes"]
    print(experiment_name)


    if specs['Network'] == "BaselineVF":
        # load model
        model = naisr.BaselineVF(
            template_attributes=template_attributes,
            in_features=in_features,
            hidden_features=specs['HiddenFeatures'],
            hidden_layers=specs['HidenLayers'],
            out_features=specs['OutFeatures'],
            device=specs['Device'],
            backbone=backbone,
            outermost_linear=False,
            pos_enc=pos_enc)
    elif specs['Network'] == "Baseline":
        # load model
        model = naisr.Baseline(
            template_attributes=template_attributes,
            in_features=in_features,
            hidden_features=specs['HiddenFeatures'],
            hidden_layers=specs['HidenLayers'],
            out_features=specs['OutFeatures'],
            device=specs['Device'],
            backbone=backbone,
            outermost_linear=False,
            pos_enc=pos_enc)
    elif specs['Network'] == 'LipNAIVF_withtempl':
        # load model
        model = naisr.LipNAIVF_withtempl(
            template_attributes=template_attributes,
            in_features=in_features,
            hidden_features=specs['HiddenFeatures'],
            hidden_layers=specs['HidenLayers'],
            out_features=specs['OutFeatures'],
            device=specs['Device'],
            backbone=backbone,
            outermost_linear=False,
            pos_enc=pos_enc)

    elif specs['Network'] == 'NAIVF_withtempl':
        # load model
        model = naisr.NAIVF_withtempl(
            template_attributes=template_attributes,
            in_features=in_features,
            hidden_features=specs['HiddenFeatures'],
            hidden_layers=specs['HidenLayers'],
            out_features=specs['OutFeatures'],
            device=specs['Device'],
            backbone=backbone,
            outermost_linear=False,
            pos_enc=pos_enc)


    root_path = os.path.join(specs["LoggingRoot"], experiment_name)
    checkpoint_path = os.path.join(root_path, 'checkpoints', 'model_final.pth')
    print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(specs['Device'])

    return model




'''
vis gradients: covariant-wise landmark
'''


def vis_covariatewise_landmark(model, args, specs):
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


    list_ringradius = [0.3, 0.5, 0.7,]
    list_crossectionradius = [0.05, 0.1, 0.15, 0.25, 0.35, 0.4, 0.45]

    list_attributes = []
    for i_ringradius in list_ringradius:
        for i_csr in list_crossectionradius:
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


    centers = []
    for i in range(5):
        for j in range(len(list_attributes_with_set)):
            centers.append(np.array([-i, j, 0]) * 2)
    centers = np.array(centers).astype('float')



    color = {'train': 'lightblue','test': 'pink'}
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", off_screen=True,window_size=[1024, 1024])
    pv.global_theme.background = 'white'
    pv.global_theme.interactive = True


    ith_shape = 0
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

            if 'crosssectionradius' in attributes.keys():
                R = attributes["crosssectionradius"] + attributes["ringradius"]
                r = attributes["ringradius"] - attributes["crosssectionradius"]
                attributes["crosssectionradius"] = r
                attributes["ringradius"] = R

            current_attributes = {'ringradius': torch.tensor([attributes['ringradius']]).float().to(specs['Device']),
                                  'crosssectionradius': torch.tensor([attributes['crosssectionradius']]).float().to(specs['Device']),}
            model_output = model(torch.from_numpy(reconstruction.points[None, :, :]).to(specs['Device']), current_attributes)
            model_maps = model_output['model_map']
            #

            current_shape = reconstruction.points + np.array([float(ith_shape), 0., 0.])*2
            p.add_mesh(current_shape, color=color[attributes['set']], specular=1.0, specular_power=10, label=str(attributes), opacity=0.4)
            current_idx = 0
            for name, map in model_maps.items():
                if '_grad' in name and (name != 'initial'):
                    print(name)
                    current_idx += 1
                    current_shape = reconstruction.points + np.array([float(ith_shape), float(current_idx), 0.]) * 2

                    current_map = np.abs(map.detach().cpu().numpy())
                    current_map = (current_map - current_map.min()) / (current_map.max() - current_map.min() + 1e-5)
                    p.add_mesh(current_shape, scalars=current_map.squeeze(), cmap="afmhot")
            ith_shape += 1
                #else:
                #    p.add_mesh(current_shape, color=color[attributes['set']], specular=1.0, specular_power=10,
                #               label=str(attributes))

    p.view_vector((5.0, 2, 3))
    p.add_floor('-z', lighting=True, color='grey', pad=1.0)
    #p.enable_shadows()
    p.screenshot(os.path.join(root_path, 'screenshot.png'))
    p.export_html(os.path.join(root_path, 'disentanglement.html'), backend='panel')
    p.close()





if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")


    arg_parser.add_argument(
        "--networksetting",
        "-n",
        dest="networksetting",
        default='examples/toy/torus/naivf_tem_vec.json',
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

    model = load_model(args, specs)




    vis_covariatewise_landmark(model, args, specs,)
