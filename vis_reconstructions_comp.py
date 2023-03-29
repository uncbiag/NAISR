#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch

import naisr
import naisr.workspace as ws
from utils import cond_mkdir
import pandas as pd
import naisr_meshing
from naisr import loss_pointcloud_sdf
from naisr import *
from visualizer import *
methods =[ #'BaseDeepSDF',
           'DeepSDF',
           #'BaseASDF',
           'A-SDF',
           #'BaseDIT',
            'DIT',
            'NDF',
           'NAISR with cov',
           'NAISR w/o cov',

           ]# "DIT", ] #'NDF']
root_identifier = {
'BaseDeepSDF': "BaseDeepSDF_ATLAS3D_0221",
'BaseASDF': "BaseASDF_ATLAS3D_0221",
'BaseDIT': "BaseDIT_ATLAS3D_0221",
'DeepSDF': "DeepSDF_ATLAS3D_0220_512_siren",
'A-SDF': "ASDF_ATLAS3D_0220_512_siren",
'DIT': "DIT_ATLAS3D_0220_512_siren",
'NDF': "NDF_ATLAS3D_0220_512_siren",
'NAISR with cov': "DeepNAIGSR_ATLAS3D_0222_256_inv",
'NAISR w/o cov': "DeepNAIGSR_ATLAS3D_0222_256_inv",
'Ours': "DeepNAIGSR_ATLAS3D_0222_256_inv",
'NAISR_no_regu': "DeepNAISR_ATLAS3D_0224_256_no_regu",
                   }



if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='examples/pediatric_airway/naivf_deepnaigsr.json',
        required=False,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    naisr.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    naisr.configure_logging(args)


    specs_filename = args.experiment_directory


    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    device = specs['Device']

    cases = naisr.get_ids(specs["Split"], split='test')
    training_cases = naisr.get_ids(specs["Split"], split='train')
    import pandas as pd
    df_data = pd.read_csv(specs["DataSource"], header=0)

    list_metrics = []


    for ii, test_idx in enumerate(cases):
        list_texts = []
        list_mesh_case = []
        arr_samples, attributes, gt = naisr.get_data_for_id(test_idx, df_data, training_cases, specs["Attributes"])
        list_mesh_case.append(gt['pvgt_path'][0])
        list_texts.append('Groundtruth')
        for i_method in methods:
            root_path = os.path.join(specs['LoggingRoot'], root_identifier[i_method])

            if 'with cov' in i_method:
                reconstruction_dir = os.path.join(root_path, ws.reconstructions_withcov_subdir)
            else:
                reconstruction_dir = os.path.join(root_path,  ws.reconstructions_subdir)
            reconstruction_meshes_dir = os.path.join(reconstruction_dir, ws.reconstruction_meshes_subdir)
            current_mesh_path= os.path.join(reconstruction_meshes_dir, str(test_idx), 'surface.stl')

            vis_comp_dir = os.path.join(root_path, ws.vis_comp_subdir)
            cond_mkdir(vis_comp_dir)
            vis_comp_meshes_dir = os.path.join(vis_comp_dir, ws.vis_comp_meshes_subdir)
            cond_mkdir(vis_comp_meshes_dir)

            current_mesh_path= os.path.join(reconstruction_meshes_dir, str(test_idx), 'surface.stl')

            list_mesh_case.append(current_mesh_path)
            list_texts.append(i_method)

        savepath = os.path.join(vis_comp_meshes_dir, str(test_idx))
        plotter_evolution_for_methods(list_mesh_case, savepath, list_text=list_texts)


