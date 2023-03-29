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
from naisr import loss_pointcloud_sdf
import naisr_meshing
from naisr import *
from visualizer import plotter_evolution, plotter_evolution_comp
import pymeshfix as pmf


def get_covariates_difference(attributes, start_covariates):
    differences = {}
    for name, value in attributes.items():
        differences[name] = attributes[name] - start_covariates[name]
    return differences

def get_transported_covariates(inferred_attributes, difference_covarites):
    transported_covarites = {}
    for name, value in inferred_attributes.items():
        transported_covarites[name] = inferred_attributes[name] + difference_covarites[name]
    return transported_covarites


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    path = '/playpen-raid/jyn/NAISR/log/DeepNAIGSR_ATLAS3D_0222_256_inv/Reconstructions/Meshes/1181/surface.stl'
    gt_path = '/playpen-raid/jyn/NAISR/NAISR/examples/pediatric_airway/3dshape/1181/1181.stl'
    pv_shape = pv.read(path)
    gt_shape = pv.PolyData(pv.read(gt_path))#.smooth()
    #gt_shape = pv.PolyData()
    #gt_shape.points = gt_pv_shape.points
    #gt_shape.faces = gt_pv_shape.faces

    gt_shape.points /= 60
    fixer = pmf.MeshFix(gt_shape.triangulate())
    fixer.repair()
    gt_shape = fixer.mesh

    fixer = pmf.MeshFix(pv_shape.triangulate())
    fixer.repair()
    pv_shape = fixer.mesh
    gt_shape = gt_shape.extract_surface().smooth()


    pc = np.array(pv_shape.points)

    colors = ((pc + 2.5) / 5 * 255).astype('uint8')

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True
    pv_shape.field_data['colors'] = colors
    p.add_mesh(pv_shape,scalars='colors', rgb=True,opacity=0.6, point_size=1) #color=colors)
    p.add_mesh(gt_shape, color='grey', point_size=1, opacity=-0.9)
    p.screenshot('/home/jyn/NAISR/figs/1032_1.png', )
    p.export_html('/home/jyn/NAISR/figs/1032_1'+ '.html', backend='panel')

    gt_volume = gt_shape.volume
    pred_volume = pv_shape.volume

    '''

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
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="epoch_3000",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    naisr.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    naisr.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).to(device)
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = args.experiment_directory


    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    device = specs['Device']
    root_path = os.path.join(specs['LoggingRoot'], specs['ExperimentName'])

    latent_size = specs["CodeLength"]

    decoder = eval(specs['Network'])(
        template_attributes=specs['TemplateAttributes'],
        in_features=specs['InFeatures'],
        hidden_features=specs['HiddenFeatures'],
        hidden_layers=specs['HidenLayers'],
        out_features=specs['OutFeatures'],
        device=specs['Device'],
        backbone=specs['Backbone'],
        outermost_linear=False,
        pos_enc=specs['PosEnc'],
        latent_size=specs["CodeLength"])
    #decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            root_path, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.to(device)

    logging.info(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    transport_dir = os.path.join(root_path, ws.transport_subdir)
    if not os.path.isdir(transport_dir):
        os.makedirs(transport_dir)

    transport_meshes_dir = os.path.join(transport_dir, ws.transport_meshes_subdir)
    if not os.path.isdir(transport_meshes_dir):
        os.makedirs(transport_meshes_dir)

    transport_codes_dir = os.path.join(transport_dir, ws.transport_codes_subdir)
    if not os.path.isdir(transport_codes_dir):
        os.makedirs(transport_codes_dir)


    #cases = naisr.get_youngest_ids(specs["Split"], split='test')
    list_patient_scans = naisr.get_patients_for_transport(specs["DataSource"], specs["Split"], split='test_multiple')
    training_cases = naisr.get_ids(specs["Split"], split='train')
    import pandas as pd
    df_data = pd.read_csv(specs["DataSource"], header=0)

    list_metrics = []
    for transport_name in ['all', 'age', 'weight']:
        transport_covariate_meshes_dir = os.path.join(transport_meshes_dir, transport_name)
        if not os.path.isdir(transport_covariate_meshes_dir):
            os.makedirs(transport_covariate_meshes_dir)


        for current_patient in list_patient_scans:
            list_pred_shapepaths = []
            list_gt_shapepaths = []
            list_text  = []
            test_idx = current_patient['youngest_scan']
            transport_covariate_meshes_dir_subj = os.path.join(transport_covariate_meshes_dir, str(test_idx))
            start_arr_samples, start_attributes, start_gt = naisr.get_data_for_id(test_idx, df_data, training_cases,
                                                                                  specs["Attributes"])

            path_3dshape = os.path.join(transport_covariate_meshes_dir_subj, "surface.stl")
            list_pred_shapepaths.append(path_3dshape)
            list_gt_shapepaths.append(start_gt['pvgt_path'][0])
            list_text.append(('age: ' + str(start_gt['covariates'][0][1]),
                              'weight: ' + str(start_gt['covariates'][0][0])))

            # transporting to other covariates
            other_scans= current_patient['other_scans']
            for ith_scan_to_transp in other_scans:
                transport_covariate_meshes_dir_subj = os.path.join(transport_covariate_meshes_dir, str(ith_scan_to_transp))
                arr_samples, attributes, gt = naisr.get_data_for_id(ith_scan_to_transp,
                                                                                  df_data,
                                                                                  training_cases,
                                                                                  specs["Attributes"])
                path_3dshape = os.path.join(transport_covariate_meshes_dir_subj, "surface.stl")
                list_pred_shapepaths.append(path_3dshape)
                list_gt_shapepaths.append(gt['pvgt_path'][0])
                list_text.append(('age: ' + str(gt['covariates'][0][1]),
                                  'weight: ' + str(gt['covariates'][0][0])))

            savepath = os.path.join(transport_covariate_meshes_dir, str(test_idx))
            print(list_gt_shapepaths)

            plotter_evolution_comp(list_pred_shapepaths, list_gt_shapepaths, savepath, list_text=list_text)

    '''