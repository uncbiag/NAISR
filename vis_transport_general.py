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


def get_covariates_difference(attributes, start_covariates, diff_names='all'):
    differences = {}
    for name, value in attributes.items():
        if diff_names == 'all' or name in diff_names:
            differences[name] = attributes[name] - start_covariates[name]
        else:
            differences[name] = 0
    return differences

def get_transported_covariates(inferred_attributes, difference_covarites):
    transported_covarites = {}
    for name, value in inferred_attributes.items():
        transported_covarites[name] = inferred_attributes[name] + difference_covarites[name].to(inferred_attributes[name].device)
    return transported_covarites

def transport_covariates(attributes, inferred_attributes, start_covariates, diff_names='all'):
    difference_covarites = get_covariates_difference(attributes, start_covariates, diff_names=diff_names)
    transported_covarites = get_transported_covariates(inferred_attributes, difference_covarites)
    return transported_covarites



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
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
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

    transport_general_dir = os.path.join(root_path, ws.transport_general_subdir)
    if not os.path.isdir(transport_general_dir):
        os.makedirs(transport_general_dir)

    transport_general_meshes_dir = os.path.join(transport_general_dir, ws.transport_general_meshes_subdir)
    if not os.path.isdir(transport_general_meshes_dir):
        os.makedirs(transport_general_meshes_dir)

    transport_general_codes_dir = os.path.join(transport_general_dir, ws.transport_general_codes_subdir)
    if not os.path.isdir(transport_general_codes_dir):
        os.makedirs(transport_general_codes_dir)


    #cases = naisr.get_youngest_ids(specs["Split"], split='test')
    list_patient_scans = naisr.get_patients_for_transport(specs["DataSource"], specs["Split"], split='test_multiple')
    training_cases = naisr.get_ids(specs["Split"], split='train')
    import pandas as pd
    df_data = pd.read_csv(specs["DataSource"], header=0)

    list_metrics = []

    for transport_name in ['all', 'age', 'weight']:
        transport_general_covariate_meshes_dir = os.path.join(transport_general_meshes_dir, transport_name)
        if not os.path.isdir(transport_general_covariate_meshes_dir):
            os.makedirs(transport_general_covariate_meshes_dir)


        for current_patient in list_patient_scans:
            list_color = []
            list_pred_shapepaths = []
            list_gt_shapepaths = []
            list_text  = []
            test_idx = current_patient['youngest_scan']
            transport_general_covariate_meshes_dir_subj = os.path.join(transport_general_covariate_meshes_dir, str(test_idx))
            start_arr_samples, start_attributes, start_gt = naisr.get_data_for_id(test_idx,
                                                                                  df_data,
                                                                                  training_cases,
                                                                                  specs["Attributes"])

            path_3dshape = os.path.join(transport_general_covariate_meshes_dir_subj, "surface.stl")
            list_pred_shapepaths.append(path_3dshape)
            list_gt_shapepaths.append(start_gt['pvgt_path'][0])
            list_text.append(start_gt['covariates'])
            batch_lat = load_transport_vectors(transport_general_codes_dir, start_gt['id'][0], device)
            infered_start_attributes = load_inferred_covariates(transport_general_codes_dir, start_gt['id'][0], device)
            current_color = naisr_meshing.revert_points_to_template(decoder,
                                                                    batch_lat,
                                                                    infered_start_attributes,
                                                                    transport_general_covariate_meshes_dir_subj,
                                                                    device)
            list_color.append(current_color)
            # transporting to other covariates
            other_scans= current_patient['other_scans']
            for ith_scan_to_transp in other_scans:
                transport_general_covariate_meshes_dir_subj = os.path.join(transport_general_covariate_meshes_dir, str(ith_scan_to_transp))
                arr_samples, attributes, gt = naisr.get_data_for_id(ith_scan_to_transp,
                                                                                  df_data,
                                                                                  training_cases,
                                                                                  specs["Attributes"])
                path_3dshape = os.path.join(transport_general_covariate_meshes_dir_subj, "surface.stl")
                list_pred_shapepaths.append(path_3dshape)
                list_gt_shapepaths.append(gt['pvgt_path'][0])
                list_text.append(gt['covariates'])

                new_covariates = transport_covariates(attributes, infered_start_attributes, start_attributes, diff_names='all')

                current_color = naisr_meshing.revert_points_to_template(decoder,
                                                                        batch_lat,
                                                                        new_covariates,
                                                                        transport_general_covariate_meshes_dir_subj,
                                                                        device)
                list_color.append(current_color)


            savepath = os.path.join(transport_general_covariate_meshes_dir, str(test_idx))
            print(list_gt_shapepaths)

            plotter_evolution_comp(list_pred_shapepaths, list_gt_shapepaths, savepath, list_text=list_text, print_on_figure=False,)
            plotter_evolution(list_pred_shapepaths, savepath, list_text=list_text, list_colors=list_color)
