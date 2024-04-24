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

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    attributes,
    stat,
    clamp_dist,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).to(device)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).to(device)

    latent.requires_grad = True
    # define attributes
    optimizer_lat = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    #loss_l1 = torch.nn.L1Loss()
    loss_fn = loss_pointcloud_sdf


    for e in range(num_iterations):

        decoder.eval()
        xyz = test_sdf#[:, :, 0:3]
        adjust_learning_rate(lr, optimizer_lat, e, decreased_by, adjust_lr_every)
        optimizer_lat.zero_grad()

        latent_inputs = latent.expand(xyz.shape[-2], -1)

        model_output = decoder(xyz, attributes, latent_inputs[None, :, :], training=False, testing=False)
        gt['sdf'] = torch.clamp(gt['sdf'], -1., 1.)
        specs['Loss']["whether_icon"] = False
        specs['Loss']['whether_disentangle'] = False
        specs['Loss']['whether_inv'] = False
        specs['Loss']["whether_inv_sampling"] = False
        losses = loss_fn(model_output, gt, latent_inputs, epoch=e, dict_losses=specs['Loss'])
        loss = 0
        for name, il in losses.items():
            if 'code_reg'  in name:
                loss += (il*10)
            else:
                loss += il

        loss.backward()
        optimizer_lat.step()

        if e % 50 == 0:
            logging.info(loss.cpu().data.numpy())
            logging.info(e)
            logging.info(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='examples/pediatric_airway/naivf_deepnaisr.json',
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
        "-i",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=True,
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
    num_dim = specs["InFeatures"]

    root_path = os.path.join(specs['LoggingRoot'], specs['ExperimentName'])

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

    latent_size = specs["CodeLength"]
    #decoder = arch.Decoder(num_atc_parts=specs["NumAtcParts"], do_sup_with_part=specs["TrainWithParts"])
    #decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            root_path, ws.model_params_subdir, args.checkpoint + ".pth"
        ), map_location=torch.device(device)
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.to(device)

    logging.info(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_withcov_dir = os.path.join(root_path, ws.reconstructions_withcov_subdir)
    if not os.path.isdir(reconstruction_withcov_dir):
        os.makedirs(reconstruction_withcov_dir)

    reconstruction_withcov_meshes_dir = os.path.join(reconstruction_withcov_dir, ws.reconstruction_withcov_meshes_subdir)
    if not os.path.isdir(reconstruction_withcov_meshes_dir):
        os.makedirs(reconstruction_withcov_meshes_dir)

    reconstruction_withcov_codes_dir = os.path.join(reconstruction_withcov_dir, ws.reconstruction_withcov_codes_subdir)
    if not os.path.isdir(reconstruction_withcov_codes_dir):
        os.makedirs(reconstruction_withcov_codes_dir)

    # loading dataset
    shapetype = specs["Class"]
    filename_dataset = specs["DataSource"]
    if shapetype == 'Airway':
        cases = naisr.get_airway_ids(specs["Split"], split='test')
        training_cases = naisr.get_airway_ids(specs["Split"], split='train')
        load_one_case = naisr.get_airway_data_for_id
        df_data = pd.read_csv(filename_dataset)
    elif shapetype == 'starman':
        cases = naisr.get_starman_ids(filename_dataset, 'test')
        training_cases = naisr.get_starman_ids(filename_dataset, 'train')
        load_one_case = naisr.get_starman_data_for_id
        df_data = pd.read_csv(filename_dataset['test'])
    elif shapetype == 'ADNI':
        cases = naisr.get_adni_ids(specs["Split"], split='test')
        training_cases = naisr.get_adni_ids(specs["Split"], split='train')
        load_one_case = naisr.get_adni_data_for_id
        df_data = pd.read_csv(filename_dataset)



    list_metrics = []
    for ii, test_idx in enumerate(cases):
        arr_samples, attributes, gt  = load_one_case(test_idx, df_data, training_cases, specs["Attributes"], stage='test')

        arr_samples = arr_samples.to(device)
        attributes = {key: value.to(device).float() for key, value in attributes.items()}
        for key, value in gt.items():
            if isinstance(value, torch.Tensor):
                gt[key] = value.to(device)

        reconstruction_withcov_meshes_dir_subj = os.path.join(reconstruction_withcov_meshes_dir, str(test_idx))
        cond_mkdir(reconstruction_withcov_meshes_dir_subj)

        if args.skip:
            if shapetype != 'starman':
                stl_path = os.path.join(reconstruction_withcov_meshes_dir_subj, 'surface.stl')
                if os.path.exists(stl_path):
                    continue
            else:
                stl_path = os.path.join(reconstruction_withcov_meshes_dir_subj, 'surface.vtk')
                if os.path.exists(stl_path):
                    continue

        #reconstruction_codes_dir_subj = os.path.join(reconstruction_codes_dir, str(gt['id'][0]))
        #cond_mkdir(reconstruction_codes_dir_subj)
        #print( str(gt['id']) + '-----')
        for k in range(repeat):

            logging.info("reconstructing {}".format(test_idx))

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                arr_samples,
                attributes,
                0.01,  # [emp_mean,emp_var],
                0.1,
                lr=5e-3,
                l2reg=True,
            )
            logging.info("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.info("current_error avg: {}".format((err_sum / (ii + 1))))


        print('attributes: ' + str(attributes))
        if not save_latvec_only:
            latent_filename = os.path.join(reconstruction_withcov_codes_dir, str(test_idx) + ".pth")
            #print(latent.shape)
            start = time.time()
            with torch.no_grad():
                #print(reconstruction_meshes_dir_subj)
                dict_metrics = naisr_meshing.create_mesh_reconstruction(shapetype)(decoder,
                                                                                   latent[None, :, ],
                                                                                 attributes,
                                                                                 gt,
                                                                                 reconstruction_withcov_meshes_dir_subj,
                                                                                 output_type='model_out',
                                                                                   dim=num_dim,
                                                                                 N=256,
                                                                                            shapetype=shapetype,
                                                                                 device=specs['Device'],
                                                                                 EVALUATE=True)
                #naisr_meshing.save_to_ply(decoder, latent[None, :, ], attributes, reconstruction_withcov_meshes_dir_subj, device)

                if not os.path.exists(os.path.dirname(latent_filename)):
                    os.makedirs(os.path.dirname(latent_filename))
                torch.save(latent.unsqueeze(0), latent_filename)
                inferred_covariate_filename = os.path.join(reconstruction_withcov_codes_dir,
                                                           str(test_idx) + "_covariate.pth")

                list_metrics.append(dict_metrics)
    # save
    pd.DataFrame.from_records(list_metrics).to_csv(os.path.join(reconstruction_withcov_meshes_dir_subj, 'metrics.csv'))



