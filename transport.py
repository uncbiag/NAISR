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
import naisr_meshing
from naisr import loss_pointcloud_sdf
from naisr import *

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
        transported_covarites[name] = inferred_attributes[name] + difference_covarites[name]
    return transported_covarites

def transport_covariates(attributes, inferred_attributes, start_attributes, diff_names='all'):
    difference_covarites = get_covariates_difference(attributes, start_attributes, diff_names=diff_names)
    transported_covarites = get_transported_covariates(inferred_attributes, difference_covarites)
    return transported_covarites




def transport(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    attributes,
        gt,
    stat,
    clamp_dist,
    num_samples=30000,
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
    # define latent code
    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).to(device)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).to(device)
    latent.requires_grad = True
    # define attributes
    attributes_unknown = {}
    for name, value in attributes.items():
        attributes_unknown[name] = torch.ones(1, 1).normal_(mean=0, std=stat).to(device)
        #attributes_unknown[name].requires_grad = True
    #optimizer = torch.optim.Adam([latent] + [attributes_unknown[name] for name in attributes.keys()], lr=lr)

    optimizer_lat = torch.optim.SGD([latent], lr=lr)
    optimizer_atc = torch.optim.SGD([attributes_unknown[name] for name in attributes.keys()], lr=lr*1)

    loss_num = 0
    #loss_l1 = torch.nn.L1Loss()
    loss_fn = loss_pointcloud_sdf
    '''
    for e in range(num_iterations):

        decoder.eval()
        xyz = test_sdf[:, :, 0:3]

        #adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        adjust_learning_rate(lr, optimizer_lat, e, decreased_by, adjust_lr_every)
        adjust_learning_rate(lr*1, optimizer_atc, e, decreased_by, adjust_lr_every)


        #optimizer.zero_grad()


        latent_inputs = latent.expand(num_samples, -1)
        #atc = attributes[None, :].expand(num_samples, -1)
        #inputs = torch.cat([latent_inputs, xyz, atc], 1).to(device)
        model_output = decoder(xyz, attributes_unknown, latent_inputs[None, :, :], testing=False)
        gt['sdf'] = torch.clamp(gt['sdf'], -1., 1.)
        optimizer_lat.zero_grad()
        optimizer_atc.zero_grad()
        #specs['Loss']['whether_disentangle']=False
        #specs['Loss']["whether_eikonal"]=False
        losses = loss_fn(model_output, gt, latent_inputs, epoch=e, dict_losses=specs['Loss'])
        loss = 0
        for name, il in losses.items():
            #if 'padding' not in name:
            loss += il #losses['sdf']+ losses['normal_constraint'] + losses['inter_constraint'] #+ loss['eikonal']
        #loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += torch.mean(latent.pow(2))*10

        loss.backward()
        #optimizer.step()
        optimizer_atc.step()
        optimizer_lat.step()

        if e % 50 == 0:
            logging.info(loss.cpu().data.numpy())
            logging.info(e)
            logging.info(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent, attributes_unknown

    '''
    '''
    for e in range(num_iterations//8):
        latent.requires_grad = False
        decoder.eval()
        xyz = test_sdf[:, :, 0:3]

        adjust_learning_rate(lr*10, optimizer_atc, e, decreased_by, adjust_lr_every)

        latent_inputs = latent.expand(num_samples, -1)
        model_output = decoder(xyz, attributes_unknown, torch.zeros_like(latent_inputs[None, :, :]), testing=True)
        gt['sdf'] = torch.clamp(gt['sdf'], -1., 1.)
        optimizer_atc.zero_grad()

        losses = loss_fn(model_output, gt, torch.zeros_like(latent_inputs[None, :, :]), epoch=e, dict_losses=specs['Loss'])
        loss = 0
        for name, il in losses.items():
            #if 'padding' in name:
            loss += il
        #if l2reg:
        #    loss += torch.mean(latent.pow(2))
        loss.backward()
        optimizer_atc.step()

        if e % 50 == 0:
            logging.info(loss.cpu().data.numpy())
            logging.info(e)
            logging.info(latent.norm())
        loss_num = loss.cpu().data.numpy()

    for name, value in attributes.items():
        attributes_unknown[name].requires_grad = False
    latent.requires_grad = True

    print(attributes_unknown)
    '''
    for e in range(num_iterations):

        decoder.eval()
        xyz = test_sdf[:, :, 0:3]

        adjust_learning_rate(lr, optimizer_lat, e, decreased_by, adjust_lr_every)
        latent_inputs = latent.expand(num_samples, -1)
        model_output = decoder(xyz, attributes, latent_inputs[None, :, :], testing=False,training=False)
        gt['sdf'] = torch.clamp(gt['sdf'], -1., 1.)
        optimizer_lat.zero_grad()

        losses = loss_fn(model_output, gt, latent_inputs, epoch=e, dict_losses=specs['Loss'])
        loss = 0
        specs['Loss']['whether_disentangle'] = False
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

    return loss_num, latent, attributes#_unknown




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
        ),map_location=torch.device(device)
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
            test_idx = current_patient['youngest_scan']
            start_arr_samples, start_attributes, start_gt  = naisr.get_data_for_id(test_idx, df_data, training_cases, specs["Attributes"])
            start_arr_samples = start_arr_samples.to(device)
            start_attributes = {key: value.to(device) for key, value in start_attributes.items()}

            for key, value in start_gt.items():
                if isinstance(value, torch.Tensor):
                    start_gt[key] = value.to(device)

            transport_covariate_meshes_dir_subj = os.path.join(transport_covariate_meshes_dir, str(test_idx))
            cond_mkdir(transport_covariate_meshes_dir_subj)

            for k in range(repeat):

                logging.info("transporting {}".format(test_idx))

                start = time.time()
                err, start_latent, infered_start_attributes = transport(
                    decoder,
                    int(args.iterations),
                    latent_size,
                    start_arr_samples,
                    start_attributes,
                    start_gt,
                    0.01,  # [emp_mean,emp_var],
                    0.1,
                    num_samples=30000,
                    lr=5e-3,
                    l2reg=True,
                )

                #for name, value in infered_start_attributes.items():
                #    infered_start_attributes[name] = torch.tanh(infered_start_attributes[name])

                print('attributes: ' + str(start_attributes))
                print('inferred_attributes: ' + str(infered_start_attributes))

                logging.info("transport time: {}".format(time.time() - start))
                err_sum += err
                logging.info("current_error avg: {}".format((err_sum / (k + 1))))

                #logging.info("latent: {}".format(latent.detach().cpu().numpy()))

            if not save_latvec_only:
                latent_filename = os.path.join(transport_codes_dir, str(test_idx) + ".pth")
                inferred_covariate_filename = os.path.join(transport_codes_dir,
                                                           str(test_idx) + "_covariate.pth")

                start = time.time()
                with torch.no_grad():
                    dict_metrics = naisr_meshing.create_mesh_3dairway_reconstruction(decoder,
                                                                                     start_latent[None, :, ],
                                                                                     infered_start_attributes,
                                                                                     start_gt,
                                                                                     transport_covariate_meshes_dir_subj,
                                                                                     output_type='model_out', N=256,
                                                                                     device=specs['Device'],
                                                                                     EVALUATE=True)

                    if not os.path.exists(os.path.dirname(latent_filename)):
                        os.makedirs(os.path.dirname(latent_filename))
                    torch.save(start_latent.unsqueeze(0), latent_filename)
                    torch.save(infered_start_attributes, inferred_covariate_filename)
                    list_metrics.append(dict_metrics)



            # transporting to other covariates
            other_scans= current_patient['other_scans']
            for ith_scan_to_transp in other_scans:
                arr_samples, attributes, gt  = naisr.get_data_for_id(ith_scan_to_transp, df_data, training_cases, specs["Attributes"])
                arr_samples = arr_samples.to(device)
                for key, value in gt.items():
                    if isinstance(value, torch.Tensor):
                        gt[key] = value.to(device)

                transport_covariate_meshes_dir_subj = os.path.join(transport_covariate_meshes_dir, str(ith_scan_to_transp))
                cond_mkdir(transport_covariate_meshes_dir_subj)

                attributes = {key: value.to(device) for key, value in attributes.items()}

                new_covariates = transport_covariates(attributes, infered_start_attributes, start_attributes, diff_names='all')
                latent_inputs = start_latent.expand(arr_samples.shape[0], -1)



                if not save_latvec_only:
                    latent_filename = os.path.join(transport_codes_dir, str(ith_scan_to_transp) + ".pth")

                    start = time.time()
                    with torch.no_grad():
                        dict_metrics = naisr_meshing.create_mesh_3dairway_reconstruction(decoder,
                                                                                         latent_inputs[None, :, ],
                                                                                         new_covariates,
                                                                                         gt,
                                                                                         transport_covariate_meshes_dir_subj,
                                                                                         output_type='model_out', N=256,
                                                                                         device=specs['Device'],
                                                                                         EVALUATE=True)

                        if not os.path.exists(os.path.dirname(latent_filename)):
                            os.makedirs(os.path.dirname(latent_filename))
                        torch.save(latent_inputs.unsqueeze(0), latent_filename)
                        list_metrics.append(dict_metrics)
    # save
    pd.DataFrame.from_records(list_metrics).to_csv(os.path.join(transport_covariate_meshes_dir, 'metrics.csv'))



