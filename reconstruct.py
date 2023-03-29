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
import torch.utils.data as data_utils
from naisr import loss_funcs
import naisr_meshing
import pandas as pd
from naisr import *
def reconstruct(
    model,
    optimizer,
    model_input,
    attributes,
    gt,
    batch_vecs,
    latent_size,
        epoch,
    stat,
    lr=5e-4,
    l2reg=False,
):
    num_iterations = 800
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    loss_num = 0
    loss_fn = loss_funcs.loss_pointcloud_sdf


    adjust_learning_rate(lr, optimizer, epoch, decreased_by, adjust_lr_every)
    optimizer.zero_grad()

    model_output = model(model_input, attributes, batch_vecs)
    losses = loss_fn(model_output, gt, batch_vecs, epoch=epoch, dict_losses=specs['Loss'])
    loss = 0
    #for name, il in losses.items():
    loss = losses['sdf'] + losses['normal_constraint'] + losses['inter_constraint'] + losses['eikonal']

    if l2reg:
        loss += 1e-4 * torch.mean(batch_vecs.pow(2))
    loss.backward()
    optimizer.step()
    loss_num = loss.cpu().data.numpy()

    return loss_num, lat_vecs

if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = 1
    arg_parser = argparse.ArgumentParser(description="Testing a DeepSDF autodecoder")

    arg_parser.add_argument(
        "--networksetting",
        "-e",
        dest="networksetting",
        default='examples/pediatric_airway/naivf_deepcondvfsdf.json',
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )

    arg_parser.add_argument(
        "--backbone",
        "-b",
        dest="backbone",
        default='siren',
        help="mlp or siren",
    )

    arg_parser.add_argument(
        "--dim",
        "-d",
        dest="dimension",
        default=3,
        type=int,
        help="shape ellipsoid or torus",
    )


    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    args = arg_parser.parse_args()
    specs = ws.load_experiment_specifications(args.networksetting)

    '''
    read network setting and IO settings
    '''

    backbone = args.backbone
    in_features = int(args.dimension)
    experiment_name = specs["ExperimentName"]
    print(experiment_name)
    template_attributes = specs["TemplateAttributes"]
    attributes = specs["Attributes"]
    split_file = specs["Split"]
    num_samp_per_scene = specs["SamplesPerScene"]
    device = specs['Device']
    latent_size = specs["CodeLength"]
    model_dir = os.path.join(specs['LoggingRoot'], specs['ExperimentName'])

    '''
    load dataset
    '''
    data_source = specs["DataSource"]

    # load model

    root_path = os.path.join(specs["LoggingRoot"], experiment_name)
    cond_mkdir(root_path)
    model = eval(specs['Network'])(
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

    checkpoint_path = os.path.join(root_path, 'checkpoints', 'latest.pth')
    print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    model.to(specs['Device'])
    model.eval()

    test_sdf_dataset = naisr.PediatricAirway3DShapeDataset(
        filename_datasource=data_source,
        filename_split=split_file,
        attributes=specs['Attributes'],
        split='test')

    batch_size = specs["BatchSize"]
    num_data_loader_threads = specs["DataLoaderThreads"]
    test_dataloader = data_utils.DataLoader(
        test_sdf_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # evaluate testing
    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(model_dir, ws.reconstructions_subdir,)
    cond_mkdir(reconstruction_dir)
    reconstruction_meshes_dir = os.path.join(reconstruction_dir, ws.reconstruction_meshes_subdir)
    cond_mkdir(reconstruction_meshes_dir)
    reconstruction_codes_dir = os.path.join(reconstruction_dir, ws.reconstruction_codes_subdir)
    cond_mkdir(reconstruction_codes_dir)


    savepath_interp = os.path.join(root_path, ws.inter_testset_subdir )
    if not os.path.exists(savepath_interp):
        os.mkdir(savepath_interp)

    list_metrics = []

    loss_fn = loss_funcs.loss_pointcloud_sdf

    for step , (model_input, attributes, gt, indices) in enumerate(test_dataloader):

        #subj_name = str(gt['id'].numpy()[0]) + '_' + str(gt['id'].numpy()[1])
        savedir_meshfile = os.path.join(reconstruction_meshes_dir, str(gt['id'][0]))
        cond_mkdir(savedir_meshfile)

        latent_filename = os.path.join(reconstruction_codes_dir,  str(gt['id'][0])  + ".pth")

        logging.info("reconstructing {}".format( str(gt['id'][0]) ))
        # reading data
        model_input = model_input.to(device)
        attributes = {key: value.to(device).float() for key, value in attributes.items()}
        for key, value in gt.items():
            if isinstance(value, torch.Tensor):
                gt[key] = value.to(device)

        #gt['sdf'] = torch.clamp(gt['sdf'], -1., 1.).float()


        start = time.time()

        indices = indices[:, None].repeat(1, num_samp_per_scene).to(device)  # indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1).to(device)
        lat_vecs = torch.nn.Embedding(len(test_sdf_dataset), latent_size, max_norm=1).to(device)
        optimizer = torch.optim.Adam(lat_vecs.parameters(), lr=5e-3)

        #np.array([indices])[:, None].repeat(1, num_samp_per_scene).to(device)
        for e in range(800):
            err, latent = reconstruct(
                model,
                optimizer,
                model_input,
                attributes,
                gt,
                lat_vecs(indices),
                latent_size,
                e,
                0.01,  # [emp_mean,emp_var],
                lr=5e-3,
                l2reg=True,
            )

        model.eval()


        if not save_latvec_only:
            start = time.time()
            with torch.no_grad():
                dict_metrics = naisr_meshing.create_mesh_3dairway_reconstruction(model, lat_vecs(indices), attributes, gt,
                                                                                 savedir_meshfile,
                                                                                 output_type='model_out', N=256,
                                                                                 device=specs['Device'], EVALUATE=True)

                if not os.path.exists(os.path.dirname(latent_filename)):
                    os.makedirs(os.path.dirname(latent_filename))
                torch.save(lat_vecs(indices).unsqueeze(0), latent_filename)
                list_metrics.append(dict_metrics)
    # save
    pd.DataFrame.from_records(list_metrics).to_csv(os.path.join(savedir_meshfile, 'metrics.csv'))




