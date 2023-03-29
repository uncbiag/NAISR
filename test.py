'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import naisr.modules

import naisr_meshing
import naisr
import naisr.workspace as ws
import argparse
import torch.utils.data as data_utils
from utils import cond_mkdir
from naisr import *

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Testing a DeepSDF autodecoder")

    arg_parser.add_argument(
        "--networksetting",
        "-e",
        dest="networksetting",
        default='examples/pediatric_airway/naivf_deepnaisr.json',
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
    attributes =  specs["Attributes"]
    split_file = specs["Split"]
    num_samp_per_scene = specs["SamplesPerScene"]
    device = specs['Device']
    latent_size = specs["CodeLength"]
    root_path = os.path.join(specs['LoggingRoot'], specs['ExperimentName'])
    cond_mkdir(root_path)
    '''
    load dataset
    '''
    data_source = specs["DataSource"]

    # load model
    latent_vectors = ws.load_latent_vectors(root_path, 'epoch_3000', device)
    # load model
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

    checkpoint_path = os.path.join(root_path, 'checkpoints', 'epoch_3000.pth')
    print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device))["model_state_dict"])
    model.to(specs['Device'])
    model.eval()

    train_sdf_dataset = naisr.PediatricAirway3DShapeDataset(
        filename_datasource=data_source,
        filename_split=split_file,
        attributes=specs['Attributes'],
        split='train')


    '''
    reconstruction
    '''

    batch_size = specs["BatchSize"]
    num_data_loader_threads = specs["DataLoaderThreads"]
    train_dataloader = data_utils.DataLoader(
        train_sdf_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )


    # evaluate testing

    savepath_testingset = os.path.join(root_path, 'representation')
    cond_mkdir(savepath_testingset)
    list_metrics = []
    for step, (model_input, attributes, gt, indices) in enumerate(train_dataloader):
        savepath_testingset_subj = os.path.join(savepath_testingset, str(gt['id'][0]))
        if not os.path.exists(savepath_testingset_subj):
            os.mkdir(savepath_testingset_subj)
        indices = indices[:, None].repeat(1, num_samp_per_scene).to(device)
        batch_lat =  latent_vectors[indices].to(device)
        attributes = {key: value.to(device) for key, value in attributes.items()}
        for key, value in gt.items():
            if not isinstance(value, list):
                gt[key] = value.to(device)


        dict_metrics = naisr_meshing.create_mesh_3dairway_reconstruction(model, batch_lat, attributes, gt, savepath_testingset_subj, output_type ='model_out', N=256, device=specs['Device'], EVALUATE=True)
        list_metrics.append(dict_metrics)


    pd.DataFrame.from_records(list_metrics).to_csv(os.path.join(savepath_testingset, 'metrics.csv'))
