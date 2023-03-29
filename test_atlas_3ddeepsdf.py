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

if __name__ == "__main__":
        arg_parser = argparse.ArgumentParser(description="Testing a DeepSDF autodecoder")

        arg_parser.add_argument(
            "--networksetting",
            "-n",
            dest="networksetting",
            default='examples/pediatric_airway/naivf_deepsdf.json',
            help="The experiment directory. This directory should include "
                 + "experiment specifications in 'specs.json', and logging will be "
                 + "done in this directory as well.",
        )

        arg_parser.add_argument(
            "--experiment",
            "-e",
            dest="experiment_directory",
            default= "DeepSDF_ATLAS3D_0119_512",
            help="experiment directory name",
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

        arg_parser.add_argument(
            "--prefix",
            "-f",
            dest="prefix",
            default="NAIVF_TEM_VEC_AnalFit_1110",
            help='prefix of experiment name',
        )


        args = arg_parser.parse_args()
        specs = ws.load_experiment_specifications(args.networksetting)

        '''
        read network setting and IO settings
        '''

        backbone = args.backbone
        prefix = args.prefix
        in_features = int(args.dimension)
        experiment_name = args.experiment_directory
        print(experiment_name)
        template_attributes = specs["TemplateAttributes"]
        attributes =  specs["Attributes"]
        split_file = specs["Split"]
        num_samp_per_scene = specs["SamplesPerScene"]
        device = specs['Device']

        '''
        load dataset
        '''
        data_source = specs["DataSource"]

        # load model
        root_path = os.path.join(specs["LoggingRoot"], experiment_name)
        latent_vectors = ws.load_latent_vectors(root_path, 'latest', device)
        # load model
        model = naisr.DeepSDF(
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

        cond_mkdir(root_path)
        checkpoint_path = os.path.join(root_path, 'checkpoints', 'latest.pth')
        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
        model.to(specs['Device'])
        model.eval()



        train_sdf_dataset = naisr.PediatricAirway3DShapeDataset(
            filename_datasource=data_source,
            filename_split=split_file,
            attributes=specs['Attributes'],
            split='all')
        test_sdf_dataset = naisr.PediatricAirway3DShapeDataset(
            filename_datasource=data_source,
            filename_split=split_file,
            attributes=specs['Attributes'],
            split='test')


        '''
        reconstruction
        '''
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

        test_dataloader = data_utils.DataLoader(
            test_sdf_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_data_loader_threads,
            drop_last=False,
        )

        # evaluate testing

        savepath_testingset = os.path.join(root_path, 'test')
        if not os.path.exists(savepath_testingset):
            os.mkdir(savepath_testingset)
        list_metrics = []
        for step, (model_input, template_input, attributes, gt, indices) in enumerate(train_dataloader):
            savepath_testingset_subj = os.path.join(savepath_testingset, str(gt['id'].numpy()[0]))
            if not os.path.exists(savepath_testingset_subj):
                os.mkdir(savepath_testingset_subj)
            indices = indices[:, None].repeat(1, num_samp_per_scene).to(device)
            batch_lat =  latent_vectors[indices]
            dict_metrics = naisr_meshing.create_mesh_3dairway_reconstruction(model, batch_lat, attributes, gt, savepath_testingset_subj, output_type ='model_out', N=256, device=specs['Device'], EVALUATE=True)
            list_metrics.append(dict_metrics)


        pd.DataFrame.from_records(list_metrics).to_csv(os.path.join(savepath_testingset, 'metrics.csv'))
        '''



        '''
        interpolation
        '''


        batch_size = specs["BatchSize"]
        num_data_loader_threads = specs["DataLoaderThreads"]
        train_dataloader = data_utils.DataLoader(
            train_sdf_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=num_data_loader_threads,
            drop_last=True,
        )

        test_dataloader = data_utils.DataLoader(
            test_sdf_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=num_data_loader_threads,
            drop_last=False,
        )

        # evaluate testing

        savepath_interp= os.path.join(root_path, ws.inter_testset_subdir )
        if not os.path.exists(savepath_interp):
            os.mkdir(savepath_interp)
        list_metrics = []
        for step, (model_input, template_input, attributes, gt, indices) in enumerate(train_dataloader):

            subj_name =  str(gt['id'].numpy()[0]) + '_' +  str(gt['id'].numpy()[1])
            savepath_interp_subj = os.path.join(savepath_interp, subj_name)
            if not os.path.exists(savepath_interp_subj):
                os.mkdir(savepath_interp_subj)

            indices_1 = indices[[0], None].repeat(1, num_samp_per_scene).to(device)
            batch_lat_1 =  latent_vectors[indices_1]

            indices_2 = indices[[1], None].repeat(1, num_samp_per_scene).to(device)
            batch_lat_2 =  latent_vectors[indices_2]

            batch_lat = (batch_lat_1 + batch_lat_2) / 2
            dict_metrics = naisr_meshing.create_mesh_3dairway_interpolation(model, batch_lat, attributes, gt, savepath_interp_subj, output_type ='model_out', N=256, device=specs['Device'], EVALUATE=False)
            list_metrics.append(dict_metrics)


        '''
        # evaluation training
        savepath_trainingset = os.path.join(root_path, 'train')
        if not os.path.exists(savepath_trainingset):
            os.mkdir(os.path.join(root_path, 'train'))
        for step, (model_input, attributes, gt) in enumerate(train_dataloader):
            sdf_meshing.create_mesh(model, attributes, savepath_trainingset, dim=in_features, N=256, device=specs['Device'])
        '''
