'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys

import pandas as pd

import visualizer

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
from visualizer import *


dict_list_cov = {}
dict_list_cov['Airway'] = {'age': np.linspace(0, 240, 20), 'weight': np.linspace(0, 160, 20).tolist(), 'sex': np.linspace(0, 1, 20).tolist()}
dict_list_cov['starman'] = {'cov_1': np.linspace(-1., 1., 20), 'cov_2': np.linspace(-1., 1., 20).tolist()}
dict_list_cov['ADNI'] = {'age': np.linspace(50, 100, 20), 'AD': np.linspace(0, 1, 20).tolist(), 'edu': np.linspace(0, 30, 20).tolist(), 'sex': np.linspace(0, 1, 20).tolist()}


def extract_latent_vector(latent_vectors, idx):
    average_latent_code = torch.mean(latent_vectors, dim=-2)[None, None, :]
    if idx == 'mean':
        return average_latent_code
    elif idx == 'zero':
        return torch.zeros_like(average_latent_code)
    else:
        try:
            return latent_vectors[int(idx)][None, None, :]
        except:
            print("Wrong index of latent vectors. Return average latent vector")
            return average_latent_code



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Testing a DeepSDF autodecoder")

    arg_parser.add_argument(
        "--networksetting",
        "-e",
        dest="networksetting",
        default="examples/hippocampus/naigsr_0920_base.json", #'examples/pediatric_airway/naigsr_0920_nodisentinv.json',  # #"examples/hyperpar_gridsearch/subset.json",#'examples/pediatric_airway/naigsr_0920_nodisentinv.json', #"examples/hippocampus/naigsr_0920_base.json",# 'examples/pediatric_airway/naigsr_0920_nodisentinv.json', #"examples/hippocampus/naigsr_0920_base.json",#"examples/hippocampus/asdf_0919.json",#"examples/hippocampus/naigsr_0920_base.json",#'examples/pediatric_airway/naigsr_0920_nodisentinv.json', #='examples/pediatric_airway/naivf_deepnaigsr.json',
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
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )



    args = arg_parser.parse_args()
    specs = ws.load_experiment_specifications(args.networksetting)

    '''
    read network setting and IO settings
    '''

    backbone = args.backbone
    experiment_name = specs["ExperimentName"]
    print(experiment_name)
    template_attributes = specs["TemplateAttributes"]
    attributes = specs["Attributes"]
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
    latent_vectors = ws.load_latent_vectors(root_path, 'latest', torch.device(device)).to(device)
    average_latent_code = torch.zeros_like(torch.mean(latent_vectors, dim=-2)[None, None, :])


    # loading dataset
    shapetype = specs["Class"]
    filename_dataset = specs["DataSource"]
    num_dim = specs["InFeatures"]

    if shapetype == 'Airway':
        cases = naisr.get_airway_ids(specs["Split"], split='test')
        training_cases = naisr.get_airway_ids(specs["Split"], split='train')
        load_one_case = naisr.get_airway_data_for_id
        df_data = pd.read_csv(filename_dataset)
        load_mean_normals_of_covariates = naisr.airway_dataset.get_mean_normals_of_covariates
    elif shapetype == 'starman':
        cases = naisr.get_starman_ids(filename_dataset, 'test')
        training_cases = naisr.get_starman_ids(filename_dataset, 'train')
        load_one_case = naisr.get_starman_data_for_id
        df_data = pd.read_csv(filename_dataset['test'])
        load_mean_normals_of_covariates = naisr.starman_dataset.get_mean_normals_of_covariates
        #load_mean_normals_of_covariates = naisr.adni_dataset.get_mean_normals_of_covariates
    elif shapetype == 'ADNI':
        cases = naisr.get_adni_ids(specs["Split"], split='test')
        training_cases = naisr.get_adni_ids(specs["Split"], split='train')
        load_one_case = naisr.get_adni_data_for_id
        df_data = pd.read_csv(filename_dataset)
        load_mean_normals_of_covariates = naisr.adni_dataset.get_mean_normals_of_covariates


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

    checkpoint_path = os.path.join(root_path, ws.model_params_subdir, args.checkpoint + '.pth')
    print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location= torch.device(device))["model_state_dict"])
    model.to(specs['Device'])
    model.eval()


    # evaluate testing

    savepath_evo = os.path.join(root_path, 'TemplateShapeEvolution')
    cond_mkdir(savepath_evo)
    savepath_evo_type = os.path.join(savepath_evo, 'average')
    cond_mkdir(savepath_evo_type)



    covariante_names = specs['Attributes'] #[specs['Attributes'][0], specs['Attributes'][1]]
    attributes_mean, attributes_std = load_mean_normals_of_covariates(df_data, training_cases, covariante_names)


    for ith_cov_name in covariante_names: # current covariate name
        ith_shape = 0
        for current_cov_value in dict_list_cov[shapetype][ith_cov_name]: # current covariate value
            attributes = specs["TemplateAttributes"].copy()
            attributes[ith_cov_name] = (current_cov_value - attributes_mean[ith_cov_name]) / attributes_std[ith_cov_name]

            attributes = {key: torch.from_numpy(np.array([value])[None, :]).to(device).float()[[0], ...] for key, value in attributes.items()}

            savedir = os.path.join(savepath_evo_type, ith_cov_name)
            cond_mkdir(savedir)
            dict_savepath = naisr_meshing.create_mesh_reconstruction(shapetype)(model,
                                                                average_latent_code,
                                                                attributes,
                                                                           {},
                                                                savedir,
                                                                output_type='model_out',
                                                                dim=num_dim,
                                                                shapetype=shapetype,
                                                                N=256,
                                                                device=specs['Device'],
                                                                EVALUATE=False,
                                                                MAKE_GT=False,
                                                                MAKE_TEMPLATE=True)
            visualizer.plotter_evolution_traj(shapetype)(dict_savepath, savepath=os.path.join(savedir, str(int(ith_shape)).zfill(6)) + '.png')
            ith_shape += 1
            print(1)



