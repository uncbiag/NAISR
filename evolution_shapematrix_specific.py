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
from visualizer import *
import numpy as np


dict_list_cov = {}
# dict_list_cov['Airway'] = {'age': np.linspace(-2.0, 2.0, 7), 'weight': np.linspace(-2., 4., 7).tolist().reverse()}
# dict_list_cov['starman'] = {'cov_1': np.linspace(-1., 1., 7), 'cov_2': np.linspace(-1., 1., 7).tolist().reverse()}
# dict_list_cov['ADNI'] = {'age': np.linspace(-3.0, 3.0, 7), 'AD': np.linspace(-1., 2., 7).tolist().reverse()}

dict_list_cov['Airway'] = {'age': np.linspace(-2.0, 2.0, 7), 'weight': np.linspace(4., -2., 7).tolist()}
dict_list_cov['starman'] = {'cov_1': np.linspace(-1., 1., 7), 'cov_2': np.linspace(1., -1., 7).tolist()}
dict_list_cov['ADNI'] = {'age': np.linspace(-3.0, 3.0, 7), 'AD': np.linspace(2., -1., 7).tolist()}


dict_select_id = {'Airway': '1364',
                  'ADNI': "ADNI_005_S_0610_MR_Hippocampal_Mask_Hi_20080228121411509_S15727_I93444.nii",
                  'starman': "0007_0"
                  }

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Testing a DeepSDF autodecoder")

    arg_parser.add_argument(
        "--networksetting",
        "-e",
        dest="networksetting",
        default= 'examples/pediatric_airway/naigsr_0920_nodisentinv.json', # ,#'examples/pediatric_airway/naigsr_0920_nodisentinv.json',# #"examples/starman/naigsr_0222_base.json",#'examples/pediatric_airway/naigsr_0920_inv_no_disent.json',
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device))["model_state_dict"])
    model.to(specs['Device'])
    model.eval()


    # loading dataset
    shapetype = specs["Class"]
    filename_dataset = specs["DataSource"]
    num_dim = specs["InFeatures"]

    '''
    evolution
    '''
    # evaluate testing
    savepath_evo = os.path.join(root_path, 'ShapeMatrixSpecific')
    cond_mkdir(savepath_evo)
    which_id = dict_select_id[shapetype]#'1364'#"ADNI_005_S_0610_MR_Hippocampal_Mask_Hi_20080228121411509_S15727_I93444.nii"#"ADNI_022_S_0129_MR_Hippocampal_Mask_Hi_20080228130211054_S27668_I93640.nii" #'ADNI_031_S_0294_MR_Hippocampal_Mask_Hi_20080228140917501_S12243_I93780.nii' #'0002_0'

    savepath_evo_id = os.path.join(savepath_evo, str(which_id))
    cond_mkdir(savepath_evo_id)

    dict_of_evolution = {}
    dict_text = {}
    dict_color = {}
    figure_name = 'shapematrix'

    '''
    load a sample
    '''




    if shapetype == 'Airway':
        cases = naisr.get_airway_ids(specs["Split"], split='test')
        training_cases = naisr.get_airway_ids(specs["Split"], split='train')
        load_one_case = naisr.get_airway_data_for_id
        df_data = pd.read_csv(filename_dataset)
        #list_patient_scans = naisr.get_airways_for_transport(specs["DataSource"], specs["Split"], split='test_multiple')
    elif shapetype == 'starman':
        cases = naisr.get_starman_ids(filename_dataset, 'test')
        training_cases = naisr.get_starman_ids(filename_dataset, 'train')
        load_one_case = naisr.get_starman_data_for_id
        df_data = pd.read_csv(filename_dataset['test'])
        #list_patient_scans = naisr.get_starmans_for_transport(specs["DataSource"], split='test')
    elif shapetype == 'ADNI':
        #cases = naisr.get_adni_ids(specs["Split"], split='test')
        training_cases = naisr.get_adni_ids(specs["Split"], split='train')
        load_one_case = naisr.get_adni_data_for_id
        df_data = pd.read_csv(filename_dataset)


    arr_samples, attributes_observed, gt = load_one_case(which_id,
                                                df_data,
                                                training_cases,
                                                specs["Attributes"],
                                                stage='test')


    attributes_observed = {key: float(value.numpy()) for key, value in attributes_observed.items()}
    # average_latent_code = latent_vectors[3][None, None, :]  # [indices]

    codes_dir = os.path.join(root_path, ws.reconstructions_subdir, ws.reconstruction_codes_subdir)
    average_latent_code = load_transport_vectors(codes_dir, which_id, device)
    average_latent_code.requires_grad = False


    #visualize_a_case(savepath_evo_id, gt['pvgt_path'][0], colors=None, colormap='rgb')
    #print(attributes)

    dict_of_evolution = {}
    dict_text = {}
    dict_color = {}
    #figure_name = 'shapematrix2'
    covariante_names = specs['Attributes']
    for cov_1 in dict_list_cov[shapetype][covariante_names[0]]:
        dict_of_evolution[cov_1] = {}
        dict_text[cov_1] = {}
        dict_color[cov_1] = {}

        for cov_2 in dict_list_cov[shapetype][covariante_names[1]]:
            #cov_2 = -cov_2
            logging.info("evolving {}{}".format(cov_1, cov_2))

            # average_latent_code = torch.mean(latent_vectors[torch.arange(5)+1], dim=0)[None, None, :] #[indices]
            attributes = attributes_observed.copy()
            attributes[covariante_names[0]] = cov_1  # np.array([cov_1])[None, :]
            attributes[covariante_names[1]] = cov_2  # np.array([cov_2])[None, :]
            attributes = {key: torch.from_numpy(np.array([value])[None, :]).to(device).float()[[0], ...] for key, value in attributes.items()}



            savedir = os.path.join(savepath_evo_id, 'cov1_' + str(cov_1) + '_cov2_' + str(cov_2))
            cond_mkdir(savedir)
            savepath = naisr_meshing.create_mesh_reconstruction(shapetype)(model,
                                                                average_latent_code,
                                                                attributes,
                                                                gt,
                                                                savedir,
                                                                output_type='model_out',
                                                                dim=num_dim,
                                                                shapetype=shapetype,
                                                                N=256,
                                                                device=specs['Device'],
                                                                EVALUATE=False,
                                                                MAKE_GT=True,
                                                                MAKE_TEMPLATE=False)
            '''
            current_color = naisr_meshing.revert_points_to_template(model,
                                                                    average_latent_code,
                                                                    attributes,
                                                                    savedir,
                                                                    device)

            '''
            if shapetype == 'starman':
                dict_of_evolution[cov_1][cov_2] = savepath #os.path.join(savedir, 'surface.vtk')

            else:
                dict_of_evolution[cov_1][cov_2] = savepath #os.path.join(savedir, 'surface.stl')
            #dict_color[age][weight] = current_color
            dict_text[cov_1][cov_2] = attributes


            plotter_evolution_shapematrix(shapetype)(
                                      dict_of_evolution,
                                      savepath_evo_id,
                                      dict_text0=dict_text,
                                      dict_colors0=None)
                                      #dict_colors0=dict_color)




