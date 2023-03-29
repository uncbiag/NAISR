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



if __name__ == "__main__":
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
    latent_vectors = ws.load_latent_vectors(root_path, 'epoch_3000', torch.device(device)).to(device)
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
    model.load_state_dict(torch.load(checkpoint_path, map_location= torch.device(device))["model_state_dict"])
    model.to(specs['Device'])
    model.eval()

    train_sdf_dataset = naisr.PediatricAirway3DShapeDataset(
        filename_datasource=data_source,
        filename_split=split_file,
        attributes=specs['Attributes'],
        split='train')


    '''
    evolution
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

    savepath_evo = os.path.join(root_path, 'evolution')
    cond_mkdir(savepath_evo)


    list_metrics = []

    '''
    load a sample
    '''
    training_cases = naisr.get_ids(specs["Split"], split='test')
    import pandas as pd

    df_data = pd.read_csv(specs["DataSource"], header=0)

    arr_samples_specific, attributes_specific, gt_specific = naisr.get_data_for_id('1193',
                                                                                   df_data,
                                                                                   training_cases,
                                                                                   specs["Attributes"])

    attributes_specific = {key: value.to(device) for key, value in attributes_specific.items()}
    #average_latent_code = latent_vectors[3][None, None, :]  # [indices]
    codes_dir = os.path.join(root_path, ws.reconstructions_subdir, ws.reconstruction_codes_subdir)

    average_latent_code  = load_transport_vectors(codes_dir, '1193', device)
    savepath_specifc = os.path.join(savepath_evo, '1193')
    cond_mkdir(savepath_specifc)


    #for step, (model_input, attributes, gt, indices) in enumerate(train_dataloader):
        logging.info("evolving {}".format( str(gt['id'][0]) ))
        # reading data
        model_input = model_input.to(device)
        attributes = {key: torch.zeros_like(value).to(device).float()[[0], ...] for key, value in attributes.items()}
        for key, value in gt.items():
            if isinstance(value, torch.Tensor):
                gt[key] = value.to(device)

        gt['sdf'] = torch.clamp(gt['sdf'], -1., 1.).float()

        subj_name = str(gt['id'][0])
        savepath_interp_subj = os.path.join(savepath_evo, subj_name)
        cond_mkdir(savepath_interp_subj)

        indices = indices[[0], None].repeat(1, num_samp_per_scene).to(device)
        batch_lat = latent_vectors[indices]

        '''
        list_of_evolution = []
        for weight in [attributes['weight']-0.2]:
        #for age in np.linspace(-1, 1, 5):
            attributes = {key: value.to(device).float()[[0], ...] for key, value in
                          attributes.items()}
            attributes['weight'] = torch.zeros_like(attributes['weight'] )# + weight
            #attributes['weight'] = torch.zeros_like(attributes['weight']) + age

            savedir_age_weight = os.path.join(savepath_interp_subj, 'template_' + str(weight))
            list_of_evolution.append(savedir_age_weight)
            cond_mkdir(savedir_age_weight)
            dict_metrics = naisr_meshing.create_mesh_3dairway_template(model,
                                                                            batch_lat,
                                                                            attributes,
                                                                            gt,
                                                                        savedir_age_weight,
                                                                        which_attribute='weight',
                                                                        output_type='model_out',
                                                                        N=256, device=specs['Device'], EVALUATE=False)
        '''

        '''
        list_of_evolution_sex = []
        #for sex in [attributes['sex']-0.2, attributes['sex']-0.1, attributes['sex'], attributes['sex']+0.1, attributes['sex']+0.2]:
        for sex in np.linspace(-1, 1, 5):
            attributes = {key: value.to(device).float()[[0], ...] for key, value in
                          attributes.items()}
            attributes['sex'] = torch.zeros_like(attributes['sex'] ) + sex
            #attributes['weight'] = torch.zeros_like(attributes['weight']) + age

            savedir_sex= os.path.join(savepath_interp_subj, 'sex_' + str(sex))
            list_of_evolution_sex.append(savedir_sex)
            cond_mkdir(savedir_sex)
            dict_metrics = naisr_meshing.create_mesh_3dairway_evolution(model,
                                                                            batch_lat,
                                                                            attributes,
                                                                            gt,
                                                                        savedir_sex,
                                                                        which_attribute='sex',
                                                                        output_type='model_out',

                                                                        N=256, device=specs['Device'], EVALUATE=False)

        '''

        list_of_evolution_age = []
        list_text_age = []
        list_color_age = []
        #for age in [attributes['age']-0.2, attributes['age']-0.1, attributes['age'], attributes['age']+0.1, attributes['age']+0.2]:
        for age in np.linspace(-1, 1, 5):
            attributes = {key: value.to(device).float()[[0], ...] for key, value in
                          attributes.items()}
            attributes['age'] = torch.zeros_like(attributes['age'] ) + age
            #attributes['weight'] = torch.zeros_like(attributes['weight']) + age

            savedir_age = os.path.join(savepath_interp_subj, 'age_' + str(age))

            cond_mkdir(savedir_age)
            savepath = naisr_meshing.create_mesh_3dairway_evolution(model,
                                                                        batch_lat,
                                                                        attributes,
                                                                        gt,
                                                                        savedir_age,
                                                                        which_attribute='age',
                                                                        output_type='model_out',
                                                                        N=256, device=specs['Device'], EVALUATE=False)
            #current_color = naisr_meshing.revert_points_to_template(model, batch_lat, attributes, savedir_age, device)
            list_of_evolution_age.append(savepath)
            #list_color_age.append(current_color)
            list_text_age.append('')

        plotter_evolution(list_of_evolution_age, os.path.join(savepath_interp_subj, 'age',), list_text=None, list_colors=None) #list_colors=list_color_age)





        list_of_evolution_weight = []
        list_text_weight = []
        list_color_weight = []
        #for weight in [attributes['weight']-0.2, attributes['weight']-0.1, attributes['weight'], attributes['weight']+0.1, attributes['weight']+0.2]:
        for weight in np.linspace(-1, 1, 5):
            attributes = {key: torch.zeros_like(value).to(device).float()[[0], ...] for key, value in
                          attributes.items()}
            attributes['weight'] = torch.zeros_like(attributes['weight'] ) + weight
            #attributes['weight'] = torch.zeros_like(attributes['weight']) + age

            savedir_weight = os.path.join(savepath_interp_subj, 'weight_' + str(weight))

            cond_mkdir(savedir_weight)
            #savepath = os.path.join(savedir_weight, 'weight')

            savepath = naisr_meshing.create_mesh_3dairway_evolution(model,
                                                                    batch_lat,
                                                                    attributes,
                                                                    gt,
                                                                    savedir_weight,
                                                                    which_attribute='weight',
                                                                    output_type='model_out',
                                                                    N=256, device=specs['Device'], EVALUATE=False)
            #current_color = naisr_meshing.revert_points_to_template(model, batch_lat, attributes, savedir_weight, device)
            list_of_evolution_weight.append(savepath)
            #list_color_weight.append(current_color)
            list_text_weight.append('')
        plotter_evolution(list_of_evolution_weight,  os.path.join(savepath_interp_subj, 'weight'), list_text=None, list_colors=None) #, list_colors=list_color_weight)

