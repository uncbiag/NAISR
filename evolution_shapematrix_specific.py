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
        default='examples/pediatric_airway/naivf_deepnaigsr.json',
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
    #latent_vectors = ws.load_latent_vectors(root_path, 'epoch_3000', torch.device(device)).to(device)

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
    '''
    train_sdf_dataset = naisr.PediatricAirway3DShapeDataset(
        filename_datasource=data_source,
        filename_split=split_file,
        attributes=specs['Attributes'],
        split='train')
    
    test_idx = train_sdf_dataset.ids[257]
    print('which subj: ' + str(test_idx))

    '''


    checkpoint_path = os.path.join(root_path, 'checkpoints', 'epoch_3000.pth')
    print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location= torch.device(device))["model_state_dict"])
    model.to(specs['Device'])
    model.eval()


    '''
    evolution
    '''
    # evaluate testing

    savepath_evo = os.path.join(root_path, 'ShapeMatrixSpecific')
    cond_mkdir(savepath_evo)
    which_id = '1391'
    #savepath_evo_type = os.path.join(savepath_evo, 'average')
    #cond_mkdir(savepath_evo_type)
    savepath_evo_id = os.path.join(savepath_evo, str(which_id))
    cond_mkdir(savepath_evo_id)
    #savepath_evo_type = os.path.join(savepath_evo_type, 'average')
    #cond_mkdir(savepath_evo_type)

    dict_of_evolution = {}
    dict_text = {}
    dict_color = {}
    figure_name = 'shapematrix'

    '''
    load a sample
    '''
    training_cases = naisr.get_ids(specs["Split"], split='train')
    import pandas as pd

    df_data = pd.read_csv(specs["DataSource"], header=0)

    arr_samples_specific, attributes_specific, gt_specific = naisr.get_data_for_id(which_id,
                                                                                   df_data,
                                                                                   training_cases,
                                                                               specs["Attributes"])

    attributes_specific = {key: value.to(device) for key, value in attributes_specific.items()}
    #average_latent_code = latent_vectors[3][None, None, :]  # [indices]

    codes_dir = os.path.join(root_path, ws.transport_general_subdir, ws.reconstruction_codes_subdir)

    average_latent_code  = load_transport_vectors(codes_dir,which_id, device)
    average_latent_code.requires_grad = False

    #savepath_specifc = savepath_evo_type #os.path.join(savepath_evo,which_id,)
    #cond_mkdir(savepath_specifc)
    '''
    current_color = naisr_meshing.revert_points_to_template(model,
                                                            average_latent_code,
                                                            attributes_specific,
                                                            arr_samples_specific[0],
                                                            device)
                                            '''
    visualize_a_case(savepath_evo_id, gt_specific['pvgt_path'][0],colors=None, colormap='rgb')
    print(attributes_specific)

    for age in np.linspace(-1.0, 1.0, 7):
        dict_of_evolution[age] = {}
        dict_text[age]  = {}
        dict_color[age]  = {}

        for weight0 in np.linspace(-1.0, 1.0, 7):
            weight = -weight0
            logging.info("evolving {}{}".format(age, weight))


            #average_latent_code = torch.mean(latent_vectors[torch.arange(5)+1], dim=0)[None, None, :] #[indices]

            attributes = {'weight': np.array([weight])[None, :], 'age': np.array([age])[None, :], 'sex': np.array([1])[None, :]}
            attributes = {key: torch.from_numpy(value).to(device).float()[[0], ...] for key, value in attributes.items()}

            savedir = os.path.join(savepath_evo_id, 'age_' + str(age) + '_weight_' + str(weight))
            cond_mkdir(savedir)
            savepath = naisr_meshing.create_mesh_3dairway_reconstruction(model,
                                                                    average_latent_code,
                                                                    attributes,
                                                                    {},
                                                                    savedir,
                                                                    output_type='model_out',
                                                                    N=128,
                                                                    device=specs['Device'],
                                                                    EVALUATE=False)
            '''
            current_color = naisr_meshing.revert_points_to_template(model,
                                                                    average_latent_code,
                                                                    attributes,
                                                                    savedir,
                                                                    device)

            '''
            dict_of_evolution[age][weight] = os.path.join(savedir, 'surface.stl')
            #dict_color[age][weight] = current_color
            dict_text[age][weight] = attributes

            plotter_evolution_shapematrix(dict_of_evolution,
                                      savepath_evo_id,
                                      dict_text0=dict_text,dict_colors0=None)#,
                                              #dict_colors0={}) #dict_color)



