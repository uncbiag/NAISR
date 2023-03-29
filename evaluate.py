#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os

import pandas as pd
import trimesh
import csv

import naisr
import naisr.workspace as ws
import glob
import re
import torch.utils.data as data_utils
import torch
import time
import skimage.measure as measure


def compute_chamfer_distance(chamfer_dist_file):
    chamfer_distance = []
    with open(chamfer_dist_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for idx, row in enumerate(spamreader):
            if idx > 0:
                chamfer_distance.append(float(row[-1]))
    print("avg chamfer distance: ", np.mean(np.array(chamfer_distance)))


def get_gt_samples(shapetype, attributes, dim=3, N=1024, ):


    start = time.time()

    max_batch = 64 ** dim

    '''
    if 'crosssectionradius' in attributes.keys():
        R = (attributes["crosssectionradius"] + attributes["ringradius"]) / 2
        r = ( attributes["ringradius"] - attributes["crosssectionradius"]) / 2
        attributes["crosssectionradius"] = r
        attributes["ringradius"] = R
    '''
    r = attributes["crosssectionradius"]
    R = attributes["ringradius"]
    if dim == 3:
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N#overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]

        num_samples = N ** 3
        # get sdf
        sdf_values = torch.tanh(
            torch.square(torch.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2) - R) + samples[:, 2] ** 2 - r ** 2).cpu().numpy()


        sdf_values = sdf_values.reshape(N, N, N)



        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf_values, level=0., spacing=[voxel_size] * 3)

        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]


        return mesh_points

    elif dim ==2:
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, ]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 2, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 2, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 1] = overall_index % N
        samples[:, 0] = (overall_index.long() / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[0]
        samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]

        num_samples = N ** 2

        sdf_values = torch.tanh(
            torch.square(torch.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2) - R) + samples[:, 2] ** 2 - r ** 2).cpu().numpy()
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
        contours = measure.find_contours(sdf_values, 0.)

        mesh_points = []
        for contour in contours:
            mesh_points += contour
        mesh_points(np.array(mesh_points) / (N - 1)  - 0.5) * 2
        return mesh_points


def evaluate(args, specs):
    '''
    read network setting and IO settings
    '''
    print(args)
    pos_enc = args.posenc
    backbone = args.backbone
    shapetype = args.shape
    prefix = args.prefix
    in_features = int(args.dimension)
    experiment_name = args.experiment_directory
    print(experiment_name)

    # get class and path
    #shapetype = specs["Class"]
    data_source = specs['DataSource']
    root_path = os.path.join(specs["LoggingRoot"], experiment_name)


    if in_features == 3:
        # load dataset
        if shapetype == 'ellipsoid':
            train_sdf_dataset = naisr.ToyEllipsoidDataset(
                data_source,
                attributes=specs['Attributes'],
                shapetype=shapetype,
                split='print_train',
                articulation=False,
                num_atc_parts=1,
            )
            test_sdf_dataset = naisr.ToyEllipsoidDataset(
                data_source,
                attributes=specs['Attributes'],
                shapetype=shapetype,
                split='print_test',
                articulation=False,
                num_atc_parts=1,
            )


        elif shapetype == 'torus':
            train_sdf_dataset = naisr.ToyTorusDataset(
                data_source,
                attributes=specs['Attributes'],
                shapetype=shapetype,
                split='print_train',
                articulation=False,
                num_atc_parts=1,
            )
            test_sdf_dataset = naisr.ToyTorusDataset(
                data_source,
                attributes=specs['Attributes'],
                shapetype=shapetype,
                split='print_test',
                articulation=False,
                num_atc_parts=1,
            )
    elif in_features == 2:
        # load dataset
        if shapetype == 'ellipsoid':
            train_sdf_dataset = naisr.Toy2DEllipsoidDataset(
                data_source,
                attributes=specs['Attributes'],
                shapetype=shapetype,
                split='print_train',
                articulation=False,
                num_atc_parts=1,
            )
            test_sdf_dataset = naisr.Toy2DEllipsoidDataset(
                data_source,
                attributes=specs['Attributes'],
                shapetype=shapetype,
                split='print_test',
                articulation=False,
                num_atc_parts=1,
            )


        elif shapetype == 'torus':
            train_sdf_dataset = naisr.Toy2DTorusDataset(
                data_source,
                attributes=specs['Attributes'],
                shapetype=shapetype,
                split='print_train',
                articulation=False,
                num_atc_parts=1,
            )
            test_sdf_dataset = naisr.Toy2DTorusDataset(
                data_source,
                attributes=specs['Attributes'],
                shapetype=shapetype,
                split='print_test',
                articulation=False,
                num_atc_parts=1,
            )

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
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )


    chamfer_results = []
    chamfer_statistics = []

    chamfer_thinner = []
    chamfer_fatter = []
    failed_thinner = 0
    failed_fatter = 0




    dict_all_names =  {'test': test_dataloader, 'train': train_dataloader, }

    interpolation_or_extrapolation = {'train': 'Interpolation',
                                      'test': 'Extrapolation'}




    for name, current_dataloader in dict_all_names.items():
        current_group_chamfer_dist = []
        current_failed_to_model = 0
        for step, (model_input, attributes, gt) in enumerate(dict_all_names[name]):
            if 'ringradius' in attributes.keys():
                R = (attributes['ringradius'] + attributes['crosssectionradius']) / 2
                r = (attributes['ringradius'] - attributes['crosssectionradius']) / 2
                attributes['ringradius'] = R
                attributes['crosssectionradius'] = r

            filename = ''
            for ith_attri in attributes.keys():
                filename = filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri].cpu().numpy()[0], 2))

            logging.debug( "evaluating " + os.path.join(root_path, name))
            reconstructed_mesh_filename = os.path.join(root_path, name, filename + '.stl')

            logging.debug('reconstructed mesh is "' + reconstructed_mesh_filename + '"')

            if os.path.isfile(reconstructed_mesh_filename):
                reconstruction = trimesh.load(reconstructed_mesh_filename)
                ground_truth_points = get_gt_samples(shapetype, attributes, dim=3, N=1024, )
                chamfer_dist = naisr.compute_depth_chamfer(ground_truth_points, reconstruction, num_mesh_samples=30000)
            else:

                current_failed_to_model += 1
                if attributes['crosssectionradius'] < 0.1499:
                    failed_thinner += 1
                elif attributes['crosssectionradius'] >= 0.3501:
                    failed_fatter += 1
                continue

            print('chamfer distance of' + str(attributes) + ': ' +  str(chamfer_dist))
            #naisr.metrics.chamfer.compute_trimesh_chamfer(
            #    ground_truth_points,
            #    reconstruction,
            #    1,
            #    1,
            #)

            logging.debug("chamfer distance: " + str(chamfer_dist))

            if attributes['crosssectionradius'] < 0.1499:
                size = 'thinner'
                chamfer_thinner.append(chamfer_dist)
            elif attributes['crosssectionradius'] >= 0.3501:
                size = 'fatter'
                chamfer_fatter.append(chamfer_dist)
            else:
                size = 'regular'

            current_eval = {
                'Task': interpolation_or_extrapolation[name],
                'Chamfer Dist': chamfer_dist,
                'Backbone': backbone,
                'PosEnc': pos_enc,
                'size': size
            }

            current_eval.update(attributes)
            chamfer_results.append(current_eval)
            current_group_chamfer_dist.append(chamfer_dist)

        chamfer_statistics.append({'Task': interpolation_or_extrapolation[name],
                                   'Metrics': 'Chamfer Distance',
                                   'Mean': str(round(np.array(current_group_chamfer_dist).mean(), 3)),
                                   'Std': str(round(np.array(current_group_chamfer_dist).std(), 3))})

        chamfer_statistics.append({'Task': interpolation_or_extrapolation[name],
                                   'Metrics': 'Failed Cases',
                                   'Value': current_failed_to_model})


    chamfer_statistics.append({'Task': 'Extrapolation->Thinner',
                                'Metrics': 'Chamfer Distance',
                                'Mean': str(round(np.array(chamfer_thinner).mean(), 3)),
                                'Std': str(round(np.array(chamfer_thinner).std(), 3))})
    chamfer_statistics.append({'Task': 'Extrapolation->Fatter',
                                'Metrics': 'Chamfer Distance',
                                'Mean': str(round(np.array(chamfer_fatter).mean(), 3)),
                                'Std': str(round(np.array(chamfer_fatter).std(), 3))})

    chamfer_statistics.append({'Task': 'Extrapolation->Fatter',
                                'Metrics': 'Failed Cases',
                                'Value': failed_fatter})
    chamfer_statistics.append({'Task': 'Extrapolation->Thinner',
                                'Metrics': 'Failed Cases',
                                'Value': failed_thinner})



    chamfer_dist_file = os.path.join(root_path, "chamfer.csv")
    chamfer_statistics_file = os.path.join(root_path, 'chamfer_stat.csv')

    pd.DataFrame.from_records(chamfer_results).to_csv(chamfer_dist_file)
    pd.DataFrame.from_records(chamfer_statistics).to_csv(chamfer_statistics_file)



    return chamfer_results, chamfer_statistics


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")


    arg_parser.add_argument(
        "--networksetting",
        "-n",
        dest="networksetting",
        #default='examples/toy/torus/naivf_tem_vec.json',
        required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )

    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        #default="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge",
        required=True,
        help="experiment directory name",
    )

    arg_parser.add_argument(
        "--posenc",
        "-p",
        dest="posenc",
        help="position encoding",
        action='store_true'
    )

    arg_parser.add_argument(
        "--backbone",
        "-b",
        dest="backbone",
        #default='mlp',
        required=True,
        help="mlp or siren",
    )

    arg_parser.add_argument(
        "--shape",
        "-s",
        dest="shape",
        #default='torus',
        required=True,
        help="shape ellipsoid or torus",
    )

    arg_parser.add_argument(
        "--dim",
        "-d",
        dest="dimension",
        #default=3,
        required=True,
        type=int,
        help="shape ellipsoid or torus",
    )

    arg_parser.add_argument(
        "--prefix",
        "-f",
        dest="prefix",
        #default="NAIVF_TEM_VEC_AnalFit_1110",
        required=True,
        help='prefix of experiment name',
    )


    #naisr.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    #naisr.configure_logging(args)
    pos_enc = args.posenc
    backbone = args.backbone
    shapetype = args.shape
    prefix = args.prefix
    in_features = int(args.dimension)
    experiment_name = args.experiment_directory

    specs_filename = args.networksetting
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    specs = json.load(open(specs_filename))
    print(args.experiment_directory)
    chamfer_dist, chamfer_statistics = evaluate(args, specs,)
    print(chamfer_statistics)
