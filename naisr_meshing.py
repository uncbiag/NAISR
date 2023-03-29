'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
# !/usr/bin/env python3

import logging

import numpy
import numpy as np
import plyfile
import skimage.measure as measure
import time
import torch
import os
import pyvista as pv
import dataio
import matplotlib.pyplot as plt
import point_cloud_utils as pcu
from visualizer import *

from colorcet.plotting import swatch, swatches

def create_curves(model, attributes, savepath, gt_csa, gt_pos, id, mean, std, device):
    slice_coords_1d = torch.linspace(0, 1, 200)[:, None]  # dataio.get_mgrid(128)
    model.eval()
    '''
    1. x
    '''
    # yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    x_slice_model_input = slice_coords_1d.to(device)

    # yz_slice_model_input = slice_coords_2d.to(device)
    fixed_attributes = {}
    for ith_attri in attributes.keys():
        fixed_attributes[ith_attri] = torch.tensor(attributes[ith_attri])  # [None, :]
    for ith_attri in fixed_attributes.keys():
        fixed_attributes[ith_attri] = fixed_attributes[ith_attri].repeat(x_slice_model_input.shape[0], ).float().to(
            device)

    x_model_out = model(x_slice_model_input, fixed_attributes)
    pred_sdf_values = x_model_out['model_out'].detach() * std + mean
    fig = make_curve(slice_coords_1d,
                     gt_pos,
                     pred_sdf_values,
                     gt_csa,
                     attributes,
                     plotname=os.path.join(savepath, str(id) + '.png')
                     )


def make_curve(slice_coords_1d,
               gt_pos,
               pred_sdf_values,
               gt_csa,
               fixed_attributes,
               plotname,
               model='lin'):
    fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
    ax.plot(slice_coords_1d, pred_sdf_values.cpu().numpy().squeeze(), c='red')
    ax.plot(gt_pos, gt_csa, c='blue')
    title = ''
    # for i_key in plotname.keys():
    #    title += i_key + ': ' + str(plotname[i_key][0].detach().cpu().numpy()) + ', \n'
    ax.set_title(str(fixed_attributes), fontsize=5)
    ax.set_axis_on()
    plt.savefig(plotname)
    plt.close()
    return fig


def create_mesh(
        decoder, attributes, filename, dim=3, N=1024, device='cpu', max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    max_batch = 64 ** dim
    if dim == 3:
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:,
        2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
        samples[:, 1] = (
                                    overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
        samples[:, 0] = ((
                                     overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]

        num_samples = N ** 3

    elif dim == 2:
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

    # print(samples.max())
    # print(samples.min())
    samples.requires_grad = False

    head = 0

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)
        attributes = {key: value.to(device) for key, value in attributes.items()}
        samples[head: min(head + max_batch, num_samples), -1] = (
            decoder(sample_subset[None, :, range(dim)], attributes)['model_out']
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))

    filename = ''

    if 'crosssectionradius' in attributes.keys():
        R = (attributes["crosssectionradius"] + attributes["ringradius"]) / 2
        r = (attributes["ringradius"] - attributes["crosssectionradius"]) / 2
        attributes["crosssectionradius"] = r
        attributes["ringradius"] = R

    for ith_attri in attributes.keys():
        filename = filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri].cpu().numpy()[0], 2))

    if dim == 3:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            os.path.join(ply_filename, filename + ".ply"),
            offset,
            scale,
        )
    elif dim == 2:
        convert_sdf_samples_in_2D(sdf_values.data.cpu()[0], os.path.join(ply_filename, filename + ".png"))


def create_mesh_airway(
        decoder,
        attributes,
        gt,
        savedir,
        dim=3,
        N=1024,
        device='cpu',
        max_batch=64 ** 3,
        offset=None,
        scale=None,
        EVALUATE=True,
):
    start = time.time()
    dict_metrics = {}

    decoder.eval()

    max_batch = 32 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 2
    voxel_size = 2.0 / (N - 1) * 2

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:,
    2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (
                                overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = ((
                                 overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    # print(samples.max())
    # print(samples.min())
    samples.requires_grad = False

    head = 0

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)
        attributes = {key: value.to(device) for key, value in attributes.items()}
        samples[head: min(head + max_batch, num_samples), -1] = (
            decoder(sample_subset[None, :, range(dim)],
                    attributes,
                    sample_subset[None, :, range(dim)])['model_out']
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))

    filename = str(
        gt['id'].numpy()[0])  # filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri].cpu().numpy()[0], 2))
    # path_aligned_surface = os.path.join(savedir, filename )
    if dim == 3:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            savedir,
            offset,
            scale,
        )

        path_centerline_normal = gt['ctl_path'][0]
        dict_metrics['path_pred'], list_pred_csa2d = reconstruct_3D_airway(path_centerline_normal, savedir)

        if EVALUATE:
            # savepath_viz_comparison = path_aligned_surface + '_comp'
            dict_metrics['hausdorff'], dict_metrics['chamfer'] = evaluate_3D_airway_reconstruction(
                path_centerline_normal,
                dict_metrics['path_pred'],
                savedir,
            )
            if not np.isnan(dict_metrics['hausdorff']):
                plotter_2d_csa(gt['csa2d'], list_pred_csa2d, savedir)

        dict_metrics['path_gt'] = gt['csa2d'][0][0]
        dict_metrics['path_ctl'] = path_centerline_normal
    print(dict_metrics)
    return dict_metrics




def create_mesh_3dairway_reconstruction(
        model,
        lat_vec,
        attr,
        gt,
        savedir,
        output_type='model_out',
        dim=3,
        N=1024,
        device='cpu',
        EVALUATE=True,
):
    dict_metrics = {}

    head = 0
    num_samples = N ** 3
    max_batch = 16 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 3
    voxel_size = 2.0 / (N - 1) * 3

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:,
    2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (
                                overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = ((
                                 overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]
    # samples *=2

    start = time.time()

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)
        if type(lat_vec) == dict:
            lat_vec_subset = {}
            for name, value in lat_vec.items():
                lat_vec_subset[name] = lat_vec[name][0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        else:
            lat_vec_subset = lat_vec[0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        samples[head: min(head + max_batch, num_samples), -1] = (
            model(sample_subset[None, :, range(dim)], attr, lat_vec_subset, training=False)[output_type]
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))

    if dim == 3:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            savedir,
        )


        dict_metrics['path_pred'], arr_pred_points = reconstruct_3D_airway(savedir)

        if EVALUATE:
            # savepath_viz_comparison = path_aligned_surface + '_comp'
            dict_metrics['hausdorff'], dict_metrics['chamfer'] = evaluate_3D_airway_reconstruction(
                dict_metrics['path_pred'],
                gt['gt_path'],
                savedir)


    print(dict_metrics)
    return dict_metrics


def create_mesh_3dairway_reconstruction_with_transferred_template(
        model,
        lat_vec,
        attr,
        gt,
        savedir,
        output_type='model_out',
        dim=3,
        N=1024,
        device='cpu',
        EVALUATE=True,
):
    dict_metrics = {}

    head = 0
    num_samples = N ** 3
    max_batch = 16 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 3
    voxel_size = 2.0 / (N - 1) * 3

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:,
    2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = ( overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]
    # samples *=2

    start = time.time()

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)
        if type(lat_vec) == dict:
            lat_vec_subset = {}
            for name, value in lat_vec.items():
                lat_vec_subset[name] = lat_vec[name][0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        else:
            lat_vec_subset = lat_vec[0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        samples[head: min(head + max_batch, num_samples), -1] = (
            model.template_transfer(sample_subset[None, :, range(dim)], attr, lat_vec_subset, new_template_type='mean', sex=1)
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))

    if dim == 3:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            savedir,
        )


        dict_metrics['path_pred'], arr_pred_points = reconstruct_3D_airway(savedir)

        if EVALUATE:
            # savepath_viz_comparison = path_aligned_surface + '_comp'
            dict_metrics['hausdorff'], dict_metrics['chamfer'] = evaluate_3D_airway_reconstruction(
                dict_metrics['path_pred'],
                gt['gt_path'],
                savedir)


    print(dict_metrics)
    return dict_metrics





def create_mesh_3dairway_inv_reconstruction(
        model,
        lat_vec,
        attr,
        gt,
        savedir,
        output_type='model_out',
        dim=3,
        N=1024,
        device='cpu',
        EVALUATE=True,
):
    dict_metrics = {}

    head = 0
    num_samples = N ** 3
    max_batch = 16 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 3
    voxel_size = 2.0 / (N - 1) * 3

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:,
    2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (
                                overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = ((
                                 overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]
    # samples *=2

    start = time.time()

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)

        samples[head: min(head + max_batch, num_samples), -1] = (
            model.template_sdf(sample_subset[None, :, range(dim)])
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))

    verts, faces = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        savedir,
    )

    template_samples = torch.from_numpy(verts[None, :, :]).to(device).float()

    lat_vec_subset = lat_vec[0, [0], :].repeat(verts.shape[0], 1).to(device)[None, :, :]
    inv_samples = model.inv_transorm(template_samples, attr, lat_vec_subset, whether_inv=True).squeeze().detach().cpu()

    convert_inv_samples_to_ply(
        inv_samples, faces,
        voxel_origin,
        voxel_size,
        savedir,
        offset=None,
        scale=None,
    )

    path_centerline_normal = gt['ctl_path'][0]
    dict_metrics['path_pred'], arr_pred_points = reconstruct_3D_airway(savedir)

    if EVALUATE:
        # savepath_viz_comparison = path_aligned_surface + '_comp'
        dict_metrics['hausdorff'], dict_metrics['chamfer'] = evaluate_3D_airway_reconstruction(
            dict_metrics['path_pred'],
            gt['gt_path'],
            savedir)

    dict_metrics['path_ctl'] = path_centerline_normal
    print(dict_metrics)
    return dict_metrics


def create_mesh_3dairway_interpolation(
        model,
        lat_vec,
        attr,
        gt,
        savedir,
        output_type='model_out',
        dim=3,
        N=1024,
        device='cpu',
        EVALUATE=False,
):
    dict_metrics = {}

    head = 0
    num_samples = N ** 3
    max_batch = 32 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 3
    voxel_size = 2.0 / (N - 1) * 3

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = (( overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]
    # samples *=2

    start = time.time()

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)
        if type(lat_vec) == dict:
            lat_vec_subset = {}
            for name, value in lat_vec.items():
                lat_vec_subset[name] = lat_vec[name][0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        else:
            lat_vec_subset = lat_vec[0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        samples[head: min(head + max_batch, num_samples), -1] = (
            model(sample_subset[None, :, range(dim)], attr, lat_vec_subset)[output_type]
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))


    if dim == 3:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            savedir,
        )

        dict_metrics['path_pred'], arr_pred_points = reconstruct_3D_airway(savedir)
        # savepath_viz_comparison = path_aligned_surface + '_comp'
        evaluate_3D_airway_interpolation(
            [dict_metrics['path_pred']],
            gt['gt_path'],
            savedir)
    return dict_metrics




def create_mesh_3dairway_template(
        model,
        lat_vec,
        attr,
        gt,
        savedir,
        which_attribute='age',
        output_type='model_out',
        dim=3,
        N=1024,
        device='cpu',
        EVALUATE=False,
):
    dict_metrics = {}

    head = 0
    num_samples = N ** 3
    max_batch = 32 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 3
    voxel_size = 2.0 / (N - 1) * 3

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = (( overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]
    # samples *=2

    start = time.time()

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)
        if type(lat_vec) == dict:
            lat_vec_subset = {}
            for name, value in lat_vec.items():
                lat_vec_subset[name] = lat_vec[name][0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        else:
            lat_vec_subset = lat_vec[0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        samples[head: min(head + max_batch, num_samples), -1] = (
            model.get_template(sample_subset[None, :, range(dim)],  lat_vec_subset)[output_type]
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))


    if dim == 3:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            savedir,
        )

        dict_metrics['path_pred'], arr_pred_points = reconstruct_3D_airway(savedir)
        filename_surface = os.path.join(savedir, 'surface.stl')


        verts, faces, vec_field = extract_data_to_plot_shape_with_vf(filename_surface, attr, lat_vec, model)
        savepath = os.path.join(savedir, which_attribute)
        plotter_3d_airway_reconstruction_with_vf(verts, faces, vec_field[which_attribute], savepath)
        # savepath_viz_comparison = path_aligned_surface + '_comp'

    return dict_metrics

def create_mesh_3dairway_evolution(
        model,
        lat_vec,
        attr,
        gt,
        savedir,
        which_attribute='age',
        output_type='model_out',
        dim=3,
        N=1024,
        device='cpu',
        EVALUATE=False,
):
    dict_metrics = {}

    head = 0
    num_samples = N ** 3
    max_batch = 32 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 3
    voxel_size = 2.0 / (N - 1) * 3

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = (( overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]
    # samples *=2

    start = time.time()

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)
        if type(lat_vec) == dict:
            lat_vec_subset = {}
            for name, value in lat_vec.items():
                lat_vec_subset[name] = lat_vec[name][0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        else:
            lat_vec_subset = lat_vec[0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        samples[head: min(head + max_batch, num_samples), -1] = (
            model.evolution(sample_subset[None, :, range(dim)], attr, lat_vec_subset, which_attribute)[output_type]
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))


    if dim == 3:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            savedir,
        )

        dict_metrics['path_pred'], arr_pred_points = reconstruct_3D_airway(savedir)
        filename_surface = os.path.join(savedir, 'surface.stl')


        #verts, faces, gradient_covariates = extract_data_to_plot_shape_with_vf(filename_surface, attr, lat_vec, model, which_attribute)
        #savepath = os.path.join(savedir, which_attribute)
        #plotter_3d_airway_reconstruction_with_vf(verts, faces, gradient_covariates, savepath)
        # savepath_viz_comparison = path_aligned_surface + '_comp'

    return filename_surface #dict_metrics['path_pred']


def create_mesh_airway_template(
        decoder,
        attributes,
        gt,
        savedir,
        dim=3,
        N=1024,
        device='cpu',
        max_batch=64 ** 3,
        offset=None,
        scale=None,
):
    start = time.time()
    dict_metrics = {}

    decoder.eval()

    max_batch = 32 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 1.5
    voxel_size = 2.0 / (N - 1) * 1.5

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:,
    2] = overall_index % N  # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (
                                overall_index.long() / N) % N  # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = ((
                                 overall_index.long() / N) / N) % N  # overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    # print(samples.max())
    # print(samples.min())
    samples.requires_grad = False

    head = 0

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:dim].to(device)
        attributes = {key: value.to(device) for key, value in attributes.items()}
        samples[head: min(head + max_batch, num_samples), -1] = (
            decoder(sample_subset[None, :, range(dim)], attributes, sample_subset[None, :, range(dim)])['template']
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    end = time.time()
    print("sampling takes: %f" % (end - start))

    filename = str(
        'template')  # filename + '_' + ith_attri + '_' + str(round(attributes[ith_attri].cpu().numpy()[0], 2))
    savedir_template = os.path.join(savedir, filename)
    if not os.path.exists(savedir_template):
        os.mkdir(savedir_template)
    if dim == 3:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            savedir_template,
            offset,
            scale,
        )
        # path_dealigned_surface = path_aligned_surface + '_dealigned'
        path_centerline_normal = gt['ctl_path'][0]
        dict_metrics['path_pred'] = reconstruct_3D_airway(savedir_template)


def convert_sdf_samples_in_2D(pytorch_2d_sdf_tensor, savepath):
    numpy_2d_sdf_tensor = pytorch_2d_sdf_tensor.numpy()
    contours = measure.find_contours(numpy_2d_sdf_tensor, 0.)
    import matplotlib.pyplot as plt
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    # ax.set_axis_on()
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    for contour in contours:
        ax.plot((contour[:, 1] / 255 - 0.5) * 2, (contour[:, 0] / 255 - 0.5) * 2, linewidth=2)

        ax.axis('image')

    # ax.set_axis_on()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # ax.set_xticks(np.arange(-1, 1, 10).tolist())
    # ax.set_yticks(np.arange(-1, 1, 10).tolist())
    # ax.set_title(str(title), fontsize=5)
    plt.savefig(savepath)
    plt.close()

    return


def convert_sdf_samples_to_ply(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        savedir,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((1, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    ##try:
    print(numpy_3d_sdf_tensor.max())
    print(numpy_3d_sdf_tensor.min())
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(
                numpy_3d_sdf_tensor, level=0., spacing=[voxel_size] * 3
            )

    except:
        path_aligned_surface = os.path.join(savedir, 'surface.stl')
        ply_filename_out = os.path.join(savedir, 'surface.ply')
        surface = pv.PolyData(verts)
        surface.save(path_aligned_surface)
        surface.save(ply_filename_out)
        return verts, faces

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    facesq = np.concatenate((np.ones((faces.shape[0], 1)) * 3, faces), axis=-1).astype('int')
    surface = pv.PolyData(mesh_points, faces=facesq)  # .triangulate()
    path_aligned_surface = os.path.join(savedir, 'surface.stl')
    ply_filename_out = os.path.join(savedir, 'surface.ply')
    surface.save(path_aligned_surface)


    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])

    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
    return verts, faces




def convert_inv_samples_to_ply(
        verts, faces,
        voxel_grid_origin,
        voxel_size,
        savedir,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    facesq = np.concatenate((np.ones((faces.shape[0], 1)) * 3, faces), axis=-1).astype('int')
    surface = pv.PolyData(mesh_points, faces=facesq)  # .triangulate()
    path_aligned_surface = os.path.join(savedir, 'surface.stl')
    ply_filename_out = os.path.join(savedir, 'surface.ply')
    surface.save(path_aligned_surface)



    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])

    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
    return



def reconstruct_3D_airway_from_2d(path_centerline_normal, savedir):
    pv_3dcsa = pv.read(os.path.join(savedir, 'surface.stl'))
    arr_centerline_normal = load_pickle(path_centerline_normal)
    eps = 1e-8
    samples_on_ctl = np.linspace(eps, 1.0 - eps, 200)
    # samples_on_ctl = samples_on_ctl[samples_on_ctl>0.49]
    samples_on_ctl = (samples_on_ctl - 0.5) * 2

    list_centerlines = []
    list_normals = []
    for idx in range(len(arr_centerline_normal)):
        arr_centerlines = arr_centerline_normal[idx][idx][1]  # [:, [0, 1, 2]]
        arr_normals = arr_centerline_normal[idx][idx][2]  # [:, [3, 4, 5]]
        # if arr_centerline_normal[idx][idx][0] > 0.49:
        list_centerlines.append(arr_centerlines[None, :])
        list_normals.append(arr_normals[None, :])

    arr_all_centerlines = np.concatenate(list_centerlines, axis=0)
    arr_all_normals = np.concatenate(list_normals, axis=0)

    list_of_dealigned_points = []
    savepath_pred_2dcsa = os.path.join(savedir, 'pred_2dcsa')
    if not os.path.exists(savepath_pred_2dcsa):
        os.mkdir(savepath_pred_2dcsa)
    list_pred_2dcsa = []
    for idx in range(arr_all_centerlines.shape[0]):
        arr_centerlines = arr_all_centerlines[idx]
        arr_normals = arr_all_normals[idx]

        ith_p_on_ctl = samples_on_ctl[idx]
        sliced = pv_3dcsa.slice(origin=np.array([0, 0, ith_p_on_ctl]),
                                normal=np.array([0, 0, 1]),
                                generate_triangles=True)
        current_ponits = sliced.points
        current_ponits[:, -1] = 0

        np.save(os.path.join(savepath_pred_2dcsa, str(idx) + '.npy'), np.array(current_ponits * 10))
        list_pred_2dcsa.append(os.path.join(savepath_pred_2dcsa, str(idx) + '.npy'))
        rmat = rotation_of_a_plane(np.array([0, 0, 1]), arr_normals)

        # rotate

        point_2dcsa = np.matmul(rmat, np.array(sliced.points * 10).T).T
        point_2dcsa += arr_centerlines
        list_of_dealigned_points.append(point_2dcsa)

    arr_dealigned_points = np.concatenate(list_of_dealigned_points, axis=0)
    savepath_pred3d = os.path.join(savedir, 'pred_3d.npy')
    np.save(savepath_pred3d, arr_dealigned_points)
    '''
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    p.add_mesh(pv.PolyData(arr_dealigned_points), point_size=1, color='lightblue')

    p.screenshot(savepath[0:-4] + '.png')
    #p.enable_zoom_style()
    #p.export_vtkjs(savepath)
    p.export_html(savepath[0:-4] + '.html', backend='panel')
    p.close()
    print(savepath)
    '''
    plotter_3d_airway_reconstruction(arr_dealigned_points, savedir)
    return savepath_pred3d, list_pred_2dcsa


def reconstruct_3D_airway(savedir):
    stl_path = os.path.join(savedir, 'surface.stl')
    pv_3dcsa = pv.read(os.path.join(savedir, 'surface.stl'))

    arr_dealigned_points = pv_3dcsa.points * 60  # np.concatenate(list_of_dealigned_points, axis=0)
    savepath_pred3d = os.path.join(savedir, 'pred_3d.npy')
    np.save(savepath_pred3d, arr_dealigned_points)
    '''
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    p.add_mesh(pv.PolyData(arr_dealigned_points), point_size=1, color='lightblue')

    p.screenshot(savepath[0:-4] + '.png')
    #p.enable_zoom_style()
    #p.export_vtkjs(savepath)
    p.export_html(savepath[0:-4] + '.html', backend='panel')
    p.close()
    print(savepath)
    '''
    plotter_3d_airway_reconstruction(arr_dealigned_points, savedir)
    return savepath_pred3d, arr_dealigned_points


def evaluate_3D_airway_reconstruction_from_2d(path_centerline_normal,
                                              list_path_gt_2d_csa,
                                              path_pred_3dcsa,
                                              savepath):
    arr_pred_3dcsa = np.load(path_pred_3dcsa)  # [0:len(list_path_gt_2d_csa)]
    arr_centerline_normal = load_pickle(path_centerline_normal)
    if len(arr_centerline_normal) - len(list_path_gt_2d_csa) < 0:
        return np.nan
    # arr_centerline_normal = arr_centerline_normal[(len(arr_centerline_normal)-len(list_path_gt_2d_csa))::]
    # assert len(arr_centerline_normal) == len(list_path_gt_2d_csa), f"Inconsistent number of 2D CSA with {path_centerline_normal}"
    if len(arr_centerline_normal) != len(list_path_gt_2d_csa):
        return np.nan
    list_centerlines = []
    list_normals = []
    for idx in range(len(arr_centerline_normal)):
        for name, data in arr_centerline_normal[idx].items():
            arr_centerlines = data[1]  # [:, [0, 1, 2]]
            arr_normals = data[2]  # [:, [3, 4, 5]]
            # if arr_centerline_normal[idx][idx][0] > 0.3372:
            list_centerlines.append(arr_centerlines[None, :])
            list_normals.append(arr_normals[None, :])

    arr_all_centerlines = np.concatenate(list_centerlines, axis=0)
    arr_all_normals = np.concatenate(list_normals, axis=0)

    list_of_dealigned_points = []
    for idx in range(arr_all_centerlines.shape[0]):
        arr_centerlines = arr_all_centerlines[idx]
        arr_normals = arr_all_normals[idx]

        current_gt_points = np.load(list_path_gt_2d_csa[idx][0])[:, [0, 1, 2]]
        rmat = rotation_of_a_plane(np.array([0, 0, 1]), arr_normals)

        # rotate
        point_2dcsa = np.matmul(rmat, np.array(current_gt_points).T).T
        point_2dcsa += arr_centerlines
        list_of_dealigned_points.append(point_2dcsa)

    arr_gt_3dcsa = np.concatenate(list_of_dealigned_points, axis=0)
    # try:
    dist = pcu.hausdorff_distance(arr_gt_3dcsa, arr_pred_3dcsa)
    # except:
    #    print(savepath)
    #    dist = np.nan
    '''
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    p.add_mesh(pv.PolyData(arr_gt_3dcsa), point_size=3, color='lightblue', opacity=0.3)
    p.add_mesh(pv.PolyData(arr_pred_3dcsa), point_size=3, color='pink', opacity=0.3)
    p.add_mesh(pv.PolyData(arr_all_centerlines), point_size=10, color='black')

    p.screenshot(savepath + '.png')
    p.export_html(savepath+ '.html', backend='panel')
    p.close()
    print(savepath)
    '''
    plotter_3d_airway_from_2d_evaluation(arr_gt_3dcsa, arr_pred_3dcsa, arr_all_centerlines, savepath)
    # plotter_2d_csa(list_path_gt_2d_csa, list_path_pred_csa, savepath + '_2d_comp')
    return dist


def normalize_3d_airway(arr_pred_point_cloud, arr_gt_point_cloud):
    max_gt = np.max(arr_gt_point_cloud, axis=0)
    min_gt = np.min(arr_gt_point_cloud, axis=0)

    arr_pred_point_cloud = arr_pred_point_cloud[arr_pred_point_cloud[:, 0] <= max_gt[0]]
    arr_pred_point_cloud = arr_pred_point_cloud[arr_pred_point_cloud[:, 1] <= max_gt[1]]
    arr_pred_point_cloud = arr_pred_point_cloud[arr_pred_point_cloud[:, 2] <= max_gt[2]]

    arr_pred_point_cloud = arr_pred_point_cloud[arr_pred_point_cloud[:, 0] >= min_gt[0]]
    arr_pred_point_cloud = arr_pred_point_cloud[arr_pred_point_cloud[:, 1] >= min_gt[1]]
    arr_pred_point_cloud = arr_pred_point_cloud[arr_pred_point_cloud[:, 2] >= min_gt[2]]


    max_length = np.max(np.linalg.norm(arr_gt_point_cloud, axis=-1))
    arr_pred_point_cloud = arr_pred_point_cloud[:, [0,1,2]] / max_length
    arr_gt_point_cloud = arr_gt_point_cloud[:, [0,1,2]]  / max_length
    return arr_pred_point_cloud.astype('float'), arr_gt_point_cloud.astype('float')


def evaluate_3D_airway_reconstruction(
        path_pred_3dcsa,
        path_gt_3dcsa,
        savepath):
    arr_pred_3dcsa = np.load(path_pred_3dcsa)[:, [0, 1, 2]]  # [0:len(list_path_gt_2d_csa)]
    print(path_gt_3dcsa)
    if path_gt_3dcsa[-3::] == 'stl':
        arr_gt_3dcsa = pv.read(path_gt_3dcsa[0]).points[:, [0, 1, 2]]
    else:
        arr_gt_3dcsa = np.load(path_gt_3dcsa[0])[:, [0, 1, 2]]
    arr_pred_3dcsa, arr_gt_3dcsa = normalize_3d_airway(arr_pred_3dcsa, arr_gt_3dcsa)

    h_dist = pcu.hausdorff_distance(arr_gt_3dcsa, arr_pred_3dcsa)
    c_dist = pcu.chamfer_distance(arr_gt_3dcsa, arr_pred_3dcsa)


    # except:
    #    print(savepath)
    #    dist = np.nan
    '''
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    p.add_mesh(pv.PolyData(arr_gt_3dcsa), point_size=3, color='lightblue', opacity=0.3)
    p.add_mesh(pv.PolyData(arr_pred_3dcsa), point_size=3, color='pink', opacity=0.3)
    p.add_mesh(pv.PolyData(arr_all_centerlines), point_size=10, color='black')

    p.screenshot(savepath + '.png')
    p.export_html(savepath+ '.html', backend='panel')
    p.close()
    print(savepath)
    '''
    plotter_3d_airway_evaluation(arr_gt_3dcsa, arr_pred_3dcsa, savepath)
    # plotter_2d_csa(list_path_gt_2d_csa, list_path_pred_csa, savepath + '_2d_comp')
    return h_dist, c_dist



def evaluate_3D_airway_interpolation(
        list_path_pred_3dcsa,
        list_path_gt_3dcsa,
        savepath):
    list_arr_pred = []
    list_arr_gt = []
    for ith_pred in range(len(list_path_pred_3dcsa)):
        list_arr_pred.append(np.load(list_path_pred_3dcsa[ith_pred])[:, [0, 1, 2]] / 60)  # [0:len(list_path_gt_2d_csa)]
    for ith_gt in range(len(list_path_gt_3dcsa)):
        list_arr_gt.append(np.load(list_path_gt_3dcsa[ith_gt])[:, [0, 1, 2]] / 60)  # [0:len(list_path_gt_2d_csa)]

    plotter_3d_airway_interpolation(list_arr_gt, list_arr_pred, savepath)
    return



def save_to_ply(decoder, lat_vec, attr, savedir, device):


    ply_filename_out = os.path.join(savedir, 'surface.ply')
    shape_pred = pv.read(ply_filename_out)
    verts = np.array(shape_pred.points)
    faces = shape_pred.faces
    num_faces = len(faces)
    lat_vec_subset = lat_vec[0, [0], :].repeat(verts.shape[0], 1).to(device)[None, :, :]
    model_output = decoder(torch.from_numpy(verts).to(device).float()[None, :, :], attr, lat_vec_subset)
    if 'vec_fields' in list(model_output.keys()):
        verts_warped = model_output['vec_fields']['overall'].squeeze()
        verts_warped = verts_warped.detach().cpu().numpy()
        verts_warped = verts + verts_warped
    else:
        verts_warped  = verts.squeeze()

    num_verts = verts_warped.shape[0]

    # store canonical coordinates as rgb color (in float format)
    verts_color = 255 * ((verts_warped+2.5) / 5)
    '''
    #verts_color = verts_color.astype(np.uint8)

    verts_tuple = np.zeros(
        (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "f4"), ("green", "f4"), ("blue", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = (verts[i][0], verts[i][1], verts[i][2],
                          verts_color[i][0], verts_color[i][1], verts_color[i][2])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    filename_out = os.path.join(savedir, 'surface_rendered')
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(filename_out + '.ply')
    
    '''
    filename_out = os.path.join(savedir, 'surface_rendered')
    import colorcet as cc
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    print(verts_color.shape)
    print(verts_color[0])
    p.add_mesh(pv.PolyData(verts, faces=pv.read(ply_filename_out).faces), scalars=verts_color, point_size=1, colormap='twilight') #cc.CET_C10s)

    p.screenshot(filename_out + '.png')
    #p.enable_zoom_style()
    #p.export_vtkjs(savepath)
    p.export_html(filename_out + '.html', backend='panel')
    p.close()


def revert_points_to_template(decoder, lat_vec, attr, savedir, device):
    if isinstance(savedir ,str):
        ply_filename_out = os.path.join(savedir, 'surface.stl')
        shape_pred = pv.read(ply_filename_out)
        verts = np.array(shape_pred.points)
    elif isinstance(savedir , torch.Tensor):
        verts = savedir.numpy()

    lat_vec_subset = lat_vec[0, [0], :].repeat(verts.shape[0], 1).to(device)[None, :, :]
    model_output = decoder(torch.from_numpy(verts).to(device).float()[None, :, :], attr, lat_vec_subset)
    if 'vec_fields' in list(model_output.keys()):
        verts_warped = model_output['vec_fields']['overall'].squeeze()
        verts_warped = verts_warped.detach().cpu().numpy()
        verts_warped = verts + verts_warped
    else:
        verts_warped = verts.squeeze()
    return verts_warped



def revert_points_to_template_for_one_cov(decoder, lat_vec, attr, savedir, device, which_covariate):
    if isinstance(savedir ,str):
        ply_filename_out = os.path.join(savedir, 'surface.stl')
        shape_pred = pv.read(ply_filename_out)
        verts = np.array(shape_pred.points)
    elif isinstance(savedir , torch.Tensor):
        verts = savedir.numpy()

    lat_vec_subset = lat_vec[0, [0], :].repeat(verts.shape[0], 1).to(device)[None, :, :]
    model_output = decoder.evolution(torch.from_numpy(verts).to(device).float()[None, :, :], attr, lat_vec_subset, which_covariate)
    if 'vec_fields' in list(model_output.keys()):
        verts_warped = model_output['vec_fields'][which_covariate].squeeze()
        verts_warped = verts_warped.detach().cpu().numpy()
        verts_warped = verts + verts_warped
    else:
        verts_warped = verts.squeeze()
    return verts_warped