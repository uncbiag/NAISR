#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    print(gt_to_gen_chamfer + gen_to_gt_chamfer, gt_to_gen_chamfer, gen_to_gt_chamfer)
    return gt_to_gen_chamfer + gen_to_gt_chamfer

def compute_depth_chamfer(gt_points, gen_mesh, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    '''
    import pyvista as pv
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1, 1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background = 'white'
    pv.global_theme.interactive = True

    p.add_mesh(gt_points, opacity=0.3, color='grey')
    p.add_mesh( gen_points_sampled, opacity=0.3, color='pink')
    p.screenshot('/home/jyn/NAISR/data_generation/a.png')
    p.export_html('/home/jyn/NAISR/data_generation/a.html', backend='panel')
    p.close()

    import pyvista as pv
    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1, 1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background = 'white'
    pv.global_theme.interactive = True

    p.add_mesh(gt_points, opacity=0.3, color='grey')
    p.add_mesh(gen_mesh.vertices, opacity=0.3, color='pink')
    p.screenshot('/home/jyn/NAISR/data_generation/b.png')
    p.export_html('/home/jyn/NAISR/data_generation/b.html', backend='panel')
    p.close()
    '''


    #gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points

    # one direction
    #gen_points_kd_tree = KDTree(gen_points_sampled)
    #one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    #gt_to_gen_chamfer = np.mean(np.abs(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.abs(two_distances))

    #gen_to_gt_chamfer = pcu.chamfer_distance(gen_points_sampled, gt_points_np)
    #gt_to_gen_chamfer = pcu.chamfer_distance(gt_points_np, gen_points_sampled)
    #print(gt_to_gen_chamfer + gen_to_gt_chamfer, gt_to_gen_chamfer, gen_to_gt_chamfer)
    #return (gt_to_gen_chamfer + gen_to_gt_chamfer) /2
    #return gt_to_gen_chamfer
    return  gen_to_gt_chamfer

