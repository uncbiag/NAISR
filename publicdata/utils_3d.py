import argparse

from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np
import SimpleITK as sitk
import trimesh
import mesh_to_sdf

def calculate_sdf_from_mesh(surf_mesh):
    # get centered mesh: mesh
    mesh = surf_mesh.copy()
    centroid = surf_mesh.bounding_box.centroid
    vertices = surf_mesh.vertices - surf_mesh.bounding_box.centroid
    distances = np.max(np.linalg.norm(vertices, axis=1))

    # get sdf field, the mesh is normalized automatically
    points, sdf = sample_sdf_near_surface(surf_mesh, number_of_points=250000, sign_method='depth')
    # get off-surface points
    off_surf_points = generate_off_surface_points(number_of_points=250000) + centroid
    off_surf_sdf = mesh_to_sdf.mesh_to_sdf(mesh, off_surf_points, surface_point_method='scan', sign_method='depth', bounding_radius=None, scan_count=100, scan_resolution=1000, sample_point_count=10000000, normal_sample_count=11)
    off_surf_points = off_surf_points[off_surf_sdf > 0 * 0.1]
    off_surf_sdf = off_surf_sdf[off_surf_sdf > 0 * 0.1]
    # denormalize
    points = points * distances + centroid
    sdf *= distances
    #
    points = np.concatenate((points, off_surf_points), axis=-2)
    sdf = np.concatenate((sdf[:, None], off_surf_sdf[:, None]), axis=0)


    '''
    colors = np.zeros(points.shape)
    colors[sdf[:, 0] < 0, 2] = 1
    colors[sdf[:, 0] > 6, 1] = 1

    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
    #viewer.save_gif(savepath)
    viewer.close()
    '''
    npz_sdf = np.concatenate((points, sdf), axis=-1)
    #np.save(savepath, npz_sdf)
    del points, sdf, off_surf_points, off_surf_sdf
    return npz_sdf



def generate_off_surface_points(number_of_points=250000):

    # numbers set according to the hippocampus dataset
    x = (np.random.rand(number_of_points) - 0.5) * 60
    y = (np.random.rand(number_of_points) - 0.5) * 60
    z = (np.random.rand(number_of_points) - 0.5) * 60

    coords = np.concatenate((x[:, None], y[:, None], z[:, None]), axis=-1)
    return coords

