import os
import numpy as np
import pandas as pd
import math
import pyvista as pv
import trimesh
from utils_3d import *
import skimage.measure as measure
import datetime


def read_origin_spacing(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    arr_rst = np.zeros_like(arr)
    if (np.unique(arr) != np.array([0, 20, 21], dtype=np.int16)).sum():
        arr = arr / 256
        print('Inconsistent label value: ' + path)
    arr_rst[arr == 20] = 20
    Origin = np.array(img.GetOrigin())
    Spacing = np.array(img.GetSpacing())
    return arr_rst, Origin, Spacing


def reverse_coords(arr):
    rev = []
    for x, y, z in arr:
        rev.append([z, y, x])
    return np.array(rev)


def read_and_align_mesh(path_source, path_target):
    mesh_target = surface_construction(path_target)
    mesh_source = surface_construction(path_source)

    _, aligned_vertices, _ = trimesh.registration.icp(mesh_source.vertices, mesh_target.vertices, scale=False)
    mesh_aligned = mesh_source.copy()
    mesh_aligned.vertices = aligned_vertices

    # scene = pyrender.Scene()
    # cloud = pyrender.Mesh.from_points(np.array(mesh_source.vertices), colors=(0,0,0.5))
    # scene.add(cloud)
    # cloud = pyrender.Mesh.from_points(mesh_target.vertices, colors= (0,0.5,0))
    # scene.add(cloud)
    # cloud = pyrender.Mesh.from_points(mesh_aligned.vertices, colors= (0.5,0,0))
    # scene.add(cloud)
    # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
    # # viewer.save_gif(savepath)
    # viewer.close()
    return mesh_aligned, mesh_source




def surface_construction(pathseg):
    # unpacking
    arr_airway, Origin, Spacing = read_origin_spacing(pathseg)
    # polydata of airway surface segments
    verts, faces, norm, val = \
        measure.marching_cubes_lewiner(arr_airway,
                                       spacing=Spacing,
                                       allow_degenerate=True)
    current_airway = pv.PolyData()
    current_airway.points = verts #(verts + Origin) #* np.array([-1, -1, 1])  # + np.array([-35, 0, 0])
    current_airway.faces = np.hstack((np.ones((faces.shape[0], 1)) * 3, faces)).ravel().astype('int')
    airway_surface =  current_airway#.extract_surface()#.smooth(n_iter=200)



    mesh = trimesh.Trimesh(vertices=airway_surface.points, faces=faces, )
    vertices = mesh.vertices - mesh.bounding_box.centroid
    mesh.vertices = vertices
    return mesh


def generate_shape_and_sdf_for_a_seg(path_seg, root_dataset, path_target):


    # get id
    scan_id = path_seg.split('/')[-1][0:-4]

    # get savepath
    savepath_on_npy = os.path.join(root_dataset, '3dshape', scan_id.split('.')[0] + '_on.npy')
    savepath_off_npy = os.path.join(root_dataset, '3dsdf', scan_id.split('.')[0] + '_off.npy')
    savepath_on_pv_aligned = os.path.join(root_dataset, '3dvis', scan_id.split('.')[0] + '_on_aligned.stl')
    savepath_on_pv_source = os.path.join(root_dataset, '3dvis', scan_id.split('.')[0] + '_on_source.stl')
    # if os.path.exists(savepath_on_pv) and os.path.exists(savepath_on_npy) and os.path.exists(savepath_off_npy):
    #     return
    #path_examples = "/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data/sub-ADNI002S0729_ses-M00.vtk"

    # off surface points
    #mesh = surface_construction(path_seg)
    mesh, mesh_source = read_and_align_mesh(path_seg, path_target)
    # on surface points
    mesh.export(savepath_on_pv_aligned)
    mesh_source.export(savepath_on_pv_source)

    del mesh_source

    # off surface points
    arr_off = calculate_sdf_from_mesh(mesh)

    np.save(savepath_on_npy, np.concatenate((np.array(mesh.vertices), np.array(mesh.vertex_normals)), axis=-1))
    np.save(savepath_off_npy, arr_off)

    #del arr_off, mesh, pv_data

    return

def generate_volume_for_a_seg(path_seg, path_target):
    # off surface points
    mesh = read_and_align_mesh(path_seg, path_target)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Given some groups (can be scans for one subject), this script generates their airway min-max plot")
    parser.add_argument("--path_seg", type=str, required=True)
    parser.add_argument("--root_dataset", type=str, required=True)
    parser.add_argument("--path_target", type=str, required=True)
    args = parser.parse_args()

    root_dataset = args.root_dataset
    path_seg = args.path_seg
    path_target = args.path_target
    generate_shape_and_sdf_for_a_seg(path_seg, root_dataset, path_target)