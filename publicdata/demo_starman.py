import pyvista as pv
import numpy as np
from pyvista import examples
import argparse

from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np
import SimpleITK as sitk
import trimesh
import mesh_to_sdf

def Gaussian(x,z,sigma=0.5):
    return np.exp((-(np.linalg.norm(x-z, axis=1)**2))/(2*sigma**2))


def create_deformation(points, control_point):
    deformed = Gaussian(points, control_point)[:, None] @ UP_VECTOR[None, :] #+ points
    return deformed

def apply_deformation(points, deformation, covariate_value=1.):
    return points + deformation * covariate_value


def compute_normals(arr_contour):
    arr_contour_pts = arr_contour.copy()
    arr_contour_pts = arr_contour_pts[:, [0, 1]]
    normals = []
    for i in range(len(arr_contour_pts)):
        x1, y1 = arr_contour_pts[i]
        x2, y2 = arr_contour_pts[(i+1) % len(arr_contour_pts)]  # wrap around for the last point
        dx = x2 - x1
        dy = y2 - y1
        normals.append((-dy, dx))
    arr_normals = np.array(normals) / np.linalg.norm(normals, axis=-1, keepdims=1)
    arr_normals = np.concatenate((arr_normals, np.zeros_like(arr_normals[:, [0]])), axis=-1)
    return arr_normals


def generate_off_surface_points(number_of_points=250000):
    x = (np.random.rand(number_of_points) - 0.5) * 6 * 60
    y = (np.random.rand(number_of_points) - 0.5) * 6 * 60
    z = (np.random.rand(number_of_points) - 0.5) * 6 * 60

    coords = np.concatenate((x[:, None], y[:, None], z[:, None]), axis=-1)
    return coords


def apply_registration_to_surface(surf, R, T):
    new_verts = T + np.matmul(R, np.array(surf.vertices).T)
    new_surf = trimesh.Trimesh(vertices=new_verts.T, faces=surf.faces)
    return new_surf

def get_centroids(arr_bbx):
    xmin, xmax, ymin, ymax, zmin, zmax = arr_bbx
    arr_center = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    return arr_center



def calculate_sdf_from_mesh(surf_mesh):
    import trimesh
    # get centered mesh: mesh
    mesh = surf_mesh
    centroid = get_centroids(mesh.bounds)
    vertices = mesh.points - centroid
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

    npz_sdf = np.concatenate((points, sdf), axis=-1)
    #np.save(savepath, npz_sdf)
    return npz_sdf




UP_VECTOR = np.array([0., 1., 0.])


a = np.array([0., 1., 0.])


path_starman_vtk = "/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__Template__GroundTruth.vtk"
path_control_points = "/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__ControlPoints__GoundTruth.txt"
   # "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen/data/ForInitialization__ControlPoints__FromLongitudinalAtlas.txt"

mesh = pv.read(path_starman_vtk)
#mesh.points = mesh.points[:, [0,1]]

control_points = np.loadtxt(path_control_points)
arr_cp = np.concatenate((control_points, np.zeros_like(control_points[:, [0]])), axis=1)


chart = pv.Chart2D()
x = mesh.points[:, 0]
y = mesh.points[:, 1]
_ = chart.line(x, y, "y", 4)
#chart.show()




pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_points(arr_cp[[2]], color='r')

# 1.
arr_deformed = apply_deformation(points=mesh.points,
                                 deformation=create_deformation(mesh.points, control_point=arr_cp[2]),
                                 covariate_value=1.)
deformed_mesh = mesh.copy()
deformed_mesh.points = arr_deformed
deformed_mesh.normals = compute_normals(deformed_mesh.points)
import trimesh
#a=calculate_sdf_from_mesh(deformed_mesh)

#deformed_mesh.save('./demo.vtk')
pl.add_mesh(deformed_mesh, color='y')

# 0.1
arr_deformed = apply_deformation(points=mesh.points,
                                 deformation=create_deformation(mesh.points, control_point=arr_cp[0]),
                                 covariate_value=0.1)
deformed_mesh1 = mesh.copy()
deformed_mesh1.points = arr_deformed
pl.add_mesh(deformed_mesh1, color='g')


# 0.5
arr_deformed = apply_deformation(points=mesh.points,
                                 deformation=create_deformation(mesh.points, control_point=arr_cp[0]),
                                 covariate_value=0.5)
deformed_mesh2 = mesh.copy()
deformed_mesh2.points = arr_deformed
pl.add_mesh(deformed_mesh2, color='b')

# 0.1
arr_deformed = apply_deformation(points=mesh.points,
                                 deformation=create_deformation(mesh.points, control_point=arr_cp[0]),
                                 covariate_value=-0.1)
deformed_mesh3 = mesh.copy()
deformed_mesh3.points = arr_deformed
pl.add_mesh(deformed_mesh3, color='cyan')


# 0.5
arr_deformed = apply_deformation(points=mesh.points,
                                 deformation=create_deformation(mesh.points, control_point=arr_cp[0]),
                                 covariate_value=-0.5)
deformed_mesh4 = mesh.copy()
deformed_mesh4.points = arr_deformed
pl.add_mesh(deformed_mesh4, color='purple')



pl.view_xy()
pl.show()