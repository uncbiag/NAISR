import os


import argparse

import numpy as np

from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import pyvista as pv

import utils
UP_VECTOR = np.array([0., 1.])
FORWARD_VECTOR = np.array([1., 0.]) #* 0.4

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resample_polygon(xy: np.ndarray, n_points: int = 100) -> np.ndarray:
    # Cumulative Euclidean distance between successive polygon points.
    # This will be the "x" for interpolation
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

    # get linearly spaced points along the cumulative Euclidean distance
    d_sampled = np.linspace(0, d.max(), n_points)

    # interpolate x and y coordinates
    xy_interp = np.c_[
        np.interp(d_sampled, d, xy[:, 0]),
        np.interp(d_sampled, d, xy[:, 1]),
    ]

    return xy_interp


def show_shapely_polygon(polygon):
    fig, ax = plt.subplots()
    ax.plot(*polygon.exterior.xy, 'b-', label='Polygon')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    plt.show()
    return


def signed_distance_polygon(polygon, point):
    """
    Calculate the signed distance from a point to a polygon.

    Parameters:
    polygon (Polygon): Shapely Polygon object representing the polygon.
    point (Point): Shapely Point object representing the point.

    Returns:
    float: Signed distance from the point to the polygon.
    """
    if polygon.contains(point):
        return -point.distance(polygon.boundary)
    else:
        return point.distance(polygon.boundary)


def extract_2d_sdf_from_contour(query_points, arr_contour_pts):
    polygon = Polygon(arr_contour_pts)
    # show_shapely_polygon(polygon)
    # Define a point for which you want to calculate the signed distance
    list_of_2dsdf = []
    for ith_query in query_points:
        query_point = Point(ith_query)
        # Calculate the signed distance
        signed_dist = signed_distance_polygon(polygon, query_point)
        # get 2dsdf
        list_of_2dsdf.append(signed_dist)
    return np.array(list_of_2dsdf)[:, None]


def generate_off_surface_points(number_of_points=250000):
    # the range of the data is around [-2, 2]
    x = (np.random.rand(number_of_points) - 0.5) * 6
    y = (np.random.rand(number_of_points) - 0.5) * 6
    coords = np.concatenate((x[:, None], y[:, None]), axis=-1)
    return coords


def calculate_2dsdf_from_polygon(arr_contour_pts, number_of_points=500000):
    query_points = generate_off_surface_points(number_of_points=number_of_points)
    arr_sdf = extract_2d_sdf_from_contour(query_points, arr_contour_pts)
    npz_sdf = np.concatenate((query_points, arr_sdf), axis=-1)
    return npz_sdf




def Gaussian(x, z, sigma=0.5):
    return np.exp((-(np.linalg.norm(x - z, axis=1) ** 2)) / (2 * sigma ** 2))


def create_deformation(points, control_point, VECTOR):
    deformed = Gaussian(points, control_point)[:, None] @ VECTOR[None, :]  # + points
    return deformed


def apply_deformation(points, deformation, covariate_value=1.):
    return points + deformation * covariate_value


def compute_normals(arr_contour):
    arr_contour_pts = arr_contour.copy()
    arr_contour_pts = arr_contour_pts[:, [0, 1]]
    normals = []
    for i in range(len(arr_contour_pts)):
        x1, y1 = arr_contour_pts[i]
        x2, y2 = arr_contour_pts[(i + 1) % len(arr_contour_pts)]  # wrap around for the last point
        dx = x2 - x1
        dy = y2 - y1
        normals.append((-dy, dx))
    arr_normals = np.array(normals) / np.linalg.norm(normals, axis=-1, keepdims=1)
    #arr_normals = np.concatenate((arr_normals, np.zeros_like(arr_normals[:, [0]])), axis=-1)
    return arr_normals


def sampling_the_covariate_space(num_of_shapes, num_of_covariates=4):

    list_of_covariates = []
    for i in range(num_of_shapes):
        list_of_covariates.append(np.random.rand(num_of_covariates))
    arr_covariates = 2 * (np.array(list_of_covariates) - 0.5)
    return arr_covariates


def generate_shape_conditioned_on_covariates(arr_points, covariates, arr_control_pts):
    covariates = [covariates[0], covariates[0], covariates[1], covariates[1]]
    arr_deformed = arr_points.copy()
    #arr_deformed = polydata.points.copy()
    list_of_VECTORS = [UP_VECTOR, UP_VECTOR, FORWARD_VECTOR * (-1), FORWARD_VECTOR]
    for ith_cov in range(len(covariates)):
        if ith_cov < 2:
            ith_control_pts = arr_control_pts[ith_cov]
            arr_deformed = apply_deformation(points=arr_deformed,
                                             deformation= create_deformation(arr_points, control_point=ith_control_pts, VECTOR=list_of_VECTORS[ith_cov]),
                                             covariate_value=covariates[ith_cov])
        else:
            ith_control_pts = arr_control_pts[ith_cov]
            arr_deformed = apply_deformation(points=arr_deformed,
                                             deformation= create_deformation(arr_points, control_point=ith_control_pts, VECTOR=list_of_VECTORS[ith_cov]),
                                             covariate_value=covariates[ith_cov] + 1)
    return arr_deformed



def generate_dataset_for_one_subject(PID,
                                     pv_contour,
                                     arr_control_pts,
                                     num_of_shapes,
                                     num_of_covariates=2,
                                     rootoath='/playpen-raid/jyn/NAISR/NAISR/examples/starmen/'):

    arr_contour_pts = pv_contour.points[:, [0,1]]

    # get path
    path_2d_shape = os.path.join(rootoath, '2dshape')
    path_2d_sdf = os.path.join(rootoath, '2dsdf')
    cond_mkdir(path_2d_shape)
    cond_mkdir(path_2d_sdf)


    # sample covariates uniformly
    arr_covariates = sampling_the_covariate_space(num_of_shapes, num_of_covariates)

    list_current_data = []
    for i in range(len(arr_covariates)):
        # 2d shape, which are the on-surface points, with normal vectors
        arr_deformed_points = generate_shape_conditioned_on_covariates(arr_contour_pts, arr_covariates[i], arr_control_pts)
        arr_deformed_normals = compute_normals(arr_deformed_points)

        arr_on = np.concatenate((arr_deformed_points, arr_deformed_normals), axis=-1)

        # 2d sdf, which are points with sdf
        arr_off = calculate_2dsdf_from_polygon(arr_deformed_points, number_of_points=5000)

        current_id = str(PID) + '_' + str(i) #str(arr_covariates[i][0]) + '_' + str(arr_covariates[i][1]) + '_' + str(arr_covariates[i][2]) + '_' + str(arr_covariates[i][3])
        # get savepath
        filename_on_npy =  current_id + '_on.npy'
        filename_off_npy = current_id + '_off.npy'

        savepath_on_npy = os.path.join(path_2d_shape, filename_on_npy)
        savepath_off_npy = os.path.join(path_2d_sdf, filename_off_npy)

        np.save(savepath_on_npy, arr_on)
        np.save(savepath_off_npy, arr_off)

        # save visualize
        filename_on_vtk = current_id + '_on.vtk'

        savepath_on_vis = os.path.join(path_2d_shape, filename_on_vtk)
        pv_deformed = pv_contour.copy()
        pv_deformed.points = np.concatenate((arr_on[:, [0,1]], np.zeros_like(arr_on[:, [0]])), axis=-1)
        pv_deformed.save(savepath_on_vis)

        #
        list_current_data.append({"id": current_id,
                                  "PID": PID,
                                  "cov_1": arr_covariates[i][0],
                                  "cov_2": arr_covariates[i][1],
                                  "2dsdf":savepath_off_npy,
                                  "2dshape": savepath_on_npy})

    return list_current_data


def save_personalized_template(pv_template, new_verts, rootpath, PID):
    pv_new_template = pv_template.copy()
    pv_new_template.points = np.concatenate((new_verts, np.zeros_like(new_verts[:, [0]])), axis=-1)
    dir_template = os.path.join(rootpath, 'template')
    cond_mkdir(dir_template)
    savepath = os.path.join(dir_template, str(PID) + "_template.vtk")
    pv_new_template.save(savepath)
    return

def save_personalized_template_contour_pts(new_verts, rootpath, PID):
    #savepath = os.path.join(rootpath, 'template', str(PID) + "_template_contour_pts.npy")
    dir_template = os.path.join(rootpath, 'template')
    cond_mkdir(dir_template)
    savepath = os.path.join(dir_template, str(PID) + "_template_contour_pts.npy")
    np.save(savepath, new_verts)
    return

def save_personalized_template_control_pts(new_verts, rootpath, PID):

    #pv_new_template = pv_template.copy()
    #pv_new_template.points = np.concatenate((new_verts, np.zeros_like(new_verts[:, [0]])), axis=-1)
    dir_template = os.path.join(rootpath, 'template')
    cond_mkdir(dir_template)
    savepath = os.path.join(dir_template, str(PID) + "_template_control_pts.npy")
    np.save(savepath, new_verts)
    return

if __name__ == "__main__":
    path_starman_vtk = "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__Template__GroundTruth.vtk"
    path_control_points = "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__ControlPoints__GoundTruth.txt"
    # "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen/data/ForInitialization__ControlPoints__FromLongitudinalAtlas.txt"

    mesh = pv.read(path_starman_vtk)
    # mesh.points = mesh.points[:, [0,1]]
    control_points = np.loadtxt(path_control_points)
    arr_cp = np.concatenate((control_points, np.zeros_like(control_points[:, [0]])), axis=1)

    chart = pv.Chart2D()
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    _ = chart.line(x, y, "y", 4)
    # chart.show()


    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_points(arr_cp[[0]], color='r')

    # 1.
    arr_deformed = apply_deformation(points=mesh.points,
                                     deformation=create_deformation(mesh.points, control_point=arr_cp[0]),
                                     covariate_value=1.)
    deformed_mesh = mesh.copy()
    deformed_mesh.points = arr_deformed
    deformed_mesh.normals = compute_normals(deformed_mesh.points)
    deformed_mesh.save('./demo.vtk')
    pl.add_mesh(deformed_mesh, color='y')
    pl.add_arrows(deformed_mesh.points, deformed_mesh.normals )

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


