import os

import pyvista as pv
import numpy as np
import pandas as pd
filename = '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/pos_on_ctl_1120.csv'
a = pd.read_csv(filename)





path= '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/1036_slice_1.npy'

rootdir_template = '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/1035/'
list_slices_template = os.listdir(rootdir_template)
airway_pc_template = []
for i_slice in list_slices_template:
    i_path = os.path.join(rootdir_template, i_slice)
    points = np.load(i_path)
    airway_pc_template += points.tolist()


rootdir_source = '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/1042/'
list_slices_source = os.listdir(rootdir_source)
airway_pc_source = []
for i_slice in list_slices_source:
    i_path = os.path.join(rootdir_source, i_slice)
    points = np.load(i_path)
    airway_pc_source += points.tolist()



rootdir_1 = '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/1050/'
list_slices_1 = os.listdir(rootdir_1)
airway_pc_1 = []
for i_slice in list_slices_1:
    i_path = os.path.join(rootdir_1, i_slice)
    points = np.load(i_path)
    airway_pc_1 += points.tolist()



rootdir_2 = '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/1285/'
list_slices_2 = os.listdir(rootdir_2)
airway_pc_2 = []
for i_slice in list_slices_2:
    i_path = os.path.join(rootdir_2, i_slice)
    points = np.load(i_path)
    airway_pc_2 += points.tolist()



def compute_frenet_frames_from_centerline(arr_centerline):
    # Number of points
    n = len(arr_centerline)

    # Calculate the first and second derivative of the points
    dX = np.apply_along_axis(np.gradient, axis=0, arr=arr_centerline)
    ddX = np.apply_along_axis(np.gradient, axis=0, arr=dX)

    # Normalize all tangents
    f = lambda m: m / np.linalg.norm(m+1e-5)
    T = np.apply_along_axis(f, axis=1, arr=dX)

    # Calculate and normalize all binormals
    B = np.cross(dX, ddX)
    B = np.apply_along_axis(f, axis=1, arr=B)

    # Calculate all normals
    N = np.cross(B, T)

    return T, B, N


def reconstructing_point_sets_of_airway_surface(arr_centerline, arr_oct_radius, PLOT=False):
    list_of_points = []
    # pv_ctl = pv.PolyData(arr_centerline)
    # pv_ctl.plot()
    # get frenet frames for the centerline curve

    # T, B, N = compute_frenet_frames_from_centerline(arr_centerline)
    '''
    pv_T = pv.PolyData(T + arr_centerline)
    pv_B = pv.PolyData(B + arr_centerline)
    pv_N = pv.PolyData(N + arr_centerline)

    if PLOT:
        p = pv.Plotter()
        p.add_mesh(pv_T, color="b",  render_points_as_spheres=True)
        p.add_mesh(pv_N, color="r",  render_points_as_spheres=True)
        p.add_mesh(pv_B, color="y",  render_points_as_spheres=True)
        p.add_mesh(pv.PolyData(arr_centerline ), color="g", render_points_as_spheres=True)
        p.show()
    '''

    # Calculate the first and second derivative of the points
    # arr_centerline = arr_centerline - arr_centerline[0]
    dX = np.zeros_like(arr_centerline)
    dX[1::, :] = arr_centerline[1::] - arr_centerline[0:-1]
    dX[0, :] = arr_centerline[1] - arr_centerline[0]

    list_coords = []
    # get cartesian coordinates of the points on the airway surface
    arr_cart_points = get_cartesian_from_polar_oct_points(arr_oct_radius)
    '''
    for ith_coords in range(len(T)):

        if ith_coords == 0:
            # get current coordinates
            current_coords = np.eye(3)
            #centroid = np.mean(current_coords, axis=0)
            # centre the points
            #current_coords = current_coords - np.tile(centroid, (1, 1))


            #R_z, t_z = rigid_transform_3D(current_coords[-1][None, :], T[0][None, :])
            transform = orthonormal_matrix_for_a_to_b(current_coords[-1], dX[0])

            #current_coords = np.matmul(current_coords, R_z.T) + t_z.reshape([1, 3])
            current_coords = transform
            Z_axis = current_coords[-1, :]
            #currents_coords = np.dstack((X_axis, Y_axis, Z_axis))[0].T
            list_coords.append(current_coords)

        elif ith_coords < len(T) -1:
            p1_ctl = arr_centerline[ith_coords]
            p2_ctl = arr_centerline[ith_coords+1]
            #X_axis, Y_axis, Z_axis = get_current_coordinates(X_axis,Y_axis, Z_axis, p1_ctl, p2_ctl, delta_omega)
            #R_z, t_z = rigid_transform_3D(Z_axis[None, :], T[ith_coords][None, :])
            current_coords = np.eye(3)
            current_coords = orthonormal_matrix_for_a_to_b(current_coords[-1], dX[ith_coords])
            #current_coords = np.matmul(current_coords, R_z.T) + t_z.reshape([1, 3])
            Z_axis = current_coords[-1, :]

            #Z_axis = T[ith_coords]
            #currents_coords = np.dstack((X_axis, Y_axis, Z_axis))[0].T
            list_coords.append(current_coords)
        else:
            #p1_ctl = arr_centerline[ith_coords]
            #p2_ctl = arr_centerline[ith_coords] +  (arr_centerline[ith_coords] - arr_centerline[ith_coords-1])
            #X_axis, Y_axis, Z_axis = get_current_coordinates(X_axis, Y_axis, Z_axis, p1_ctl, p2_ctl, delta_omega)
            #Z_axis = T[ith_coords-1]
            #currents_coords = np.dstack((X_axis, Y_axis, Z_axis))[0].T
            current_coords = current_coords + current_coords - list_coords[-2]
            list_coords.append(current_coords)
        '''
    p_axis = np.eye(3)[-1]
    for ith_coords in range(len(dX)):
        current_coords = orthonormal_matrix_for_a_to_b(p_axis, dX[ith_coords])
        list_coords.append(current_coords)

        transformed_ps = np.matmul(arr_cart_points[ith_coords], current_coords) + arr_centerline[ith_coords]
        list_of_points.append(transformed_ps)

    if PLOT:
        pv_T = pv.PolyData(np.array(list_coords)[:, 2, :] + arr_centerline)
        pv_B = pv.PolyData(np.array(list_coords)[:, 1, :] + arr_centerline)
        pv_N = pv.PolyData(np.array(list_coords)[:, 0, :] + arr_centerline)
        p = pv.Plotter()
        p.add_mesh(pv_T, color="b", render_points_as_spheres=True)
        p.add_mesh(pv_N, color="r", render_points_as_spheres=True)
        p.add_mesh(pv_B, color="y", render_points_as_spheres=True)
        p.add_mesh(pv.PolyData(np.array(list_of_points).reshape(-1, 3)), color="r", point_size=0.01,
                   render_points_as_spheres=True)
        # p.show()

    return np.array(list_of_points).reshape(-1, 3)


p = pv.Plotter()
p.add_mesh(np.array(airway_pc_template), point_size=0.1, color='b')
p.add_mesh(np.array(airway_pc_source), point_size=0.1, color='r')
#p.add_mesh(np.array(airway_pc_1), point_size=0.1, color='y')
#p.add_mesh(np.array(airway_pc_2), point_size=0.1, color='g')
p.show()
print('1')



import pickle
def pickle_load_object(f_path):
    with open(f_path, "rb") as f_in:
        obj = pickle.load(f_in)
    return obj

path_centerline = '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/1035_CENTERLINE.p3'
arr_centerline = pickle_load_object(path_centerline)

p = pv.Plotter()
p.add_mesh(np.array(airway_pc_template), point_size=0.1, color='b')
p.add_mesh(arr_centerline[0], point_size=0.1, color='r')
p.show()

print('1')