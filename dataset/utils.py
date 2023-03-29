import pyvista as pv
from pyvista import examples
import numpy as np

def read_a_toy_shape(filename, savepath, num_of_samples=2000):
    #if shapetype == 'ellipsoid':

    mesh = pv.read(filename)
    mesh = pv.CylinderStructured(radius=2, height=1.0, theta_resolution=500, z_resolution=500)
    data_to_probe = examples.load_uniform()
    result = mesh.sample(data_to_probe)


    slt_idx = np.random.randint(0, len(result.points), num_of_samples)
    sampled_points = result.points[slt_idx]
    normals = result.compute_normals()
    sampled_norms = normals.points[slt_idx]

    #data = {'pts': sampled_points,
    #        'norms': sampled_norms}

    np.savez(savepath, pts=sampled_points, norms=sampled_norms)
    return result

def read_a_toy_shape(filename, savepath, num_of_samples=2000):
    #if shapetype == 'ellipsoid':

    mesh = pv.read(filename)

    data_to_probe = examples.load_uniform()
    result = mesh.sample(data_to_probe)


    slt_idx = np.random.randint(0, len(result.points), num_of_samples)
    sampled_points = result.points[slt_idx]
    normals = result.compute_normals()
    sampled_norms = normals.points[slt_idx]

    #data = {'pts': sampled_points,
    #        'norms': sampled_norms}

    np.savez(savepath, pts=sampled_points, norms=sampled_norms)
    return #result

#filename = '/Users/jyn/jyn/research/projects/NAISR/toydata/cylinder/cylinder_height_1.0_radius_1.0.stl' #'/Users/jyn/jyn/research/projects/NAISR/toydata/torus/torus_ringradius_10.0_crosssectionradius_5.0.stl'
#savepath = 'cylinder.npz'
#pts = read_a_toy_shape(filename, savepath)

#read_a_toy_shape(filename, savepath, num_of_samples=2000)