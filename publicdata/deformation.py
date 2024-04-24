import numpy as np
from scipy.ndimage import gaussian_filter


def rotate_polygon(polygon, rotation_deg=30):
    """
    Rotate the polygon.
    """
    theta = np.deg2rad(rotation_deg)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Apply the rotation
    rotated_polygon = np.dot(polygon, rotation_matrix.T)

    return rotated_polygon


def translate_polygon(polygon, translation=(10, 10)):
    """
    Translate the polygon.
    """
    translated_polygon = polygon + translation
    return translated_polygon

def Gaussian(x, z, sigma=0.5):
    return np.exp((-(np.linalg.norm(x - z, axis=1) ** 2)) / (2 * sigma ** 2))

def create_deformation(points, control_point, VECTOR):
    deformed = Gaussian(points, control_point)[:, None] @ VECTOR[None, :]  # + points
    return deformed

def random_deform_polygon(arr_contour_pts, arr_control_pts):
    """
    Add random deformation to the polygon.
    """

    nx, ny = (5, 5)
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    xv, yv = np.meshgrid(x, y)
    arr_control_grids = np.concatenate((xv.ravel()[:, None], yv.ravel()[:, None]), axis=1)

    arr_deformation_contour = np.zeros_like(arr_contour_pts.copy())
    arr_deformation_control = np.zeros_like(arr_control_pts.copy())
    for ith_con_pt in arr_control_grids:
        a = (np.random.rand(2) - 0.5) * 0.25
        arr_deformation_contour += create_deformation(arr_contour_pts, ith_con_pt, a)
        arr_deformation_control += create_deformation(arr_control_pts, ith_con_pt, a)

    arr_deformed_contour = arr_contour_pts + arr_deformation_contour
    arr_deformed_controls = arr_control_pts + arr_deformation_control
    return arr_deformed_contour, arr_deformed_controls


def deform_polygon_randomly(arr_contour_pts, arr_control_pts):#, rotation_deg=130):
    """
    Combine the rotation, translation, and random deformation.
    """

    deformed = random_deform_polygon(arr_contour_pts, arr_control_pts)
    #deformed = rotate_polygon(polygon, rotation_deg=rotation_deg)

    return deformed

def show_shapely_polygon(polygon):
    fig, ax = plt.subplots()
    ax.plot(*polygon.exterior.xy, 'b-', label='Polygon')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    plt.show()
    return

# Example usage
polygon = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

if __name__ == "__main__":
    import pyvista as pv
    import numpy as np
    from shapely.geometry import Polygon, Point
    from shapely.geometry import Point, LineString, Polygon
    import matplotlib.pyplot as plt
    import pyvista as pv

    path_starman_vtk = "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__Template__GroundTruth.vtk"
    path_control_points = "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__ControlPoints__GoundTruth.txt"
    # "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen/data/ForInitialization__ControlPoints__FromLongitudinalAtlas.txt"
    arr_contour_pts = np.load(path_control_points)
    mesh = pv.read(path_starman_vtk)
    arr_contour_pts = mesh.points[:, [0,1]]
    deformed_polygon = deform_polygon_randomly(arr_contour_pts,arr_control_pts )
    polygon, _ = Polygon(deformed_polygon)
    show_shapely_polygon(polygon)

    print(deformed_polygon)
