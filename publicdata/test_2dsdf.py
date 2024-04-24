import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import pyvista as pv






def show_shapely_polygon(polygo):
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
    #show_shapely_polygon(polygon)
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
    x = (np.random.rand(number_of_points) - 0.5) * 4
    y = (np.random.rand(number_of_points) - 0.5) * 4
    coords = np.concatenate((x[:, None], y[:, None]), axis=-1)
    return coords



def calculate_2dsdf_from_polygon(arr_contour_pts, number_of_points=500000):
    query_points = generate_off_surface_points(number_of_points=number_of_points)
    arr_sdf = extract_2d_sdf_from_contour(query_points, arr_contour_pts)
    npz_sdf = np.concatenate((query_points, arr_sdf), axis=-1)
    return npz_sdf




path_starman_vtk = "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__Template__GroundTruth.vtk"
path_control_points = "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__ControlPoints__GoundTruth.txt"
   # "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen/data/ForInitialization__ControlPoints__FromLongitudinalAtlas.txt"

mesh = pv.read(path_starman_vtk)


# Define the vertices of the polygon
polygon_vertices = mesh.points #[(0, 0), (2, 0), (2, 2), (1, 3), (0, 2)]

polygon = Polygon(polygon_vertices)
show_shapely_polygon(polygon)
# Define a point for which you want to calculate the signed distance
query_point = Point(0, 0)
query_point2 = Point(polygon_vertices[0])

# Calculate the signed distance
signed_dist = signed_distance_polygon(polygon, query_point)
print("Signed distance:", signed_dist)

