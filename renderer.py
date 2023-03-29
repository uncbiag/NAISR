import numpy as np
import trimesh
import plyfile
from pyrender import IntrinsicsCamera, DirectionalLight, Mesh, Scene, Viewer
import logging

def rotationx(theta):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
        [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotationy(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def load_ply_data(mesh_path):
    mesh = plyfile.PlyData.read(mesh_path)
    mesh_v = []
    mesh_vc = []
    mesh_f = []
    for v in mesh.elements[0]:
        mesh_v.append(np.array((v[0], v[1], v[2])))
        mesh_vc.append(np.array((v[3], v[4], v[5])))
    for f in mesh.elements[1]:
        f = f[0]
        mesh_f.append(np.array([f[0], f[1], f[2]]))
    mesh_v = np.asarray(mesh_v)
    mesh_f = np.asarray(mesh_f)
    mesh_vc = np.asarray(mesh_vc) / 255.0
    return mesh_v, mesh_vc, mesh_f


def render_correspndances(mesh_path):
    # rendering conf
    ambient_light = 0.8
    directional_light = 1.0
    img_res = 512
    cam_f = 500
    cam_c = img_res / 2.0

    scene = Scene(ambient_light=np.array([ambient_light, ambient_light, ambient_light, 1.0]))

    mesh_v, mesh_vc, mesh_f = load_ply_data(mesh_path)
    mesh_ = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f, vertex_colors=mesh_vc)
    points_mesh = Mesh.from_trimesh(mesh_, smooth=True, material=None)
    mesh_node = scene.add(points_mesh)

    cam = IntrinsicsCamera(fx=cam_f, fy=cam_f, cx=cam_c, cy=cam_c)
    cam_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_1 = scene.add(direc_l, pose=np.matmul(rotationy(30), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_2 = scene.add(direc_l, pose=np.matmul(rotationy(-30), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_3 = scene.add(direc_l, pose=np.matmul(rotationy(-180), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=(directional_light-0.5))
    light_node_4 = scene.add(direc_l, pose=np.matmul(rotationy(0), rotationx(-10)))

    ################
    # rendering
    cam_node = scene.add(cam, pose=cam_pose)
    render_flags = {
        'flip_wireframe': False,
        'all_wireframe': False,
        'all_solid': False,
        'shadows': True,
        'vertex_normals': False,
        'face_normals': False,
        'cull_faces': True,
        'point_size': 1.0,
    }
    viewer_flags = {
        'mouse_pressed': False,
        'rotate': False,
        'rotate_rate': np.pi / 6.0,
        'rotate_axis': np.array([0.0, 1.0, 0.0]),
        'view_center': np.array([0.0, 0.0, 0.0]),
        'record': False,
        'use_raymond_lighting': False,
        'use_direct_lighting': False,
        'lighting_intensity': 3.0,
        'use_perspective_cam': True,
        'window_title': 'DIT',
        'refresh_rate': 25.0,
        'fullscreen': False,
        'show_world_axis': False,
        'show_mesh_axes': False,
        'caption': None,
        'save_one_frame': False,
    }
    v = Viewer(scene, viewport_size=(512, 512), render_flags=render_flags,
                    viewer_flags=viewer_flags, run_in_thread=False)
    v.close()




def save_to_ply(decoder, lat_vec, attr, ply_filename_out, device):

    import pyvista as pv
    shape_pred = trimesh.load(ply_filename_out)
    verts = np.array(shape_pred.vertices)
    faces = shape_pred.faces
    num_faces = len(faces)
    lat_vec_subset = lat_vec[0, [0], :].repeat(verts.shape[0], 1).to(device)[None, :, :]
    verts_warped = decoder(verts, attr, lat_vec_subset)['vec_fields']['overall'].squeeze().detach().cpu()

    num_verts = verts_warped.shape[0]

    # store canonical coordinates as rgb color (in float format)
    verts_color = 255 * (0.5 + 0.5 * verts_warped)
    verts_color = verts_color.astype(np.uint8)

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
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str)
    args = parser.parse_args()
    main(args.input_path)
