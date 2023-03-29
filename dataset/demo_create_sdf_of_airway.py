import torch
import numpy as np
import SimpleITK as sitk
import trimesh

'''
def calculate_sdf_from_mesh(surf_mesh, savepath):

    points, sdf = sample_sdf_near_surface(surf_mesh, number_of_points=40000)
    npz_sdf = np.concatenate((points, sdf[:, None]), axis=-1)
    return npz_sdf

'''

import os

import pyvista as pv
import numpy as np
import pandas as pd
import skimage.measure as measure
import pickle
import pickle as pk

import torch


def unitcross(a, b):
    c = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    return c
def rotation_of_a_plane(M, N):
    costheta = np.matmul(M, N) / (np.linalg.norm(M) * np.linalg.norm(N))

    axis = unitcross(M, N)

    c = costheta
    s = np.sqrt(1 - c * c)
    C = 1 - c
    x, y, z = axis[0], axis[1], axis[2]
    rmat = np.array([[x * x * C + c,    x * y * C - z * s,  x * z * C + y * s],
                    [y * x * C + z * s,  y * y * C + c,    y * z * C - x * s],
                    [z * x * C - y * s, z * y * C + x * s, z * z * C + c]])
    return rmat


#path = '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/1030_UNITIZED_SEGMENTATION.mha'

#arr, Origin, Spacing = read_origin_spacing(path)
#surface = surface_construction(arr, Origin, Spacing)
    # register


def read_origin_spacing(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    Origin = np.array(img.GetOrigin())
    Spacing = np.array(img.GetSpacing())
    return arr, Origin, Spacing


def reverse_coords(arr):
    rev = []
    for x, y, z in arr:
        rev.append([z, y, x])
    return np.array(rev)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pk.load(f)

def landmark_correspondence(src): #, target_landmark_path, target_segmentation_path):

    target_segmentation_path = '/Users/jyn/jyn/research/projects/NAISR/data2viz/dataset/unitized_segmentations/1035_UNITIZED_SEGMENTATION.mha'   #'/home/jyn/pediatric_airway_atlas/data_extraction/target/1035_UNITIZED_SEGMENTATION.mha' #
    target_landmark_path = '/Users/jyn/jyn/research/projects/NAISR/data2viz/dataset/transformed_landmarks/1035_LANDMARKS.p3' # '/home/jyn/pediatric_airway_atlas/data_extraction/target/1035_LANDMARKS.p3'#  #

    arr_target_seg, target_Origin, target_Spacing = read_origin_spacing(target_segmentation_path)
    target_landmarks = load_pickle(target_landmark_path)['continuous']


    #for ikey in src.keys():
    #    src[ikey] = reverse_coords([src[ikey]])
    #    src[ikey] = src[ikey] * source_Spacing

    new_target_landmarks = {}
    for ikey in target_landmarks.keys():
        if ikey != 'carina':
            new_target_landmarks[ikey] = reverse_coords([target_landmarks[ikey]])
            new_target_landmarks[ikey] = new_target_landmarks[ikey]* target_Spacing


    list_tgt_ldm = []
    list_src_ldm = []

    for name, landmark in src.items():
        if name in new_target_landmarks.keys():
            list_src_ldm.append(landmark)
            list_tgt_ldm.append(new_target_landmarks[name])
        else:
            #continue
            print("Failed to match landmark: " + str(name))
    return np.array(list_src_ldm), np.array(list_tgt_ldm)

def get_registration_of_landmarks(path_landmark, path_seg):
    landmarks = load_pickle(path_landmark)['continuous']
    arr, Origin, Spacing = read_origin_spacing(path_seg)
    for ikey in landmarks:
        landmarks[ikey] = reverse_coords([landmarks[ikey]])
        landmarks[ikey] = landmarks[ikey] * Spacing

    arr_src_landmarks, arr_tgt_landmarks = landmark_correspondence(landmarks)
    R, T = register_landmarks(arr_src_landmarks, arr_tgt_landmarks)

    arr_aligned_center = apply_registration_to_pts(np.array(landmarks['tvc']).T, R, np.zeros_like(T)).T
    return R, arr_aligned_center.T* (-1), arr_aligned_center


def register_landmarks(p1_current, p2_template):
    assert p1_current.shape == p2_template.shape


    p1_current = p1_current.squeeze()
    p2_template = p2_template.squeeze()
    n, dim = p1_current.shape
    #Take transpose as columns should be the points
    p1 = p1_current#.transpose()
    p2 = p2_template#.transpose()

    #Calculate centroids
    p1_c = np.mean(p1, axis = 0).reshape((1, -1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 0).reshape((1, -1))

    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    #Calculate covariance matrix
    C = np.dot(np.transpose(q1), q2) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(p1, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = p2_c- p1_c.dot(c*R)


    #Check result
    print("R =\n", R)
    print("c =", c)
    print("t =\n", t)
    print("Check:  a1*cR + t = a2  is", np.allclose(p1.dot(c * R) + t, p2))
    err = ((p1.dot(c * R) + t - p2) ** 2).sum()
    print("Residual error", err)

    return R.T, t



def apply_registration_to_pts(p1, R, T):
    result = T + np.matmul(R, p1)
    return result




def pickle_load_object(f_path):
    with open(f_path, "rb") as f_in:
        obj = pickle.load(f_in)
    return obj

def parse_4d_metadata_for_atlas(metadata_fpath):
    import pandas
    def cast_list_dtype(lst, dtype):
        return [dtype(elm) for elm in lst]
    def cast_list_to_numpy(lst, dtype):
        data_list = []
        for elm in lst:
            try:
                data_list.append(dtype(elm))
            except:
                data_list.append(np.nan)
        return data_list

    def cast_sex_to_numpy(lst):
        data_list = []
        for elm in lst:
            if not isinstance(elm, str):
                data_list.append(np.nan)
            elif 'm' in elm or ('M' in elm):
                data_list.append(1)
            elif 'f' in elm or ('F' in elm):
                data_list.append(0)
        return data_list

    id_column = "scan #"
    age_column = "age in mon"
    weight_column = "weight in Kg"
    height_column = 'Height in cm'
    sex_column = 'sex'
    pid_column = 'UNC  MR#'

    # age_start_column, age_end_column = "DOB", "DOV"

    df = pd.read_excel(metadata_fpath)
    ids, ages_month, weights, heights, sexes,  pids = [list(df[elm]) for elm in [id_column, age_column, weight_column, height_column, sex_column, pid_column]]
    ids = cast_list_dtype(ids, str)
    ages_month = cast_list_dtype(ages_month, float)
    weights = cast_list_dtype(weights, float)
    heights = cast_list_to_numpy(heights, float)
    sexes = cast_sex_to_numpy(sexes)
    pids = cast_list_dtype(pids, str)

    # ids, ages_start, ages_end, weights = [list(df[elm]) for elm in [id_column, age_start_column, age_end_column, weight_column]]
    # ages_month_2 = [(e - s).total_seconds() / 2628000.0 for (s, e) in zip(ages_start, ages_end)]

    data = {id: {"age": ages_month[i], "weight": weights[i], 'height': heights[i], 'sex': sexes[i], 'PID': pids[i]} for i, id in enumerate(ids)}
    return data


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
    airway_surface =  current_airway.extract_surface().smooth(n_iter=500)

    surf_airway = trimesh.Trimesh(vertices=airway_surface.points, faces=faces, )
    return surf_airway



def apply_registration_to_surface(surf, R, T):
    new_verts = T + np.matmul(R, np.array(surf.vertices).T)
    new_surf = trimesh.Trimesh(vertices=new_verts.T, faces=surf.faces)
    return new_surf


def transform_trimesh_to_pv(surf_trimesh):
    surf_pv = pv.PolyData()
    surf_pv.points = surf_trimesh.vertices
    surf_pv.faces = np.hstack((np.ones((surf_trimesh.faces.shape[0], 1)) * 3, surf_trimesh.faces)).ravel().astype('int')
    return surf_pv

def get_canonical_3Dshape(source_subj,
                         rootdir_centerline,
                         rootdir_landmark,
                         rootdir_seg,
                         savedir,
                         list_3dshape_dataset = [],
                         VIS=False,
                         template_subj='1035',
                         metadatapath="/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/FilteredControlBlindingLogUniqueScanFiltered_19Sep2022.xls",
                         savepath_dataset='./3dshape.csv'):

    '''

    :param path_centerline_src:
    :param dir_2dcsa_src:
    :param path_centerline_templ:
    :param dir_2dcsa_templ:
    :return:
    '''

    # get path of source and target shapes
    #rootdir_2dcsa_src = os.path.join(rootdir_2dcsa, source_subj)
    #rootdir_2dcsa_templ = os.path.join(rootdir_2dcsa, template_subj)

    path_centerline_src = os.path.join(rootdir_centerline, str(source_subj) + '_CENTERLINE.p3')
    #path_centerline_templ = os.path.join(rootdir_centerline, str(template_subj) + '_CENTERLINE.p3')
    path_landmark = os.path.join(rootdir_landmark, str(source_subj) + '_LANDMARKS.p3')


    # read target centerline

    dict_centerline_source = pickle_load_object(path_centerline_src)
    arr_centerline_source = []
    arr_normal_source = []
    arr_idx_source = []
    arr_depth_source = []
    for idict in dict_centerline_source:
        for ith, data in idict.items():
            arr_depth_source.append(data[0])
            arr_idx_source.append(ith)
            arr_centerline_source.append(data[1])
            arr_normal_source.append(data[2])

    unitized_segmentation_path = os.path.join(rootdir_seg, f"{source_subj}_UNITIZED_SEGMENTATION.mha")
    surf_airway = surface_construction(unitized_segmentation_path)



    # save results
    savedir_3dshape = os.path.join(savedir, '3dshape')
    if not os.path.exists(savedir_3dshape):
        os.mkdir(savedir_3dshape)
    savedir_3dshape_subj = os.path.join(savedir_3dshape, source_subj)
    if not os.path.exists(savedir_3dshape_subj):
        os.mkdir(savedir_3dshape_subj)
    savedir_3dshape_stl= os.path.join(savedir_3dshape_subj, source_subj + '.stl')
    savedir_3dshape_sdf = os.path.join(savedir_3dshape_subj, source_subj + '.npy')
    savedir_3dshape_stl_npy = os.path.join(savedir_3dshape_subj, source_subj + '_stl.npy')

    savedir_aligned_ctl = os.path.join(savedir, 'aligned_interp_ctl')
    meta_data = parse_4d_metadata_for_atlas(metadatapath)


    list_3dshape_dataset.append({
                     'id': source_subj,
                    'PID': meta_data[source_subj]['PID'],
                     'age': meta_data[source_subj]['age'],
                     'weight': meta_data[source_subj]['weight'],
                     'height': meta_data[source_subj]['height'],
                     'sex': meta_data[source_subj]['sex'],
                     '3dsdf': savedir_3dshape_sdf,
                     '3dshape': savedir_3dshape_stl,
                     '3dshape_npy': savedir_3dshape_stl_npy,
                     'ctl': os.path.join(savedir_aligned_ctl, f"{source_subj}_CENTERLINE.p3"),
                     #'tvc_x': arr_aligned_center[0][0],
    #'tvc_y': arr_aligned_center[0][1],
    #'tvc_z': arr_aligned_center[0][2],
    'length': len(arr_centerline_source)}
    )

    #pd.DataFrame.from_records(list_3dshape_dataset).to_csv(savepath_dataset)

    '''
    if VIS:
        max_p = np.max(np.abs(np.array(surf_airway.vertices)))

        #pv.start_xvfb()
        pv.global_theme.background = 'white'
        p = pv.Plotter(lighting="light_kit", off_screen=True, window_size=[1024, 1024])
        pv.global_theme.background = 'white'
        pv.global_theme.interactive = True


        #for ith in range(len(list_aligned_2D_CSA)):
        #    if len(list_aligned_2D_CSA[ith] ) > 0:
        #        ith_2dcsa = list_aligned_2D_CSA[ith][:, [0, 1, 2]] / max_p  + centers[ith]
        #        ith_2dnormal = list_normals_source[ith][:, [0, 1]]
        #       ith_2dnormal = np.concatenate((ith_2dnormal, np.zeros((ith_2dnormal.shape[0], 1))), axis=-1)

        pv_aligned_surf = transform_trimesh_to_pv(aligned_surf)
        p.add_mesh(pv_aligned_surf, color='grey', point_size=3,)
        p.add_arrows(pv_aligned_surf.points, direction=pv_aligned_surf.point_normals, color='yellow', mag=0.3)

        p.view_vector((5.0, 2, 3))
        p.add_floor('-z', lighting=True, color='grey', pad=1.0)
        #p.enable_shadows()
        p.screenshot(os.path.join(savedir_3dshape_subj, 'screenshot.png'))
        #p.export_obj(os.path.join(savedir_3dshape_subj, 'vis.obj'))
        p.close()
    '''
    return list_3dshape_dataset




#source_subj = '1030'
#rootdir_centerline = '/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/aligned_interp_ctl/'
#rootdir_2dcsa = '/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/new_cross_sections_contours/'
savedir =  '/playpen-raid/jyn/NAISR/NAISR/examples/pediatric_airway/' #'/Users/jyn/jyn/research/projects/NAISR/data2viz/dataset/' #

rootpath_atlas = '/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/' #'/Users/jyn/jyn/research/projects/NAISR/data2viz/dataset/' # #'/playpen-raid/jyn/pediatric_atlas/' #  #
rootdir_seg = os.path.join(rootpath_atlas, "unitized_segmentations")
rootdir_landmark = os.path.join(rootpath_atlas, "transformed_landmarks")
rootdir_centerline = os.path.join(rootpath_atlas, "aligned_interp_ctl")
rootdir_2dcsa = os.path.join(rootpath_atlas, "pediatric_airway", "2dcsa")

list_subjs = sorted(os.listdir(rootdir_centerline))
print("there are " + str(len(list_subjs)) + ' cases in the atlas.')
list_3dshape_dataset =[]

for source_subj in list_subjs:

    source_subj = source_subj[0:4]
    #if not os.path.exists(os.path.join(savedir, '3dshape', str(source_subj), str(source_subj) + '.npy')):

    list_3dshape_dataset = get_canonical_3Dshape(
                     source_subj,
                     rootdir_centerline,
                    rootdir_landmark,
                     rootdir_seg,
                     savedir,
                     list_3dshape_dataset= list_3dshape_dataset,
                     VIS=False,
                     template_subj='1035',
                     metadatapath= "/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/FilteredControlBlindingLogUniqueScanFiltered_19Sep2022.xls", # # "/Users/jyn/jyn/research/projects/NAISR/data2viz/dataset/FilteredControlBlindingLogUniqueScanFiltered_19Sep2022.xls",                    metadatapath="/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/FilteredControlBlindingLogUniqueScanFiltered_19Sep2022.xls", # #
                     savepath_dataset='./3dshape.csv')

    #os.system('python create_sdf_from_mesh.py --subj ' + str(source_subj))
    print(source_subj)
