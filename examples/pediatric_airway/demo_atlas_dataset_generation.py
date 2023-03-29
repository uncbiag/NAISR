import os

import pyvista as pv
import numpy as np
import pandas as pd
import skimage.measure as measure
import pickle

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

def register_landmarks(p1_current, p2_template):

    p1_current = p1_current.squeeze()
    p2_template = p2_template.squeeze()
    #Take transpose as columns should be the points
    p1 = p1_current.transpose()
    p2 = p2_template.transpose()

    #Calculate centroids
    p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 1).reshape((-1,1))

    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())

    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

    #Calculate rotation matrix
    R = np.matmul(V_t.transpose(),U.transpose())
    print(np.linalg.det(R))
    #assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    #Calculate translation matrix
    T = p2_c - np.matmul(R,p1_c)

    #Check result
    result = T + np.matmul(R,p1)
    if np.allclose(result,p2):
        print("transformation is correct!")
    else:
        print("transformation is wrong...")
    return  R, T

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

    # age_start_column, age_end_column = "DOB", "DOV"

    df = pd.read_excel(metadata_fpath)
    ids, ages_month, weights, heights, sexes = [list(df[elm]) for elm in [id_column, age_column, weight_column, height_column, sex_column]]
    ids = cast_list_dtype(ids, str)
    ages_month = cast_list_dtype(ages_month, float)
    weights = cast_list_dtype(weights, float)
    heights = cast_list_to_numpy(heights, float)
    sexes = cast_sex_to_numpy(sexes)

    # ids, ages_start, ages_end, weights = [list(df[elm]) for elm in [id_column, age_start_column, age_end_column, weight_column]]
    # ages_month_2 = [(e - s).total_seconds() / 2628000.0 for (s, e) in zip(ages_start, ages_end)]

    data = {id: {"age": ages_month[i], "weight": weights[i], 'height': heights[i], 'sex': sexes[i]} for i, id in enumerate(ids)}
    return data


def filter_landmarks(landmarks, desired):
    return {name: landmark for name, landmark in landmarks.items() if name in desired}





def read_origin_spacing(path):
    import SimpleITK as sitk
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


def landmark_correspondence(src): #, target_landmark_path, target_segmentation_path):

    target_segmentation_path = '/home/jyn/pediatric_airway_atlas/data_extraction/target/1035_UNITIZED_SEGMENTATION.mha'
    target_landmark_path = '/home/jyn/pediatric_airway_atlas/data_extraction/target/1035_LANDMARKS.p3'

    arr_target_seg, target_Origin, target_Spacing = read_origin_spacing(target_segmentation_path)
    target_landmarks = pickle_load_object(target_landmark_path)['continuous']


    #for ikey in src.keys():
    #    src[ikey] = reverse_coords([src[ikey]])
    #    src[ikey] = src[ikey] * source_Spacing

    for ikey in target_landmarks.keys():
        target_landmarks[ikey] = reverse_coords([target_landmarks[ikey]])
        target_landmarks[ikey] = target_landmarks[ikey] * target_Spacing

    list_tgt_ldm = []
    list_src_ldm = []

    for name, landmark in src.items():
        if name in target_landmarks.keys():
            list_src_ldm.append(landmark)
            list_tgt_ldm.append(target_landmarks[name])
        else:
            print("Failed to match landmark: " + str(name))
    return np.array(list_src_ldm), np.array(list_tgt_ldm)

def read_and_register_landmarks(path_landmarks, path_segmentation):

    landmarks = pickle_load_object(path_landmarks)['continuous']
    target_landmark = []
    for i_target_ldm in landmarks.keys():
        target_landmark.append(i_target_ldm)

    # filter and reverse coord
    landmarks = filter_landmarks(landmarks, target_landmark)
    arr, Origin, Spacing = read_origin_spacing(path_segmentation)
    for ikey in landmarks:
        landmarks[ikey] = reverse_coords([landmarks[ikey]])
        landmarks[ikey] = landmarks[ikey] * Spacing

    # register
    arr_src_landmarks, arr_tgt_landmarks = landmark_correspondence(landmarks)
    R, T = register_landmarks(arr_src_landmarks, arr_tgt_landmarks)

    for name, value in landmarks.items():
        landmarks[name] = (T.T + np.matmul(landmarks[name], R.T))[0]
    return landmarks

def get_canonical_2D_CSA(source_subj,
                         rootdir,
                         savedir,
                         list_2dcsa_dataset = [],
                         VIS=False,
                         template_subj='1035',
                         metadatapath="/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/FilteredControlBlindingLogUniqueScanFiltered_19Sep2022.xls",
                         savepath_dataset='./2dcsa.csv'):

    '''

    :param path_centerline_src:
    :param dir_2dcsa_src:
    :param path_centerline_templ:
    :param dir_2dcsa_templ:
    :return:
    '''
    path_centerline_src = os.path.join(rootdir, f"aligned_interp_ctl/{source_subj}_CENTERLINE.p3")
    path_landmark = os.path.join(rootdir, f"transformed_landmarks/{source_subj}_LANDMARKS.p3")
    path_segmentations = os.path.join(rootdir, f"unitized_segmentations/{source_subj}_UNITIZED_SEGMENTATION.mha")

    # get path of source and target shapes
    rootdir_2dcsa_src = os.path.join(rootdir_2dcsa, source_subj)
    rootdir_2dcsa_templ = os.path.join(rootdir_2dcsa, template_subj)

    landmarks = read_and_register_landmarks(path_landmark, path_segmentations)


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

    # get idx of ldm on centerline

    list_idm_on_ctl = {}
    for name, value in landmarks.items():
        dist = np.linalg.norm(np.array(arr_centerline_source) - value, axis=-1)
        list_idm_on_ctl[name] = {'name': name,
                                'depth':  np.array(arr_depth_source)[dist == dist.min()],
                                'idx': np.array(arr_idx_source)[dist == dist.min()],
                                'ctl': np.array(arr_centerline_source)[dist == dist.min()],
                                'norm': np.array(arr_normal_source)[dist == dist.min()],}
    print(list_idm_on_ctl.keys())


    if 'carina' in list(list_idm_on_ctl.keys()) and 'tvc' in  list(list_idm_on_ctl.keys()):
        list_idm_on_ctl['midtrachea'] = {'name': 'midtrachea',
                                'idx': (list_idm_on_ctl['carina']['idx']+list_idm_on_ctl['tvc']['idx']) //2, }


    '''
    # read template centerline
    dict_centerline_template = pickle_load_object(path_centerline_templ)
    arr_centerline_template = []
    arr_normal_template = []
    for ith in arr_idx_source:
        idict = dict_centerline_template[ith]
        for _, data in idict.items():
            arr_centerline_template.append(data[1])
            arr_normal_template.append(data[2])
    '''


    '''
    # read 2D slices of template shape
    airway_pc_template = []
    list_pts_template = []
    for i_slice in arr_idx_source:
        i_path = os.path.join(rootdir_2dcsa_templ, template_subj + '_slice_' + str(i_slice) + '.npy')
        points = np.load(i_path)
        airway_pc_template += points.tolist()
        list_pts_template.append(points)
    '''

    # read 2D slices of source shape
    list_pts_source = []
    airway_pc_source = []
    list_normals_source = []
    for i_slice in arr_idx_source:
        i_path = os.path.join(rootdir_2dcsa_src, str(source_subj) + '_slice_' + str(i_slice) + '.npy')
        points_normals = np.load(i_path)
        points = points_normals[:, [0, 1, 2]]
        normals = points_normals[:, [3, 4, 5]]
        airway_pc_source += points.tolist()
        list_pts_source.append(points)
        list_normals_source.append(normals)

    # align centerline
    #print(len(arr_centerline_source))
    #sample_idx = (np.arange(len(arr_centerline_source) // 30 )  )* 30
    #print(sample_idx)

    #R, T = register_landmarks(np.array(arr_centerline_source), np.array(arr_centerline_template))
    arr_aligned_centerline = arr_centerline_source #apply_registration_to_pts(np.array(arr_centerline_source).T, R, T).T
    arr_aligned_normal = arr_normal_source# np.matmul(R, np.array(arr_normal_source).T).T


    # registration
    list_pts_aligned = []
    for i in arr_idx_source:
        pts_source = list_pts_source[i]
        pts_aligned = pts_source #apply_registration_to_pts(pts_source.T, R, T).T
        list_pts_aligned.append(pts_aligned)


    # register to 2D plane
    list_aligned_2D_CSA = []
    arr_aligned_2D_CSA = []
    for ith in range(len(list_pts_aligned)):
        if len(list_pts_aligned[ith]) > 0:
            # get the plane rotate
            rmat = rotation_of_a_plane(arr_aligned_normal[ith], np.array([0, 0, 1]))
            # rotate
            newpoint = np.matmul(rmat, np.array(list_pts_aligned[ith] - arr_aligned_centerline[ith]).T).T
            newpoint[:, -1] = 0
            newnormal = np.matmul(rmat, np.array(list_normals_source[ith] ).T).T * (-1)
            newpoint = np.concatenate((newpoint, newnormal), axis=-1)
            list_aligned_2D_CSA.append(newpoint)
            arr_aligned_2D_CSA += newpoint.tolist()
            #p.add_mesh(newpoint, color='pink')
            #p.add_mesh(pv.PolyData(list_pts_1[ith]), color='b')
            #rmat = rotation_of_a_plane(arr_normal_1[ith], np.array([0, 0, 1]))
            #newpoint = np.matmul(rmat, np.array(list_pts_1[ith] - arr_centerline_1[ith]).T).T
            #p.add_mesh(newpoint, color='lightblue')
        else:
            list_aligned_2D_CSA.append([])


    # save results
    savedir_2dcsa = os.path.join(savedir, '2dcsa')
    if not os.path.exists(savedir_2dcsa):
        os.mkdir(savedir_2dcsa)
    savedir_2dcsa_subj = os.path.join(savedir_2dcsa, source_subj)
    if not os.path.exists(savedir_2dcsa_subj):
        os.mkdir(savedir_2dcsa_subj)

    savedir_aligned_ctl = os.path.join(savedir, 'aligned_interp_ctl')
    meta_data = parse_4d_metadata_for_atlas(metadatapath)
    #list_2dcsa_dataset = []
    for idx, ith_depth, ith_2dcsa in zip(arr_idx_source, arr_depth_source, list_aligned_2D_CSA):
        if len(ith_2dcsa) >0:
            current_savepath = os.path.join(savedir_2dcsa_subj, str(idx) + '.npy')
            np.save(current_savepath, ith_2dcsa)
            list_2dcsa_dataset.append({
                             'id': source_subj,
                             'age': meta_data[source_subj]['age'],
                             'weight': meta_data[source_subj]['weight'],
                             'height': meta_data[source_subj]['height'],
                             'sex': meta_data[source_subj]['sex'],
                             'idx': idx,
                             'depth': ith_depth,
                             '2dcsa': current_savepath,
                             'ctl': os.path.join(savedir_aligned_ctl, f"{source_subj}_CENTERLINE.p3")}
            )

    pd.DataFrame.from_records(list_2dcsa_dataset).to_csv(savepath_dataset)


    if VIS:

        '''
        plot centerline
        '''


        max_p = np.max(np.abs(np.array(arr_aligned_2D_CSA)))
        centers = []
        for i in range(15):
            for j in range(15):
                centers.append(np.array([-i, j, -0.5]) * 2)
        centers = np.array(centers).astype('float')

        pv.start_xvfb()
        pv.global_theme.background = 'white'
        p = pv.Plotter(lighting="light_kit", off_screen=True, window_size=[1024, 1024])
        pv.global_theme.background = 'white'
        pv.global_theme.interactive = True


        for ith in range(len(list_aligned_2D_CSA)):
            if len(list_aligned_2D_CSA[ith] ) > 0:
                ith_2dcsa = list_aligned_2D_CSA[ith][:, [0, 1, 2]] / max_p  + centers[ith]
                #ith_2dnormal = list_normals_source[ith][:, [0, 1]]
                #ith_2dnormal = np.concatenate((ith_2dnormal, np.zeros((ith_2dnormal.shape[0], 1))), axis=-1)

                p.add_mesh(ith_2dcsa, color='b', point_size=3)
                #p.add_arrows(ith_2dcsa, direction=ith_2dnormal, color='b', mag=0.5)

        p.view_vector((5.0, 2, 3))
        p.add_floor('-z', lighting=True, color='grey', pad=1.0)
        #p.enable_shadows()
        p.screenshot(os.path.join(savedir_2dcsa_subj, 'screenshot.png'))
        p.export_html(os.path.join(savedir_2dcsa_subj, 'vis.html'), backend='panel')
        p.close()



        '''
        plot landmarks
        '''

        landmarks_names = ['choana', 'baseoftongue', 'epiglottistip', 'tvc', 'subglottis', 'midtrachea', 'trachea', 'carina']
        valid_landmarks_names = []
        for i in landmarks_names:
            if i in list(list_idm_on_ctl.keys()):
                valid_landmarks_names.append(i)
        print(valid_landmarks_names)
        pv.start_xvfb()
        pv.global_theme.background = 'white'
        p = pv.Plotter(lighting="light_kit", shape=(1, len(valid_landmarks_names)), off_screen=True,
                       window_size=(1024, 512), border=False)
        pv.global_theme.background = 'white'
        pv.global_theme.interactive = True

        for ith in range(len(valid_landmarks_names)):
            p.subplot(0, ith)
            idx = list_idm_on_ctl[valid_landmarks_names[ith]]['idx']
            name = list_idm_on_ctl[valid_landmarks_names[ith]]['name']
            ith_2dcsa = list_aligned_2D_CSA[idx[0]][:, [0, 1, 2]] / max_p
            p.add_mesh(pv.PolyData(ith_2dcsa), color='b', point_size=3)
            p.add_text(str(name), color='black')

        p.link_views()
        p.view_xy()
        p.screenshot(os.path.join(savedir_2dcsa_subj, 'screenshot_ldm.png'))
        p.export_html(os.path.join(savedir_2dcsa_subj, 'vis_lm.html'), backend='panel')
        p.close()


    '''ind'''

    savedir_ldm_vis = os.path.join(savedir, 'landmarks_2dvis')
    if not os.path.exists(savedir_ldm_vis):
        os.mkdir(savedir_ldm_vis)
    savedir_ldm_vis_subj = os.path.join(savedir_ldm_vis, source_subj)
    if not os.path.exists(savedir_ldm_vis_subj):
        os.mkdir(savedir_ldm_vis_subj)

    landmarks_names = ['choana', 'baseoftongue', 'epiglottistip', 'tvc', 'subglottis', 'midtrachea', 'trachea',
                       'carina']
    valid_landmarks_names = []
    for i in landmarks_names:
        if i in list(list_idm_on_ctl.keys()):
            valid_landmarks_names.append(i)
    for ith in range(len(valid_landmarks_names)):

        pv.start_xvfb()
        pv.global_theme.background = 'white'

        p = pv.Plotter(lighting="light_kit", shape=(1, 1), off_screen=True, window_size=(512, 512), border=False)
        pv.global_theme.background = 'white'
        pv.global_theme.interactive = True

        p.subplot(0, 0)
        idx = list_idm_on_ctl[valid_landmarks_names[ith]]['idx']
        name = list_idm_on_ctl[valid_landmarks_names[ith]]['name']
        ith_2dcsa = list_aligned_2D_CSA[idx[0]][:, [0, 1, 2]] / max_p
        #poly = pv.Spline(ith_2dcsa )
        p.add_mesh(pv.PolyData(ith_2dcsa), color='b', point_size=10)
        #p.add_text(str(name), color='black')

        p.link_views()
        p.view_xy()
        p.screenshot(os.path.join(savedir_ldm_vis_subj, name + '.png'))
        p.export_html(os.path.join(savedir_ldm_vis_subj, name + '.html'), backend='panel')
        p.close()

    return list_2dcsa_dataset




#source_subj = '1030'

rootdir = '/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/'

rootdir_2dcsa = '/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/new_cross_sections_contours/'
savedir = '/playpen-raid/jyn/NAISR/NAISR/examples/pediatric_airway/'

list_subjs = os.listdir(rootdir_2dcsa)
print("there are " + str(len(list_subjs)) + ' cases in the atlas.')
list_2dcsa_dataset =[]
for source_subj in list_subjs:
    list_2dcsa_dataset = get_canonical_2D_CSA(source_subj,
                     rootdir,
                     savedir,
                     list_2dcsa_dataset = list_2dcsa_dataset,
                     VIS=True,
                     template_subj='1035',
                     metadatapath="/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/FilteredControlBlindingLogUniqueScanFiltered_19Sep2022.xls",
                     savepath_dataset='./2dcsa.csv')
    print(source_subj)