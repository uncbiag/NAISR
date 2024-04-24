import os
import numpy as np
import pandas as pd
import math
import pyvista as pv
import trimesh
from utils_3d import *
import skimage.measure as measure
import datetime
# def extract_subj_list(root_dataset):
#     #root_dataset = "/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data" #"/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data"
#     list_scans = os.listdir(root_dataset)
#     list_subjects = []
#     dict_subj_scans = {}
#     for ith_scan in list_scans:
#         if '.vtk' in ith_scan and 'sub' in ith_scan:
#             ith_subj = int(ith_scan.split('_')[0][-4::])
#             list_subjects.append(ith_subj)
#             # attach scans to subjects
#             if ith_subj not in list(dict_subj_scans.keys()):
#                 dict_subj_scans[ith_subj] = []
#             dict_subj_scans[ith_subj].append(ith_scan)
#
#     list_subjects = np.unique(np.array(list_subjects))
#     return list_subjects, dict_subj_scans


def extract_subj_list(root_dataset):
    #root_dataset = "/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data" #"/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data"
    list_all_subjects = os.listdir(root_dataset)
    list_of_subjects = []
    dict_subj_scans = {}
    for ith_scan in list_all_subjects:
        # current patient
        if os.path.isdir(os.path.join(root_dataset, ith_scan)):
            ith_subj = int(ith_scan.split('_')[-1])
            # current folder
            current_scan_folder = os.path.join(root_dataset, ith_scan, "Hippocampal_Mask")
            # attach scans to subjects
            dict_subj_scans[ith_scan] = os.listdir(current_scan_folder)
            list_of_subjects.append(ith_scan)
    return list_of_subjects, dict_subj_scans


def transform_group_to_num(str_class):
    if str_class == 'CN':
        return 0
    elif str_class == 'MCI':
        return 1
    elif str_class == 'AD':
        return 2
def extract_patient_convariates(root_adni_dataset):
    #root_dataset = "/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data"  # "/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data"
    list_subjects, dict_subj_scans = extract_subj_list(root_adni_dataset)
    pd_demmog = pd.read_csv(ROOTPATH_DEMMOG)
    pd_dataset_info = pd.read_csv(ROOTPATH_DATASET_INFO)
    pd_dataset_info['Study Date'] = pd.to_datetime(pd_dataset_info['Study Date'], yearfirst=True)
    list_data = []

    for ith_subject in list_subjects:
        # print("processing " + str(ith_subject))
        RID = int(ith_subject.split('_')[-1])
        df_current_subj = pd_demmog[pd_demmog['RID'] == RID]
        list_edu_level = []
        # get ages
        for i in range(len(df_current_subj)):
            # get education
            list_edu_level.append(df_current_subj.iloc[i]["PTEDUCAT"])

        edu_level = np.array(list_edu_level).max()

        list_current_scans = dict_subj_scans[ith_subject]


        current_datasetinfo = pd_dataset_info[pd_dataset_info['Subject ID']==ith_subject]
        # id,PID,cov_1,cov_2,cov_3,cov_4,2dsdf,2dshape
        for ith_scan in list_current_scans:
            scandate = ith_scan.split("_")[0]
            current_folder = os.path.join(root_adni_dataset, ith_subject, "Hippocampal_Mask", ith_scan)
            current_scanfile_folder = os.path.join(current_folder, os.listdir(current_folder)[0])
            current_scanfile = os.listdir(current_scanfile_folder)[0]
            current_path = os.path.join(current_scanfile_folder, current_scanfile)


            time_diff = (current_datasetinfo['Study Date'] - pd.to_datetime(scandate)) / pd.Timedelta(1, 'D')
            current_age = float(current_datasetinfo[time_diff.abs()<=1]['Age'].values[0])
            current_sex = float(current_datasetinfo[time_diff.abs() <= 1]['Sex'].values[0]=='M')
            #except:
            #    print('a')
            #try:
            current_group = current_datasetinfo[time_diff.abs()<=1]['Research Group'].values[0]
            current_group = float(transform_group_to_num(current_group))
            #except:
            #    print('a')

            #datetime.strftime(format)
            list_data.append({'ID': ith_subject,
                              'age': current_age,
                              'sex':current_sex,
                              "edu": edu_level,
                              'group': current_group,
                              "scan": current_scanfile,
                              "path": current_path
                              })

    print("Done")
    return list_subjects, dict_subj_scans, list_data




# def pair_covariates_with_shapes(list_subjects, dict_subj_scans, list_subjects_data):
#
#     dict_subjects_data = pd.DataFrame.from_records(list_subjects_data)
#     list_scan_data = []
#     for ith_subject in list_subjects:
#         print("processing " + str(ith_subject))
#         current_scans = list(dict_subjects_data[dict_subjects_data['ID'] == ith_subject]['scans'])[0]
#         current_paths = list(dict_subjects_data[dict_subjects_data['ID'] == ith_subject]['paths'])[0]
#
#         for ith_scan in dict_subj_scans[ith_subject]:
#             if '.vtk' in ith_scan and 'sub' in ith_scan:
#                 scantime = float(str(ith_scan.split("-")[2][1:3]))
#
#                 # current scan with covariates
#                 current_age = scantime / 12 + float(dict_subjects_data[dict_subjects_data['ID'] == ith_subject]['age'])
#                 current_edu_level = float(dict_subjects_data[dict_subjects_data['ID'] == ith_subject]['edu'])
#                 current_path = np.array(current_paths)[ith_scan == np.array(current_scans)][0]
#                 current_pid = ith_subject
#                 current_scan = ith_scan
#
#                 #
#                 generate_shapes_and_sdfs(current_path, ROOT_DATASET)
#
#                 list_scan_data.append({"ID": current_scan,
#                                        "PID": current_pid,
#                                        'age': current_age,
#                                        'edu': current_edu_level,
#                                        'path': current_path})
#
#     return list_scan_data



def create_adni_dataset_sheet(list_data):

    dict_subjects_data = pd.DataFrame.from_records(list_data)
    list_scan_data = []
    for ith_dict in list_data:
        current_path = ith_dict['path']
        print("processing " + str(current_path))

        # current scan with covariates
        current_age = ith_dict['age']
        current_edu_level = ith_dict['edu']
        current_pid = ith_dict['ID']
        current_scan = ith_dict['scan']

        current_AD = int(ith_dict['group'] == 2)
        current_MCI = int(ith_dict['group'] > 0)
        current_sex = ith_dict['sex']

        #

        #generate_shapes_and_sdfs(current_path, ROOT_DATASET)
        #os.system('python create_sdf_from_mesh.py --subj ' + str(source_subj))

        savepath_on_npy = os.path.join(ROOT_DATASET, '3dshape', current_scan.split('.')[0] + '_on.npy')
        savepath_off_npy = os.path.join(ROOT_DATASET, '3dsdf', current_scan.split('.')[0] + '_off.npy')
        savepath_on_pv = os.path.join(ROOT_DATASET, '3dvis', current_scan.split('.')[0] + '_on_aligned.stl')


        list_scan_data.append({"id": current_scan,
                               "PID": current_pid,
                               'age': current_age,
                               'sex':current_sex,
                               'edu': current_edu_level,
                               'AD': current_AD,
                               'MCI': current_MCI,
                               'path': current_path,
                               '3dshape': savepath_on_npy,
                               '3dsdf': savepath_off_npy,
                               '3dvis': savepath_on_pv})

    return list_scan_data

def read_mesh_as_trimesh(path_mesh):
    # unpacking
    pv_data = pv.read(path_mesh)
    #current_airway.faces = np.hstack((np.ones((faces.shape[0], 1)) * 3, faces)).ravel().astype('int')
    faces = np.reshape(pv_data.faces, (-1, 4))[:, [1, 2, 3]]
    surf_airway = trimesh.Trimesh(vertices=pv_data.points, faces=faces)
    #surf_airway.show()
    return surf_airway




# def generate_shapes_and_sdfs(path_vtk, root_dataset):
#
#
#     # get id
#     scan_id = path_vtk.split('/')[-1]
#
#     # get savepath
#     savepath_on_npy = os.path.join(root_dataset, '3dshape', scan_id.split('.')[0] + '_on.npy')
#     savepath_off_npy = os.path.join(root_dataset, '3dsdf', scan_id.split('.')[0] + '_off.npy')
#     savepath_on_pv = os.path.join(root_dataset, '3dshape', scan_id.split('.')[0] + '_on.vtk')
#
#     #path_examples = "/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data/sub-ADNI002S0729_ses-M00.vtk"
#
#     # off surface points
#     mesh = read_mesh_as_trimesh(path_vtk)
#     # off surface points
#     arr_off = calculate_sdf_from_mesh(mesh)
#     # on surface points
#     pv_data = pv.read(path_vtk)
#     pv_data.save(savepath_on_pv)
#
#     np.save(savepath_on_npy, np.concatenate((np.array(mesh.vertices), np.array(mesh.vertex_normals)), axis=-1))
#     np.save(savepath_off_npy, arr_off)
#
#     del arr_off, mesh, pv_data
#
#     return






if __name__ == "__main__":

    path_seg = "/playpen-raid/jyn/NAISR/NAISR/examples/hippocampus/ADNI" #"/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/hippocampus/ADNI"
    list_subjects = os.listdir(path_seg)


    root_adni_dataset = "/playpen-raid/jyn/NAISR/NAISR/examples/hippocampus/ADNI" #"/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/hippocampus/ADNI" #"/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data" # "/playpen-raid/jyn/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data"  # "/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data"
    ROOTPATH_DEMMOG = "/playpen-raid/jyn/NAISR/NAISR/examples/hippocampus/PTDEMOG_18Aug2023.csv" #"/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/hippocampus/PTDEMOG_18Aug2023.csv" # "/home/jyn/NAISR/examples/hippocampus/PTDEMOG_18Aug2023.csv"
    ROOTPATH_DATASET_INFO = "/playpen-raid/jyn/NAISR/NAISR/examples/hippocampus/DATASET_INFO.csv"
    list_subjects, dict_subj_scans, list_data = extract_patient_convariates(root_adni_dataset)

    ROOT_DATASET = "/playpen-raid/jyn/NAISR/NAISR/examples/hippocampus/" #"/playpen-raid/jyn/NAISR/NAISR/examples/hippocampus/

    list_scan_data = create_adni_dataset_sheet(list_data)

    # save the data set info to a csv
    savepath_dataset = "/home/jyn/NAISR/examples/hippocampus/3dshape.csv" #"/home/jyn/NAISR/examples/starman/2dshape_train.csv"
    # save data
    a = pd.DataFrame.from_records(list_scan_data)
    a.loc[np.isnan(a['edu'].values), 'edu'] = a.loc[np.isfinite(a['edu'].values), 'edu'].mean()
    a.to_csv(savepath_dataset)

    print('finished')


