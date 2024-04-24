import os
import numpy as np
import pandas as pd
import math
import pyvista as pv
import trimesh
from utils_3d import *
import skimage.measure as measure
import datetime
from sklearn.model_selection import train_test_split
import yaml

def transform_group_to_num(str_class):
    if str_class == 'CN':
        return 0
    elif str_class == 'MCI':
        return 1
    elif str_class == 'AD':
        return 2

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

            #date = str(int(scandate.split('-')[1])) + '/' + str(scandate.split('-')[2]) + '/' + str(scandate.split('-')[0])
            #try:

                #current_age = float(current_datasetinfo[current_datasetinfo['Study Date'] == scandate]['Age'].values[0])
            time_diff = (current_datasetinfo['Study Date'] - pd.to_datetime(scandate)) / pd.Timedelta(1, 'D')
            current_age = float(current_datasetinfo[time_diff.abs()<=1]['Age'].values[0])
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
                              "edu": edu_level,
                              'group': current_group,
                              "scan": current_scanfile,
                              "path": current_path
                              })

    print("Done")
    return list_subjects, dict_subj_scans, list_data


if __name__ == "__main__":

    path_adni_dataset = "/home/jyn/NAISR/examples/hippocampus/3dshape.csv"
    root_adni_dataset = "/playpen-raid/jyn/NAISR/NAISR/examples/hippocampus/ADNI"
    ROOTPATH_DEMMOG = "/home/jyn/NAISR/examples/hippocampus/PTDEMOG_18Aug2023.csv" # "/home/jyn/NAISR/examples/hippocampus/PTDEMOG_18Aug2023.csv"
    ROOTPATH_DATASET_INFO = "/home/jyn/NAISR/examples/hippocampus/DATASET_INFO.csv"

    df_dataset = pd.read_csv(path_adni_dataset)
    np.unique(list(df_dataset['PID'].values))
    list_subjects, dict_subj_scans, list_data = extract_patient_convariates(root_adni_dataset)
    train_split, test_split = train_test_split(list_subjects, train_size=0.8)


    list_train = []
    list_test = []
    list_train_single = []
    list_train_multiple = []
    list_test_single = []
    list_test_multiple = []

    for i_train in train_split:
        list_current_scans = list(df_dataset[df_dataset['PID']==i_train]['ID'].values) #dict_subj_scans[i_train]
        list_train += list_current_scans
        if len(list_current_scans) == 1:
            list_train_single += list_current_scans
        elif len(list_current_scans) > 1:
            list_train_multiple.append({"name": i_train, "value": list_current_scans})

    for i_test in test_split:
        list_current_scans = list(df_dataset[df_dataset['PID']==i_test]['ID'].values) #dict_subj_scans[i_test]
        list_test += list_current_scans
        if len(list_current_scans) == 1:
            list_test_single += list_current_scans
        elif len(list_current_scans) > 1:
            list_test_multiple.append({"name": i_test, "value":list_current_scans})

    dict_split = {'train': list_train,
                  'test': list_test,
                  'train_single': list_train_single,
                  'train_multiple': list_train_multiple,
                  'test_single': list_test_single,
                  'test_multiple': list_test_multiple
                  }

    savepath = '/home/jyn/NAISR/examples/hippocampus/newsplit.yaml'
    with open(savepath, 'w') as f:
        yaml.dump(dict_split, f, default_flow_style=False, sort_keys=False)

    dict_timeline = {}
    for ith_case in dict_split['train_multiple'] + dict_split['test_multiple']:
        dict_timeline[str(ith_case['name'])] = ith_case['value']

    savepath = '/home/jyn/NAISR/examples/pediatric_airway/timeline_patients.yaml'
    with open(savepath, 'w') as f:
        yaml.dump(dict_timeline, f, default_flow_style=False, sort_keys=False)

    print('finished')


