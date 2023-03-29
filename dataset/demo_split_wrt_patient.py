import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml




def extract_scans_from_patients(list_patient_scans):
    list_scans = []
    for case in list_patient_scans:
        list_scans+=case['value']
    return list_scans

datasetpath = '/home/jyn/NAISR/examples/pediatric_airway/csa_1120.csv'
metinfopath = '/playpen-raid/jyn/NAISR/data2viz/dataset/FilteredControlBlindingLogUniqueScanFiltered_19Sep2022.xls'

metainfo = pd.read_excel(metinfopath)
df_dataset = pd.read_csv(datasetpath)

list_id = np.unique(np.array(df_dataset['id'])).astype('str').tolist()
metainfo = metainfo.loc[metainfo['scan #'].astype('str').isin(list_id)][0:-1]
PID= np.array(metainfo['UNC  MR#'])
list_PID = []


for i_PID in PID:
    if i_PID not in list_PID and str(i_PID)!='nan':
        list_PID.append(i_PID)
print('how many patients: ' + str(len(list_PID)))


dict_patient_scans = {}

for i_PID in list_PID:
    current_patient = metainfo[metainfo['UNC  MR#'] == i_PID]
    dict_patient_scans[i_PID] = np.array(current_patient['scan #']).astype('str').tolist()

# dataset statistics
num_single_case = 0
num_multiple_case = 0
num_single_time = 0
num_multiple_time = 0
single_cases = []
multiple_cases = []
for name, value in dict_patient_scans.items():
    if len(value) == 1:
        num_single_case += 1
        num_single_time += len(value)
        single_cases += value
    elif len(value) > 1:
        num_multiple_case += 1
        num_multiple_time += len(value)
        multiple_cases += value
    else:
        print('???' + str(name))
print('single case: ' + str(num_single_time))
print('multiple case: ' + str(num_multiple_time))
print('single time: ' + str(num_single_case))
print('multiple time: ' + str(num_multiple_case))

#for i_case in list_id:
#    if i_case not in single_cases+multiple_cases:
#        print(i_case)
#        dict_patient_scans[i_case] = [i_case]

# correct dataset statistics
num_single_case = 0
num_multiple_case = 0
num_single_time = 0
num_multiple_time = 0
single_cases = []
multiple_cases = []
single_patients = []
multiple_patients = []

for name, value in dict_patient_scans.items():
    if len(value) == 1:
        num_single_case += 1
        num_single_time += len(value)
        single_cases += value
        single_patients.append({'name': str(name),
                                'value':value})
    elif len(value) > 1:
        num_multiple_case += 1
        num_multiple_time += len(value)
        multiple_cases += value
        multiple_patients.append({'name': str(name),
                                'value': value})
    else:
        print('???' + str(name))
print('corrected single case: ' + str(num_single_time))
print('corrected multiple case: ' + str(num_multiple_time))
print('corrected single time: ' + str(num_single_case))
print('corrected multiple time: ' + str(num_multiple_case))



'''
multiple_train_set, multiple_test_set = train_test_split(multiple_patients, train_size=0.8, random_state=98)
single_train_set, single_test_set = train_test_split(single_patients, train_size=0.8, random_state=98)

list_trains = extract_scans_from_patients(multiple_train_set) + extract_scans_from_patients(single_train_set)
list_test = extract_scans_from_patients(multiple_test_set) + extract_scans_from_patients(single_test_set)

#list_train_ids = df_dataset.loc[df_dataset['id'].isin(train_set)]
#list_test_ids = df_dataset[test_set]
#list_train_train_ids = df_dataset[train_train_set]
#list_train_val_ids = df_dataset[train_val_set]


dict_split = {'train': list_trains,
              'test': list_test,
              'all': list_id,
              'train_single': extract_scans_from_patients(single_train_set),
              'train_multiple': multiple_train_set,
              'test_single': extract_scans_from_patients(single_test_set),
              'test_multiple':multiple_test_set
              }



savepath = '/home/jyn/NAISR/examples/pediatric_airway/newsplit.yaml'
with open(savepath, 'w') as f:
    yaml.dump(dict_split, f, default_flow_style=False, sort_keys=False)


dict_timeline = {}
for ith_case in multiple_patients:
    dict_timeline[str(ith_case['name'])] = ith_case['value']

savepath = '/home/jyn/NAISR/examples/pediatric_airway/timeline_patients.yaml'
with open(savepath, 'w') as f:
    yaml.dump(dict_timeline, f, default_flow_style=False, sort_keys=False)
'''
print('done')