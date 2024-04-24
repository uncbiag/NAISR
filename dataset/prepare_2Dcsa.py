import os
import pandas as pd
import numpy as np
import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)



rootpath = '/playpen-raid/jyn/pediatric_atlas/FinalData/AtlasExperimentPipeline/atlas_airway_data_new_v2/new_cross_sections/'
os.listdir(rootpath)

datasetpath = '/home/jyn/NAISR/examples/pediatric_airway/csa.csv'
#wrong_cases = ['1300', '1301', '1302', '1303', '1304', '1305', '1306', '1307', '1308']

#
df_dataset = pd.read_csv(datasetpath)
list_ids = np.unique(np.array(df_dataset['id'])).astype('str').tolist()

#
for id in list_ids:
    filename = os.path.join(rootpath, str(id) + '_2DCSA.p3')
    csa2d = load_pickle(filename)
    print('a')
