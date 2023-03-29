import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


datasetpath = '/home/jyn/NAISR/examples/pediatric_airway/csa_1120.csv'
#wrong_cases = ['1300', '1301', '1302', '1303', '1304', '1305', '1306', '1307', '1308']

df_dataset = pd.read_csv(datasetpath)



list_id = np.unique(np.array(df_dataset['id'])).astype('str').tolist()
#for to_remove in wrong_cases:
#    list_id.remove(to_remove)

train_set, test_set = train_test_split(list_id, train_size=0.8)
train_train_set, train_val_set = train_test_split(train_set, test_size=0.125)

#list_train_ids = df_dataset.loc[df_dataset['id'].isin(train_set)]
#list_test_ids = df_dataset[test_set]
#list_train_train_ids = df_dataset[train_train_set]
#list_train_val_ids = df_dataset[train_val_set]


dict_split = {'train': train_set,
              'train_train': train_train_set,
              'train_val': train_val_set,
              'test': test_set,
              'all': list_id}
savepath = '/home/jyn/NAISR/examples/pediatric_airway/split.yaml'
with open(savepath, 'w') as f:
    yaml.dump(dict_split, f, default_flow_style=False, sort_keys=False)
