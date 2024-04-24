import pyvista as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_yaml_as_dict(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict

timeline = '/home/jyn/NAISR/examples/pediatric_airway/newsplit.yaml'
datasource = "/home/jyn/NAISR/examples/pediatric_airway/3dshape.csv"


dict_num_of_observations = {}
num_observations = 0
subj_to_show = 'train_single' #'4134915-0'
num_observations += len(load_yaml_as_dict(timeline)[subj_to_show])
subj_to_show = 'test_single' #'4134915-0'
num_observations += len(load_yaml_as_dict(timeline)[subj_to_show])
dict_num_of_observations[1] = num_observations

subj_to_show = 'test_multiple'
split = load_yaml_as_dict(timeline)[subj_to_show]
for i_split in split:
    value = i_split['value']
    if len(value) in dict_num_of_observations.keys():
        dict_num_of_observations[len(value)] += 1
    else:
        dict_num_of_observations[len(value)] = 1

subj_to_show = 'train_multiple'
split = load_yaml_as_dict(timeline)[subj_to_show]
for i_split in split:
    value = i_split['value']
    if len(value) in dict_num_of_observations.keys():
        dict_num_of_observations[len(value)] += 1
    else:
        dict_num_of_observations[len(value)] = 1


print(dict_num_of_observations)

