import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt


def load_yaml_as_dict(yaml_path):
    import yaml
    with open(yaml_path, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict

def get_ids(filename_split, split='train'):
    split = load_yaml_as_dict(filename_split)[split]
    return split


def read_data(split, filename_datasource, attributes):
    df_data = pd.read_csv(filename_datasource, header=0)
    df_data_split = df_data.loc[df_data['id'].astype('str').isin(split)]

    # read covariates
    list_attributes = []
    for ith_attribute in attributes:
        arr_current_attribute = np.array(df_data_split[ith_attribute])
        list_attributes.append(arr_current_attribute)
    features = np.array(list_attributes).T
    return features



def get_airway_data_for_id(test_idx, filename_datasource, train_split, attributes_names):
    train_covariates = read_data(train_split, filename_datasource, attributes_names)
    case_covariates = read_data(test_idx, filename_datasource, attributes_names)

    attributes = {}
    mean_attributes = {}
    std_attributes = {}
    for ith_attri in range(len(attributes_names)):
        attributes[attributes_names[ith_attri]] = (case_covariates[:, ith_attri] - train_covariates[:, ith_attri].mean()) / train_covariates[:,ith_attri].std()
        attributes[attributes_names[ith_attri]] = attributes[attributes_names[ith_attri]][None, :]
        mean_attributes[attributes_names[ith_attri]]  = train_covariates[:,ith_attri].mean()
        std_attributes[attributes_names[ith_attri]] = train_covariates[:,ith_attri].std()

    return attributes, mean_attributes, std_attributes



def get_adni_data_for_id(test_idx, df_data, train_split, attributes_names):
    train_covariates = read_data(train_split, df_data, attributes_names)
    case_covariates = read_data(test_idx, df_data, attributes_names)

    attributes = {}
    mean_attributes = {}
    std_attributes = {}
    for ith_attri in range(len(attributes_names)):
        attributes[attributes_names[ith_attri]] = (case_covariates[:, ith_attri] - train_covariates[:,
                                                                                   ith_attri].mean()) / train_covariates[
                                                                                                        :,
                                                                                                        ith_attri].std()
        attributes[attributes_names[ith_attri]] = attributes[attributes_names[ith_attri]][None, :]
        mean_attributes[attributes_names[ith_attri]]  = train_covariates[:,ith_attri].mean()
        std_attributes[attributes_names[ith_attri]] = train_covariates[:,ith_attri].std()

    return attributes, mean_attributes, std_attributes

def get_data_for_id(shapetype):
    if shapetype == 'ADNI':
        return get_adni_data_for_id
    elif shapetype == 'Airway':
        return get_airway_data_for_id



if __name__ == "__main__":
    path_dataset = 'examples/pediatric_airway/3dshape.csv'
    filename_split = 'examples/pediatric_airway/newsplit.yaml'
    pd_cov = pd.read_csv(path_dataset)

    all_ids = get_ids(filename_split, split='train') + get_ids(filename_split, split='test')
    train_ids = get_ids(filename_split, split='train')
    attributes, mean_attributes, std_attributes = get_data_for_id("Airway")(all_ids, path_dataset, train_ids, ['weight', 'age', 'sex'])

    pd_cov = pd_cov.loc[pd_cov['id'].astype('str').isin(all_ids)]
    pd_cov_train= pd_cov.loc[pd_cov['id'].astype('str').isin(train_ids)]

    train_ages = pd_cov_train['age'].values
    train_weight = pd_cov_train['weight'].values

    pd_cov['age'] = attributes['age'][0]
    pd_cov['weight'] = attributes['weight'][0]


    scatters = sns.jointplot(data=pd_cov, x="age", y="weight", hue="sex", palette={
        1: "#00FFFF",
        0: "#FF00FF"
    },alpha=0.35
                             )
    palette={
        1: "#00FFFF",
        0: "#FF00FF"
    }
    #for i,gr in pd_cov.groupby('sex'):
    #    sns.regplot(x="age", y="weight", data=gr, scatter=False, ax=scatters.ax_joint, truncate=False,
    #                scatter_kws={"color": palette[i]}, line_kws={"color": palette[i]})

    scatters.ax_marg_x.set_xlim(-2.2, 2.2)
    scatters.ax_marg_y.set_ylim(-2.2, 4.2)

    #scatters.ax_marg_x.set_xlim(-1. * std_attributes['age'] + mean_attributes['age'], 2. * std_attributes['age'] + mean_attributes['age'])
    #scatters.ax_marg_y.set_ylim(-1. * std_attributes['weight'] + mean_attributes['weight'], 2. * * std_attributes['weight'] + mean_attributes['weight'])


    # set the labels
    scatters.ax_marg_x.set_xticks(np.linspace(-2., 2., 7))# * std_attributes['age'] + mean_attributes['age'])
    scatters.ax_marg_x.set_xticklabels(np.round(np.linspace(-2., 2., 7) * std_attributes['age'] + mean_attributes['age'], 1))
    scatters.ax_marg_y.set_yticks(np.linspace(-2., 4., 7))# * std_attributes['weight'] + mean_attributes['weight'])
    scatters.ax_marg_y.set_yticklabels(np.round(np.linspace(-2., 4., 7) * std_attributes['weight'] + mean_attributes['weight'], 1))


    scatters.set_axis_labels('age / month', 'weight / kg', fontsize=16)
    # boxes.set(title=title)
    #fig = scatters.get_figure()
    #sns.move_legend(scatters, "lower right")
    scatters.ax_joint.legend_.remove()
    scatters.savefig('./data_generation/a.svg', transparent=True)