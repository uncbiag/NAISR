import pyvista as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotter_evolution(list_of_shapepath,  savepath, list_text=None, list_colors=None):
    # plotting
    # pv.global_theme.background = 'white'
    # Spacing = np.array([1, 1, 1])

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,len(list_of_shapepath)), off_screen=True, window_size=(256*len(list_of_shapepath), 512), border=False)
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    print(list_of_shapepath)

    list_shapes = []
    for i in range(len(list_of_shapepath)):
        a = pv.read(list_of_shapepath[i])
        a.points /= 60
        list_shapes.append(a)

    for ith_pred in range(len(list_shapes)):
        p.subplot(0, ith_pred)

        if list_colors is not None:
            a = np.array(list_colors[ith_pred])
            a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
            a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
            a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
            a = a * 0.6 + 0.4
            a *= 255
            list_shapes[ith_pred].field_data['colors'] = a.astype('uint8')
            p.add_mesh(list_shapes[ith_pred],
                       scalars='colors',
                       rgb=True,
                       point_size=1)
            p.camera_position = 'yz'
            p.camera.azimuth = -30
        else:
            a = np.array(list_shapes[ith_pred].points.copy())
            a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
            a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
            a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
            a = a * 0.6 + 0.4
            a *= 255
            list_shapes[ith_pred].field_data['colors'] = a.astype('uint8')

            p.add_mesh(list_shapes[ith_pred],
                       scalars='colors',
                       rgb=True,
                       point_size=1)
            p.camera_position = 'yz'
            p.camera.azimuth = -30
        volume_pred = np.round(list_shapes[ith_pred].volume * 60 * 60 * 60 / 1000, decimals=2)
        list_text[ith_pred].update({'est. volume': volume_pred})

        p.camera.zoom(0.6)

        #if list_text is not None:
        #    p.add_text(str(list_text[ith_pred]), color='black')
        #p.camera.zoom(1)
    #p.export_obj('/home/jyn/NAISR/examples/pediatric_airway/statistics/' + str(filename) + '.obj')
    p.link_views()
    p.screenshot(savepath +  '.png')
    p.export_html(savepath + '.html', backend='panel')
    p.close()
    pd.DataFrame.from_records(list_text).transpose().to_csv(savepath + '_cov_vol_.csv')
    pd.DataFrame.from_records(list_text).transpose().to_latex(savepath + '_cov_vol_.txt')

    '''
    '''

    for ith_pred0 in range(len(list_shapes)):
        ith_pred = len(list_shapes) - ith_pred0 - 1
        pv.start_xvfb()
        pv.global_theme.background = 'white'
        p = pv.Plotter(lighting="light_kit", window_size=(300, 400), off_screen=True, border=False)
        pv.global_theme.background = 'white'
        pv.global_theme.interactive = True
        # pv.rcParams['transparent_background'] = True

        a = np.array(list_shapes[ith_pred].points.copy())
        a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
        a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
        a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
        a = a * 0.6 + 0.4
        a *= 255
        list_shapes[ith_pred].field_data['colors'] = a.astype('uint8')

        p.add_mesh(list_shapes[ith_pred],
                   scalars='colors',
                   rgb=True,
                   point_size=1)
        p.camera_position = 'yz'
        p.camera.azimuth = -30
        p.camera.zoom(1.5)

        if ith_pred0 == 0:
            cpos = p.camera_position
        p.camera_position = cpos

        if ith_pred == 0:
            cpos = p.camera_position
        p.camera_position = cpos

        p.screenshot(
            os.path.join(savepath, str(ith_pred) + '.png'))  # , window_size=(256*len(list_of_pred_shapepath), 512))
        p.export_html(os.path.join(savepath, str(ith_pred) + '.html'), backend='panel')

        p.close()

def visualize_airway(list_of_shapepath, list_text, filename):
    # plotting
    # pv.global_theme.background = 'white'
    # Spacing = np.array([1, 1, 1])



    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,len(list_of_shapepath)), off_screen=True, window_size=(1024, 512))
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    centers = []
    list_shapes = []
    for i in range(len(list_of_shapepath)):
        list_shapes.append(pv.read(list_of_shapepath[i]))

    for ith_pred in range(len(list_shapes)):
        p.subplot(0, ith_pred)
        list_shapes[ith_pred].points = list_shapes[ith_pred].points / 60
        p.add_mesh(list_shapes[ith_pred], colormap='twilight', scalars=list_shapes[ith_pred].points[:, -1])
        p.add_text(str(list_text[ith_pred]), color='black')
    #p.export_obj('/home/jyn/NAISR/examples/pediatric_airway/statistics/' + str(filename) + '.obj')

    p.screenshot('/home/jyn/NAISR/examples/pediatric_airway/statistics/' + str(filename) +  '.png')
    p.export_html('/home/jyn/NAISR/examples/pediatric_airway/statistics/' + str(filename) + '.html', backend='panel')

    p.close()

    return


def load_yaml_as_dict(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


timeline = '/home/jyn/NAISR/examples/pediatric_airway/timeline_patients.yaml'
datasource = "/home/jyn/NAISR/examples/pediatric_airway/3dshape.csv"

subj_to_show = '18241505' #'4134915-0'
split = load_yaml_as_dict(timeline)[subj_to_show]

df_data = pd.read_csv(datasource, header=0)
df_data_split = df_data.loc[df_data['id'].astype('str').isin(split)]


# read covariates
attributes = ['age', 'weight', 'sex']
path_3dshape = np.array(df_data_split['3dshape'])
id_values = np.array(df_data_split['id'].astype('str'))


#dict_age_scans = {}
#dict_weight_scans = {}
list_ages = []
list_weights = []
list_sex = []
list_path = []
list_ids = []
for ith_scan in split:
    current_data = df_data_split[df_data['id'].astype('str') == ith_scan]
    #dict_age_scans[ith_scan] = current_data['age'].values[0]
    #dict_weight_scans[ith_scan] = current_data['weight'].values[0]
    list_ages.append(current_data['age'].values[0])
    list_weights.append(current_data['weight'].values[0])
    list_sex.append(current_data['sex'].values[0])
    list_ids.append(current_data['id'].values[0])
    list_path.append(df_data[df_data['id'].astype('str')==ith_scan]['3dshape'].values[0])



list_sorted_ages = np.argsort(list_ages).astype('int')
#list_sorted_weights = np.argsort(list_weights).astype('int')

list_sorted_age_scans = np.array(list_path)[list_sorted_ages]
list_sorted_weight_scans = np.array(list_path)[list_sorted_ages]  #np.array(list_path)[list_sorted_weights]
#list_ages_text = [str(np.array(list_ages)[i])  + ' months' for i in list_sorted_ages]
#list_weights_text = [str(np.array(list_weights)[i]) + ' kgs.' for i in list_sorted_weights]
list_ages_text = [{'id': list_ids[i], 'age': np.array(list_ages)[i], 'weight': np.array(list_weights)[i], 'sex': np.array(list_sex)[i]} for i in list_sorted_ages]

#visualize_airway(list_sorted_age_scans, list_ages_text, subj_to_show + 'age')
#visualize_airway(list_sorted_weight_scans, list_weights_text, subj_to_show + 'weight')

import os
# read covariates
savepath = os.path.join('/home/jyn/NAISR/examples/pediatric_airway/statistics/', str(subj_to_show + 'age'))
if not os.path.exists(savepath):
    os.makedirs(savepath)
plotter_evolution(list_sorted_age_scans,  savepath, list_text=list_ages_text, list_colors=None)
savepath = os.path.join('/home/jyn/NAISR/examples/pediatric_airway/statistics/', str(subj_to_show + 'weight'))
if not os.path.exists(savepath):
    os.makedirs(savepath)
plotter_evolution(list_sorted_weight_scans,  savepath, list_text=list_ages_text, list_colors=None)
