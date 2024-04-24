import pyvista as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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
            p.camera_position = 'xy'
            p.camera.view_angle = 160.0
            p.camera.roll = -10
        volume_pred = np.round(list_shapes[ith_pred].volume  / 1000, decimals=2)
        list_text[ith_pred].update({'est. volume': volume_pred})

        p.camera.zoom(10)

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
        p.camera_position = 'xy'
        p.camera.view_angle = 160.0
        p.camera.roll = -10
        p.camera.zoom(6)

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
    p = pv.Plotter(lighting="light_kit", shape=(1,len(list_of_shapepath)), off_screen=True, window_size=(650, 512), border=False)
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True



    centers = []
    list_shapes = []
    for i in range(len(list_of_shapepath)):
        centers.append(np.array([0, i, -0.5]) * 3)
        list_shapes.append(pv.read(list_of_shapepath[i]))

    for ith_pred in range(len(list_shapes)):
        p.subplot(0, ith_pred)
        #coords = list_shapes[ith_pred].points
        p.add_mesh(list_shapes[ith_pred], cmap='twilight', scalars=list_shapes[ith_pred].points[:, -1])
        p.add_text(str(list_text[ith_pred]), color='black')
        #p.camera.zoom(1)
    #p.export_obj('/home/jyn/NAISR/examples/pediatric_airway/statistics/' + str(filename) + '.obj')
    p.link_views()
    p.screenshot('/home/jyn/NAISR/examples/hippocampus/statistics/' + str(filename) +  '.png')
    p.export_html('/home/jyn/NAISR/examples/hippocampus/statistics/' + str(filename) + '.html', backend='panel')

    p.close()


    return


def load_yaml_as_dict(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict



def get_percentiles_of_covariates(covariates, ids, list_paths):
    cov_0 = np.percentile(covariates, 0)
    cov_10 = np.percentile(covariates, 10)
    cov_20 = np.percentile(covariates, 20)
    cov_30 = np.percentile(covariates, 30)
    cov_40 = np.percentile(covariates, 40)
    cov_50 = np.percentile(covariates, 50)
    cov_60 = np.percentile(covariates, 60)
    cov_70 = np.percentile(covariates, 70)
    cov_80 = np.percentile(covariates, 80)
    cov_90 = np.percentile(covariates, 90)
    cov_100 = np.percentile(covariates, 100)


    list_paths_perct = []
    list_conv_perct = []
    list_ids = []
    for i_cov in [cov_0, cov_10, cov_20, cov_30, cov_40, cov_50, cov_60, cov_70, cov_80, cov_90, cov_100]:
        ipath = np.array(list_paths)[np.argmin(np.abs(covariates - i_cov))]
        icov = np.array(covariates)[np.argmin(np.abs(covariates - i_cov))]
        iid = np.array(ids)[np.argmin(np.abs(covariates - i_cov))]
        list_paths_perct.append(ipath)
        list_conv_perct.append(icov)
        list_ids.append(iid)
    return list_ids, list_paths_perct, list_conv_perct


timeline = '/home/jyn/NAISR/examples/hippocampus/newsplit.yaml'
datasource = "/home/jyn/NAISR/examples/hippocampus/3dshape.csv"

subj_to_show = 'all' #'4134915-0'
if subj_to_show == 'all':
    split = load_yaml_as_dict(timeline)['train'] + load_yaml_as_dict(timeline)['test']

df_data = pd.read_csv(datasource, header=0)
df_data_split = df_data.loc[df_data['id'].astype('str').isin(split)]

arr_ages = df_data_split['age'].values
arr_AD = df_data_split['AD'].values
arr_sex = df_data_split['sex'].values
arr_edu = df_data_split['edu'].values
arr_ids = df_data_split['id']
list_paths = df_data_split['3dvis']

list_ids_age, list_paths_perct_age, list_conv_perct_age = get_percentiles_of_covariates(arr_ages, arr_ids, list_paths)
list_conv_perct_age_AD = []
list_conv_perct_age_sex = []
list_conv_perct_age_edu = []


for ith_id in list_ids_age:
    list_conv_perct_age_AD.append(arr_AD[arr_ids == ith_id][0])
    list_conv_perct_age_edu.append(arr_edu[arr_ids == ith_id][0])
    list_conv_perct_age_sex.append(arr_sex[arr_ids == ith_id][0])


pertentiles = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']#['0', '25', '50', '75','100']
list_ages_text = [{'id': list_ids_age[i],
                   'age': np.array(list_conv_perct_age )[i],
                   'AD': np.array(list_conv_perct_age_AD)[i],
                   'sex': np.array(list_conv_perct_age_sex)[i],
                   'edu': np.array(list_conv_perct_age_edu)[i],}
                  for i in range(len(list_conv_perct_age))]

# read covariates
savepath = os.path.join('/home/jyn/NAISR/examples/hippocampus/statistics/', str(subj_to_show + 'age'))
if not os.path.exists(savepath):
    os.makedirs(savepath)
plotter_evolution(list_paths_perct_age,  savepath, list_text=list_ages_text, list_colors=None)


#visualize_airway(list_paths_perct_age, list_ages_text, subj_to_show + 'age')
#visualize_airway(list_paths_perct_weight, list_weights_text, subj_to_show + 'weight')
