
import logging
import numpy as np
import plyfile
import skimage.measure as measure
import time
import torch
import os
import pyvista as pv
import dataio
import matplotlib.pyplot as plt
from utils import *
#import point_cloud_utils as pcu
#import hausdorff
import trimesh
from naisr import diff_operators

import pickle
import pymeshfix as pmf
import pandas as pd



def save_pickle(dictionary, name):
    with open(name, 'wb') as f:
        pickle.dump(dictionary, f)


def fix_mesh(pv_shape):
    pv_shape = pv_shape.triangulate()#.extract_surface()
    #fixer = pmf.MeshFix(pv_shape.triangulate().extract_surface())
    #fixer.repair()
    #pv_shape = fixer.mesh
    return pv_shape

def plotter_evolution_comp(list_of_pred_shapepath, list_of_gt_shapepath, savepath, list_text=None, print_on_figure=False):
    # plotting
    # pv.global_theme.background = 'white'
    # Spacing = np.array([1, 1, 1])



    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,len(list_of_pred_shapepath)), off_screen=True, window_size=(300*len(list_of_pred_shapepath), 512), border=False)
    #pv.rcParams['transparent_background'] = True
    pv.global_theme.background = 'white'
    pv.global_theme.interactive = True

    list_pred_shapes = []
    list_gt_shapes = []
    for i in range(len(list_of_pred_shapepath)):
        pv_pred_shape = pv.read(list_of_pred_shapepath[i])
        list_pred_shapes.append(fix_mesh(pv_pred_shape))

        pv_gt_shape = pv.read(list_of_gt_shapepath[i])
        pv_gt_shape.points /= 60
        list_gt_shapes.append(fix_mesh(pv_gt_shape))

    for ith_pred in range(len(list_pred_shapes)):
        p.subplot(0, ith_pred)
        #pred_coords = list_pred_shapes[ith_pred].points #/60
        #gt_coords = list_gt_shapes[ith_pred].points  # /60

        p.add_mesh(list_pred_shapes[ith_pred], color='red', point_size=1, opacity=0.3)
        p.add_mesh(list_gt_shapes[ith_pred], color='blue', point_size=1, opacity=0.3)

        p.camera_position = 'yz'
        p.camera.azimuth = -30

        if list_text is not None:
            if print_on_figure:
                print(list_text[ith_pred])
                p.add_text('age: ' + list_text[ith_pred]['age'] +
                           '\n weight' + list_text[ith_pred]['weight'],
                           color='black',
                           position='upper_edge',
                           font_size=12,
                           font='times')
            else:
                volume_pred = np.round(list_pred_shapes[ith_pred].volume * 60* 60 * 60 / 1000, decimals=2)
                volume_gt = np.round(list_gt_shapes[ith_pred].volume * 60 * 60 * 60 / 1000, decimals=2)
                list_text[ith_pred].update({'est. volume': volume_pred})
                list_text[ith_pred].update({'ori. volume': volume_gt})


            #p.add_text('\n', color='black', position='upper_edge')
            #p.add_text(list_text[ith_pred][1], color='black',  position='upper_edge')
        #p.camera.zoom(1)
    #p.export_obj('/home/jyn/NAISR/examples/pediatric_airway/statistics/' + str(filename) + '.obj')

        p.camera.zoom(1.1)
    p.link_views()
    p.screenshot(savepath +  '_comp.png')#, window_size=(256*len(list_of_pred_shapepath), 512))
    p.export_html(savepath + '_comp.html', backend='panel')

    p.close()
    pd.DataFrame.from_records(list_text).transpose().to_csv(savepath + '_cov_vol.csv')
    pd.DataFrame.from_records(list_text).transpose().to_latex(savepath + '_cov_vol.txt')


    '''
    '''

    for ith_pred in range(len(list_pred_shapes)):

        pv.start_xvfb()
        pv.global_theme.background = 'white'
        p = pv.Plotter(lighting="light_kit", window_size=(300, 400),off_screen=True, border=False)
        pv.global_theme.background = 'white'
        pv.global_theme.interactive = True
        #pv.rcParams['transparent_background'] = True



        p.add_mesh(list_pred_shapes[ith_pred], color='red', point_size=1, opacity=0.3)
        p.add_mesh(list_gt_shapes[ith_pred], color='blue', point_size=1, opacity=0.3)

        p.camera_position = 'yz'
        p.camera.azimuth = -30
        p.camera.zoom(1.2)

        if ith_pred == 0:
            cpos = p.camera_position
        p.camera_position = cpos

        p.screenshot(os.path.join(savepath, str(ith_pred) + '.png'))#, window_size=(256*len(list_of_pred_shapepath), 512))
        p.export_html(os.path.join(savepath, str(ith_pred) +'.html'), backend='panel')
        print(os.path.join(savepath, str(ith_pred) + '.png'))
        p.close()

    return

def Spherical_np(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def plotter_evolution(list_of_shapepath,  savepath, list_text=None, list_colors=None):
    # plotting
    # pv.global_theme.background = 'white'
    # Spacing = np.array([1, 1, 1])

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,len(list_of_shapepath)), off_screen=True, window_size=(256*len(list_of_shapepath), 512), border=False)
    #pv.rcParams['transparent_background'] = True
    pv.global_theme.interactive = True

    print(list_of_shapepath)

    list_shapes = []
    if list_text is None:
        list_text = []
        for i in range(len(list_of_shapepath)):
            list_text.append({})
    for i in range(len(list_of_shapepath)):
        #pv.read(list_of_shapepath[i]).points /= 60
        list_shapes.append(pv.read(list_of_shapepath[i]))


    for ith_pred in range(len(list_shapes)):
        p.subplot(0,ith_pred)

        if list_colors is not None:
            #list_colors[ith_pred] = Spherical_np(np.array(list_colors[ith_pred]))
            #colors = (list_colors[ith_pred]  / np.array([[2.6, 3.2, 3.2]]) * 255).astype(
            #    'uint8')
            a = np.array(list_colors[ith_pred])
            a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
            a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
            a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())

            list_shapes[ith_pred].field_data['colors'] = a
            p.add_mesh(list_shapes[ith_pred],
                       scalars='colors',
                       rgb=True,
                       point_size=1)
            volume_pred = np.round(list_shapes[ith_pred].volume * 60 * 60 * 60 / 1000, decimals=2)

            p.add_text('Vol.: ' + str(volume_pred), color='black',
                       position='lower_edge',
                       font_size=12,
                       font='times')
            p.camera_position = 'yz'
            p.camera.azimuth = -30
        else:
            '''
            list_shapes[ith_pred] = Spherical_np(np.array(list_shapes[ith_pred]))
            colors = (list_shapes[ith_pred] / np.array([[2.6, 3.2, 3.2]]) * 255).astype(
                'uint8')

            #colors = ((list_shapes[ith_pred].points + 2.5) / 5 * 255).astype('uint8')
            list_shapes[ith_pred].field_data['colors'] = colors
            p.add_mesh(list_shapes[ith_pred],
                       scalars='colors',
                       rgb=True,
                       point_size=1)
            '''
            #p.add_mesh(list_shapes[ith_pred],
            #           colormap='twilight',
            #           scalars=list_shapes[ith_pred].points,#[: ,-1],
            #           point_size=1,
            #           show_scalar_bar=False)
            a = list_shapes[ith_pred].points.copy()
            a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
            a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
            a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
            a *= 255
            a = a.astype('uint8')
            list_shapes[ith_pred].field_data['colors'] = a  # pv.ColorLike(list_shapes[ith_pred].points)
            p.add_mesh(list_shapes[ith_pred],
                       scalars='colors',
                       rgb=True,
                       point_size=1)
            volume_pred = np.round(list_shapes[ith_pred].volume * 60 * 60 * 60 / 1000, decimals=2)
            p.add_text('Vol.: ' + str(volume_pred), color='black',
                       position='lower_edge',
                       font_size=12,
                       font='times')

            p.camera_position = 'yz'
            p.camera.azimuth = -30





        list_text[ith_pred].update({'est. volume': volume_pred})


        p.camera.zoom(1.3)

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


    return



def plotter_evolution_for_methods(list_of_shapepath,  savepath, list_text=None):
    # plotting
    # pv.global_theme.background = 'white'
    # Spacing = np.array([1, 1, 1])

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,len(list_of_shapepath)), off_screen=True, window_size=(330*len(list_of_shapepath), 512), border=False)
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True
    pv.rcParams['transparent_background'] = True
    print(list_of_shapepath)

    list_shapes = []
    for i in range(len(list_of_shapepath)):
        current_shape = pv.read(list_of_shapepath[i])
        if i == 0:
            current_shape.points /= 60
            #list_of_shapepath[i].append(current_shape)
        list_shapes.append(current_shape)


    for ith_pred0 in range(len(list_shapes)):
        ith_pred = len(list_shapes) - ith_pred0 - 1
        p.subplot(0, ith_pred)
        '''
        p.add_mesh(list_shapes[ith_pred],
                   scalars=list_shapes[ith_pred].points[:, -1],
                   colormap='twilight',
                   point_size=1,
                   show_scalar_bar=False,)
        '''
        a = list_shapes[ith_pred].points.copy()
        a[:, 0] = (a[:, 0] - a[:, 0].min())/(a[:,0].max() - a[:, 0].min())
        a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
        a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
        a = a * 0.6 + 0.4
        a *= 255
        a =a.astype('uint8')
        list_shapes[ith_pred].field_data['colors'] = a#pv.ColorLike(list_shapes[ith_pred].points)
        p.add_mesh(list_shapes[ith_pred],
                   scalars='colors',
                    rgb=True,
                   point_size=1)
        p.camera.zoom(1.3)
        p.camera_position = 'yz'
        p.camera.azimuth = -30

        if list_text is not None:
            p.add_text(str(list_text[ith_pred]), color='black',position='lower_edge', font_size=12, font='times')


        #p.camera.zoom(1)
    #p.export_obj('/home/jyn/NAISR/examples/pediatric_airway/statistics/' + str(filename) + '.obj')
    p.link_views()
    p.screenshot(savepath +  '.png')
    p.export_html(savepath + '.html', backend='panel')
    p.close()



    # plot individual figures
    for ith_pred0 in range(len(list_shapes)):
        pv.start_xvfb()
        pv.global_theme.background = 'white'
        p = pv.Plotter(lighting="light_kit", off_screen=True, window_size=(512, 512), border=False)
        pv.global_theme.background = 'white'
        pv.global_theme.interactive = True
        pv.rcParams['transparent_background'] = True

        ith_pred = len(list_shapes) - ith_pred0 - 1
        '''
        p.add_mesh(list_shapes[ith_pred],
                   scalars=list_shapes[ith_pred].points[:, -1],
                   colormap='twilight',
                   point_size=1,
                   show_scalar_bar=False,)
        '''
        a = list_shapes[ith_pred].points.copy()
        a[:, 0] = (a[:, 0] - a[:, 0].min())/(a[:,0].max() - a[:, 0].min())
        a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
        a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
        a = a * 0.6 + 0.4
        a *= 255
        a =a.astype('uint8')
        list_shapes[ith_pred].field_data['colors'] = a#pv.ColorLike(list_shapes[ith_pred].points)
        p.add_mesh(list_shapes[ith_pred],
                   scalars='colors',
                    rgb=True,
                   point_size=1)
        p.camera.zoom(1.3)
        p.camera_position = 'yz'
        p.camera.azimuth = -30
        if ith_pred0 == 0:
            cpos = p.camera_position
        p.camera_position = cpos

        #if list_text is not None:
        #    p.add_text(str(list_text[ith_pred]), color='black',position='lower_edge', font_size=12, font='times')

        cond_mkdir(savepath)
        name = list_text[ith_pred].replace('/', '-')
        p.screenshot(os.path.join(savepath, name + 'yz.png'))
        p.export_html(os.path.join(savepath, str(name)+ 'yz.html'), backend='panel')

        p.close()

    # plot individual figures
    for ith_pred0 in range(len(list_shapes)):
        pv.start_xvfb()
        pv.global_theme.background = 'white'
        p = pv.Plotter(lighting="light_kit", off_screen=True, window_size=(512, 512), border=False)
        pv.global_theme.background = 'white'
        pv.global_theme.interactive = True
        pv.rcParams['transparent_background'] = True

        ith_pred = len(list_shapes) - ith_pred0 - 1
        '''
        p.add_mesh(list_shapes[ith_pred],
                   scalars=list_shapes[ith_pred].points[:, -1],
                   colormap='twilight',
                   point_size=1,
                   show_scalar_bar=False,)
        '''
        a = list_shapes[ith_pred].points.copy()
        a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
        a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
        a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
        a = a * 0.6 + 0.4
        a *= 255
        a = a.astype('uint8')
        list_shapes[ith_pred].field_data['colors'] = a  # pv.ColorLike(list_shapes[ith_pred].points)
        p.add_mesh(list_shapes[ith_pred],
                   scalars='colors',
                   rgb=True,
                   point_size=1)
        p.camera.zoom(1.3)
        p.camera_position = 'xz'
        p.camera.azimuth = -30
        if ith_pred0 == 0:
            cpos = p.camera_position
        p.camera_position = cpos
        # if list_text is not None:
        #    p.add_text(str(list_text[ith_pred]), color='black',position='lower_edge', font_size=12, font='times')

        cond_mkdir(savepath)
        name = list_text[ith_pred].replace('/', '-')
        p.screenshot(os.path.join(savepath, name + 'xz.png'))
        p.export_html(os.path.join(savepath, str(name) + 'xz.html'), backend='panel')
        p.close()

    return



def visualize_a_case(savepath_specifc, path_to_show, colormap='twilight', colors=None, normalize=False):
    pv_shape = pv.read(path_to_show)
    #if pv_shape.volume < 1000:
    if normalize:
        pv_shape.points *= 60



    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit",
                   window_size=(200, 256),
                   off_screen=True,
                   border=False)
    if colors is not None:
        #colors = Spherical_np(colors)
        #colors = (colors / np.array([[2.6, 3.2, 3.2]]) * 255).astype('uint8')
        a = colors
        a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
        a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
        a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())

        pv_shape.field_data['colors'] = a
        p.add_mesh(pv_shape,
                   scalars='colors',
                   rgb=True,
                   point_size=1)
        vol = np.around(pv_shape.volume / 1000, 2)
        p.add_text('Vol.: ' + str(vol), color='black',
                   position='lower_left',
                   font_size=12,
                   font='times')
        p.camera_position = 'yz'
        p.camera.azimuth = -30
        p.camera.zoom(1.4)
        p.screenshot(savepath_specifc + '.png')
        p.export_html(savepath_specifc + '.html', backend='panel')
        p.close()
    else:
        if colormap == 'twilight':
            p.add_mesh(pv_shape,
                       colormap='twilight',
                       scalars=pv_shape.points[:, -1],
                       point_size=1,show_scalar_bar=False)
            p.camera_position = 'yz'
            p.camera.azimuth = -30
            vol = np.around(pv_shape.volume / 1000, 2)
            p.add_text('Vol.: ' + str(vol), color='black',
                       position='lower_left',
                       font_size=12,
                       font='times')
        if colormap == 'rgb':

            a = pv_shape.points.copy()
            a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
            a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
            a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
            a *= 255
            a = a.astype('uint8')
            pv_shape.field_data['colors'] = a  # pv.ColorLike(list_shapes[ith_pred].points)
            p.add_mesh(pv_shape,
                       scalars='colors',
                       rgb=True,
                       point_size=1)
            vol = np.around(pv_shape.volume/1000, 2)
            p.add_text('Vol.: ' + str(vol), color='black',
                       position='lower_left',
                       font_size=12,
                       font='times')
            #p.camera.position = (0., 0., 0.0)
            p.camera_position = 'yz'
            p.camera.azimuth = -30
            p.camera.zoom(1.4)

        p.screenshot(savepath_specifc + '.png')
        p.export_html(savepath_specifc + '.html', backend='panel')
        p.close()
    return

def plotter_evolution_shapematrix(dict_of_shapepath,  savepath, dict_text0, dict_colors0):
    # plotting
    # pv.global_theme.background = 'white'
    # Spacing = np.array([1, 1, 1])
    dict_text = dict_text0.copy()

    pv.start_xvfb()
    #pv.global_theme.background = 'white'
    pv.rcParams['transparent_background'] = True
    num_of_x = len(dict_of_shapepath.keys())
    num_of_y = len(dict_of_shapepath[list(dict_of_shapepath.keys())[0]].keys())
    p = pv.Plotter(lighting="light_kit",
                   shape=(num_of_y, num_of_x),
                   window_size=(num_of_x*256, num_of_y*256),
                   off_screen=True,
                   border=False)

    pv.global_theme.background ='white'
    pv.global_theme.interactive = True



    dict_shapes = {}
    list_of_i = sorted(list(dict_of_shapepath.keys()))
    list_of_j = {}
    for i, dict_subset in dict_of_shapepath.items():
        dict_shapes[i] = {}
        list_of_j[i] = list(reversed(sorted(list(dict_of_shapepath[i].keys())))) #sorted(list(dict_of_shapepath[i].keys()))
        for j, path in dict_subset.items():
            dict_shapes[i][j] = pv.read(dict_of_shapepath[i][j])


    for ith_i_idx in range(len(list_of_i)):
        i = list_of_i[ith_i_idx]
        for ith_j_idx in range(len(list_of_j[i])):
            j = list_of_j[i][ith_j_idx]
            p.subplot(ith_j_idx, ith_i_idx,)

            if dict_colors0 is not None:
                dict_colors = dict_colors0.copy()
                #current_color = Spherical_np(np.array(dict_colors[i][j]))
                #colors = (current_color  / np.array([[2.6, 3.2, 3.2]]) * 255).astype('uint8')

                a = dict_colors[i][j]
                a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
                a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
                a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
                a = a * 0.6 + 0.4
                a *= 255

                dict_shapes[i][j].field_data['colors'] = a.astype('uint8')
                p.add_mesh(dict_shapes[i][j],
                           scalars='colors',
                           rgb=True,
                           point_size=1)
                p.camera_position = 'yz'
                p.camera.azimuth = -30
            else:
                #current_color = Spherical_np(np.array(dict_shapes[i][j].points))
                #colors = (current_color / np.array([[4., 3.2, 3.2]]) * 255).astype('uint8')
                #dict_shapes[i][j]
                #dict_shapes[i][j].field_data['colors'] = colors
                #p.add_mesh(dict_shapes[i][j],
                #           scalars='colors',
                #           rgb=True,
                #           point_size=1)

                a = dict_shapes[i][j].copy().points
                a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
                a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
                a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
                a = a * 0.7 + 0.3
                a *= 255
                dict_shapes[i][j].field_data['colors'] = a.astype('uint8')
                p.add_mesh(dict_shapes[i][j],
                           scalars='colors',
                           rgb=True,
                           point_size=1)
                p.camera_position = 'yz'
                p.camera.azimuth = -30
                '''
                p.add_mesh(dict_shapes[i][j],
                           scalars=dict_shapes[i][j].points[:,2],
                           colormap='twilight',
                           point_size=1,
                           show_scalar_bar=False,)
                p.camera_position = 'yz'
                p.camera.azimuth = -30
                
                a = dict_shapes[i][j].points.copy()
                a[:, 0] = (a[:, 0] - a[:, 0].min()) / (a[:, 0].max() - a[:, 0].min())
                a[:, 1] = (a[:, 1] - a[:, 1].min()) / (a[:, 1].max() - a[:, 1].min())
                a[:, 2] = (a[:, 2] - a[:, 2].min()) / (a[:, 2].max() - a[:, 2].min())
                a *= 255
                a = a.astype('uint8')
                dict_shapes[i][j].field_data['colors'] = a  # pv.ColorLike(list_shapes[ith_pred].points)
                p.add_mesh(dict_shapes[i][j],
                           scalars='colors',
                           rgb=True,
                           point_size=1)
                '''

            volume_pred = np.round(dict_shapes[i][j].volume * 60 * 60 * 60 / 1000, decimals=2)
            dict_text[i][j].update({'Vol.': volume_pred})
            #dict_text[i][j]['age'] = (230 - 3.) * dict_text[i][j]['age'].detach().cpu().numpy()[0] + 3.
            #dict_text[i][j]['weight'] = (150.0 - 6.75) * dict_text[i][j]['weight'].detach().cpu().numpy()[0]  + 6.75

            p.camera.zoom(0.8)

            if dict_text is not None:
                current_text = ''
                for name, value in dict_text[i][j].items():
                    if name == 'Vol.':
                        current_text += name + ': ' + str(value) + '\n'
                    p.add_text(current_text, color='black',
                               position='lower_left',
                                font_size=15,
                               font='times')
            #p.add_text(str(np.round(i, 1)) + ', ' + str(np.round(j, 1)), color='black',
            #           position='lower_left',
            #            font_size=15,
            #           font='times')
        #p.camera.zoom(1)
    #p.export_obj('/home/jyn/NAISR/examples/pediatri c_airway/statistics/' + str(filename) + '.obj')
    p.link_views()
    p.screenshot(savepath +  'shapemat.png')
    p.export_html(savepath + 'shappemat.html', backend='panel')
    p.close()
    #pd.DataFrame.from_records(list_text).transpose().to_csv(savepath + '_cov_vol_.csv')


    return

'''
def plotter_evolution(list_evolution, savepath):

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,len(list_evolution)), off_screen=True, window_size=[1024, 512], border=False)
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    centers = []
    list_shapes = []
    for i in range(len(list_evolution)):
        #centers.append(np.array([0, i, -0.5]) * 3)
        list_shapes.append(load_pickle(list_evolution[i]+ '.pkl') )
    for ith_pred in range(len(list_shapes)):
        p.subplot(0, ith_pred)
        p.add_mesh(pv.PolyData(list_shapes[ith_pred]['verts']), cmap=plt.cm.get_cmap("cool"), scalars=list_shapes[ith_pred]['color'], clim=[-1, 1])

    p.link_views()
    p.screenshot(savepath +  '.png')
    p.export_html(savepath + '.html', backend='panel')
    p.close()


    returnƒcomp
'''

def extract_data_to_plot_shape_with_vf(path_surface, attributes, lat_vec, model, which_attribute):
    pv_3dairway = trimesh.load(path_surface)
    verts = pv_3dairway.vertices
    faces = pv_3dairway.faces

    lat_vec_subset = lat_vec[0, [0], :].repeat(verts.shape[0], 1).to(model.device)[None, :, :]

    for name, value in attributes.items():
        attributes[name].requires_grad_()
    model_out = model.evolution(torch.from_numpy(verts).to(model.device)[None, ...].float(), attributes, lat_vec_subset, which_attribute)

    gradient_covaraites= (torch.sign(diff_operators.gradient(model_out['model_out'], model_out['covariates'])) * torch.sign(model_out['covariates'])).detach().cpu().numpy().squeeze()
    print( model_out['covariates'].shape)
    print(gradient_covaraites.shape)
    #names = ['weight', 'age', 'sex']
    #gradient_covariates = gradient_covaraites[..., np.array(names) == which_attribute]
    #sdf_on_template = model.get_template(torch.from_numpy(verts).to(model.device)[None, ...].float(), lat_vec_subset)['model_out'][0].squeeze()

    vec_field = torch.norm(model_out['vec_fields'][which_attribute], dim=-1).detach().cpu().numpy().squeeze()
    #for name, arr in vec_field.items():
    #    vec_field[name] = vec_field[name].detach().cpu().numpy()[0]
    #pv_3dairway = pv.read(path_surface)
    #pv_3dairway.add_field_data(sdf_on_template.detach().cpu().numpy()[0], 'sdf_for_covariates')
    #pv_3dairway.save(path_surface)
    #print(sdf_on_template.shape)
    #print(vec_field.shape)
    a = vec_field * gradient_covaraites  #sdf_on_template.detach().cpu().numpy() #torch.sign(sdf_on_template).detach().cpu().numpy() #* vec_field #sdf_on_template.detach().cpu().numpy() #
    #print(a.shape)
    return verts, faces, a#gradient_covariates.detach().cpu().numpy()

def plotter_3d_airway_reconstruction_with_vf(verts, faces, gradient_covariates, savepath):

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    # store canonical coordinates as rgb color (in float format)
    #for name, vec_field in vec_field.items():
    #verts_color = 255 * (0.5 + 0.5 * gradient_covariates)
    #verts_color = verts_color.astype(np.uint8)

    verts_tuple = np.zeros(
        (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "f4"), ("green", "f4"), ("blue", "f4")])

    #for i in range(0, num_verts):
    #    verts_tuple[i] = (verts[i][0], verts[i][1], verts[i][2],
    #                      verts_color[i][0], verts_color[i][1], verts_color[i][2])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    html_filename_out = savepath +  '.html'
    png_filename_out = savepath + '.png'
    stl_filename_out = savepath + '.stl'
    '''
    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
    '''
    pv_shape = pv.PolyData()
    pv_shape.points = verts

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1, 1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background = 'white'
    pv.global_theme.interactive = True
    p.add_mesh(pv.PolyData(verts),cmap=plt.cm.get_cmap("cool"), scalars=gradient_covariates,  clim=[-0.1, 0.1]) #scalars=verts_color, rgb=True)
    p.screenshot(png_filename_out)
    p.export_html(html_filename_out, backend='panel')
    print(png_filename_out)

    #pv_shape.add_field_data(gradient_covariates, 'sdf_for_covariates')
    #pv_shape.save(stl_filename_out)
    #np.save(, )
    save_pickle({'verts': verts, 'color': gradient_covariates}, savepath+'.pkl')
    return



    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    p.add_mesh(pv.PolyData(arr_dealigned_points), point_size=1, color='lightblue')

    p.screenshot(os.path.join(savedir,  'recons3d.png'))
    #p.enable_zoom_style()
    #p.export_vtkjs(savepath)
    p.export_html(os.path.join(savedir,  'recons3d.html'), backend='panel')
    p.close()
    return




def save_to_ply(verts, verts_warped, faces, ply_filename_out):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    # store canonical coordinates as rgb color (in float format)
    verts_color = 255 * (0.5 + 0.5 * verts_warped)
    verts_color = verts_color.astype(np.uint8)

    verts_tuple = np.zeros(
        (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "f4"), ("green", "f4"), ("blue", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = (verts[i][0], verts[i][1], verts[i][2],
                          verts_color[i][0], verts_color[i][1], verts_color[i][2])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)




def plotter_2d_csa(list_path_gt_csa, list_path_pred_csa, savedir_2dcsa_subj):
    #
    assert len(list_path_gt_csa) == len(list_path_pred_csa), f"Inconsistent number of 2D CSA with {savedir_2dcsa_subj}"
    '''
    get pred and gt point cloud
    '''
    list_pred_pc = []
    list_gt_pc = []
    for ith in range(len(list_path_gt_csa)):
        if len(list_path_gt_csa[ith]) > 0:
            list_gt_pc.append(np.load(list_path_gt_csa[ith][0])[:, [0,1,2]])
            list_pred_pc.append(np.load(list_path_pred_csa[ith])[:, [0,1,2]])
    max_p = np.abs(np.concatenate(list_gt_pc, axis=0)).max()

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

    for ith in range(len(list_path_gt_csa)):
        if len(list_path_gt_csa[ith]) > 0:
            ith_gt_2dcsa = list_gt_pc[ith] / max_p + centers[ith]
            ith_pred_2dcsa = list_pred_pc[ith] / max_p + centers[ith]
            p.add_mesh(ith_gt_2dcsa, color='b', point_size=3)
            p.add_mesh(ith_pred_2dcsa, color='r', point_size=3)

    p.view_vector((5.0, 2, 3))
    p.add_floor('-z', lighting=True, color='grey', pad=1.0)
    # p.enable_shadows()
    p.screenshot(os.path.join(savedir_2dcsa_subj, 'comp2d.png'))
    p.export_html(os.path.join(savedir_2dcsa_subj, 'comp2d.html'), backend='panel')
    p.close()

    return


def plotter_3d_airway_reconstruction(arr_dealigned_points, savedir):

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    p.add_mesh(pv.PolyData(arr_dealigned_points), point_size=1, color='lightblue')

    p.screenshot(os.path.join(savedir,  'recons3d.png'))
    #p.enable_zoom_style()
    #p.export_vtkjs(savepath)
    p.export_html(os.path.join(savedir,  'recons3d.html'), backend='panel')
    p.close()
    return


def plotter_3d_airway_from_2d_evaluation(arr_gt_3dcsa, arr_pred_3dcsa, arr_all_centerlines, savedir):

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    p.add_mesh(pv.PolyData(arr_gt_3dcsa), point_size=3, color='lightblue', opacity=0.3)
    p.add_mesh(pv.PolyData(arr_pred_3dcsa), point_size=3, color='pink', opacity=0.3)
    p.add_mesh(pv.PolyData(arr_all_centerlines), point_size=10, color='black')

    p.screenshot(os.path.join(savedir,  'comp3d.png'))
    p.export_html(os.path.join(savedir,  'comp3d.html'), backend='panel')
    p.close()
    return



def plotter_3d_airway_evaluation(arr_gt_3dcsa, arr_pred_3dcsa, savedir):

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    p.add_mesh(pv.PolyData(arr_gt_3dcsa), point_size=3, color='lightblue', opacity=0.3)
    p.add_mesh(pv.PolyData(arr_pred_3dcsa), point_size=3, color='pink', opacity=0.3)
    p.screenshot(os.path.join(savedir,  'comp3d.png'))
    p.export_html(os.path.join(savedir,  'comp3d.html'), backend='panel')
    p.close()
    return


def plotter_3d_airway_interpolation(list_arr_gt_3dcsa, list_arr_pred_3dcsa, savedir):

    pv.start_xvfb()
    pv.global_theme.background = 'white'
    p = pv.Plotter(lighting="light_kit", shape=(1,1), off_screen=True, window_size=[1024, 1024])
    pv.global_theme.background ='white'
    pv.global_theme.interactive = True

    centers = []
    for i in range(len(list_arr_gt_3dcsa) + len(list_arr_pred_3dcsa)):
        centers.append(np.array([0, i, -0.5]) * 3)

    p.add_mesh(pv.PolyData(list_arr_gt_3dcsa[0] + centers[0]), point_size=3, color='lightblue', opacity=0.3)
    for ith_pred in range(len(list_arr_pred_3dcsa)):
        p.add_mesh(pv.PolyData(list_arr_pred_3dcsa[ith_pred] + centers[1+ith_pred]), point_size=3, color='pink', opacity=0.3)
    p.add_mesh(pv.PolyData(list_arr_gt_3dcsa[1] + centers[-1]), point_size=3, color='lightblue', opacity=0.3)

    p.screenshot(os.path.join(savedir,  'interpolation.png'))
    p.export_html(os.path.join(savedir,  'interpolation.html'), backend='panel')
    p.close()
    return