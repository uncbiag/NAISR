import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import os
import pickle as pk
import open3d as o3d
# Monkey-patch torch.utils.tensorboard.SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter


def unitcross(a, b):
    c = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    return c
def rotation_of_a_plane(M, N):
    costheta = np.matmul(M, N) / (np.linalg.norm(M) * np.linalg.norm(N))

    axis = unitcross(M, N)

    c = costheta
    s = np.sqrt(1 - c * c)
    C = 1 - c
    x, y, z = axis[0], axis[1], axis[2]
    rmat = np.array([[x * x * C + c,    x * y * C - z * s,  x * z * C + y * s],
                    [y * x * C + z * s,  y * y * C + c,    y * z * C - x * s],
                    [z * x * C - y * s, z * y * C + x * s, z * z * C + c]])
    return rmat


#path = '/Users/jyn/jyn/research/projects/NAISR/NAISR/examples/pediatric_airway/1030_UNITIZED_SEGMENTATION.mha'

#arr, Origin, Spacing = read_origin_spacing(path)
#surface = surface_construction(arr, Origin, Spacing)

def register_landmarks(p1_current, p2_template):

    p1_current = p1_current.squeeze()
    p2_template = p2_template.squeeze()
    #Take transpose as columns should be the points
    p1 = p1_current.transpose()
    p2 = p2_template.transpose()

    #Calculate centroids
    p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 1).reshape((-1,1))

    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())

    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

    #Calculate rotation matrix
    R = np.matmul(V_t.transpose(),U.transpose())
    print(np.linalg.det(R))
    #assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    #Calculate translation matrix
    T = p2_c - np.matmul(R,p1_c)

    #Check result
    result = T + np.matmul(R,p1)
    if np.allclose(result,p2):
        print("transformation is correct!")
    else:
        print("transformation is wrong...")
    return  R, T

def apply_registration_to_pts(p1, R, T):
    result = T + np.matmul(R, p1)
    return result


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pk.load(f)

def make_vector_field():

    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # Vector origin location
    X = [0]
    Y = [0]

    # Directional vectors
    U = [2]
    V = [1]

    # Creating plot
    plt.quiver(X, Y, U, V, color='b', units='xy', scale=1)
    plt.title('Single Vector')

    # x-lim and y-lim
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    # Show plot with grid
    plt.grid()
    plt.show()

    return

def make_displacement(arr_sdf_2d, vecter_field, plotname):
    vecter_field = dataio.lin2img(vecter_field).detach().squeeze().cpu().numpy()

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    #grid_x, grid_y = np.meshgrid(np.linspace(-1, 1,128), np.linspace(-1, 1,128))
    slice_coords_2d = dataio.get_mgrid(128)
    gd = dataio.lin2img(slice_coords_2d[None, ...]).squeeze().cpu().numpy()
    grid_x, grid_y = gd[0], gd[1]

    def plot_grid(x, y, ax=None, **kwargs):
        ax = ax or plt.gca()
        segs1 = np.stack((x, y), axis=2)
        segs2 = segs1.transpose(1, 0, 2)
        ax.add_collection(LineCollection(segs1,  **kwargs))
        ax.add_collection(LineCollection(segs2, **kwargs))
        ax.autoscale()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)


    def plot_shape(ax, arr_sdf_2d, vecter_field, plotname):
        num_levels = 10
        max_v = np.maximum(np.abs(arr_sdf_2d.min()), np.abs(arr_sdf_2d.max()))
        levels = np.linspace(-max_v, max_v + 0.001, num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

        sample = arr_sdf_2d #np.flipud(arr_sdf_2d)
        CS = ax.contourf(grid_x, grid_y, sample, levels=levels, colors=colors)
        cbar = fig.colorbar(CS)
        ax.contour(grid_x, grid_y, sample, levels=levels, colors='k', linewidths=0.1)
        ax.contour(grid_x, grid_y, sample, levels=[0], colors='k', linewidths=0.3)
        #plt.sca(ax)
        ax.contour(grid_x, grid_y, grid_x - vecter_field[0], np.linspace(-1, 1, 25), linestyles='solid', linewidths=2)
        ax.contour(grid_x, grid_y, grid_y - vecter_field[1], np.linspace(-1, 1, 25), linestyles='solid', linewidths=2)
        title = ''
        for i_key in plotname.keys():
            title += i_key + ': ' + str(plotname[i_key].detach().cpu().numpy()) + ', \n'
        ax.set_title(str(title), fontsize=5)
        ax.set_xlim(-1., 1.)
        ax.set_ylim(-1., 1.)
        ax.set_axis_on()
        return ax

    #f = lambda x, y: (x + 0.8 * np.exp(-x ** 2 - y ** 2), y)

    #plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
    #distx, disty = f(grid_x, grid_y)
    fig, ax = plt.subplots()  # figsize=(2.75, 2.75), dpi=300)
    ax = plot_shape(ax=ax,arr_sdf_2d=arr_sdf_2d, vecter_field=vecter_field,plotname=plotname)

    #plot_grid(grid_x, grid_y, ax=ax, color="lightgrey", linewidths=0.4,)
    #plot_grid(grid_x + vecter_field[0], grid_y + vecter_field[1], ax=ax, color="red",  linewidths=0.4,)

    #plt.show()
    return fig

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def make_contour_plot(array_2d, plotnamne, mode='log'):
    fig, ax = plt.subplots()#figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 20
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 20
        max_v = np.maximum(np.abs(array_2d.min()), np.abs(array_2d.max()))
        levels = np.linspace(-max_v, max_v+0.001 ,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    #grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    slice_coords_2d = dataio.get_mgrid(128)
    gd = dataio.lin2img(slice_coords_2d[None, ...]).squeeze().cpu().numpy()
    grid_x, grid_y = gd[0], gd[1]


    sample = array_2d #np.flipud(array_2d)
    CS = ax.contourf(grid_x, grid_y,sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(grid_x, grid_y, sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(grid_x, grid_y, sample, levels=[0], colors='k', linewidths=0.3)
    title = ''
    if type(plotnamne) == 'dict':
        for i_key in plotnamne.keys():
            title += i_key + ': ' + str(plotnamne[i_key].detach().cpu().numpy()) + ', \n'
    else:
        title = 'plot'
    ax.set_title(str(title), fontsize=5)
    ax.set_axis_on()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    #ax.axis('off')
    return fig

def make_curve(arr_1d, plotname, model='lin'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)
    ax.plot(np.linspace(0, 1, arr_1d.shape[0]), arr_1d.cpu().numpy().squeeze())
    title = ''
    for i_key in plotname.keys():
        title += i_key + ': ' + str(plotname[i_key][0].detach().cpu().numpy()) + ', \n'
    ax.set_title(str(title), fontsize=5)
    ax.set_axis_on()
    return fig

def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)

def write_dsp_summary(model, model_input, lat_vecs, attributes, gt, model_output, writer, total_steps, prefix='train_', device='cpu'):

    slice_coords_2d = dataio.get_mgrid(128)

    #with torch.no_grad():
    model.eval()

    '''
    1. x
    '''
    yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input =  yz_slice_coords.to(device)[None, ...]

    fixed_attributes = {}
    for ith_attri in attributes.keys():
        #print(attributes[ith_attri][0])
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]#[None, :]
    yz_model_out = model(yz_slice_model_input, fixed_attributes, yz_slice_model_input)

    if 'vec_fields'  in model_output.keys():
        maps = yz_model_out['vec_fields']
        for ith_attri in maps.keys():
            if ith_attri in yz_model_out['model_map'].keys():
                sdf_values = yz_model_out['template'].detach() #yz_model_out['model_map'][ith_attri].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [1, 2]], fixed_attributes)
                #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'yz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'templ' in ith_attri:
                sdf_values = yz_model_out['template'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [1, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'yz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'overall' in ith_attri:
                sdf_values = yz_model_out['template'].detach() #yz_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [1, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'yz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)


    '''
    2. y
    '''
    xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                 torch.zeros_like(slice_coords_2d[:, :1]),
                                 slice_coords_2d[:,-1:]), dim=-1)
    xz_slice_model_input = xz_slice_coords.to(device)[None, ...]

    xz_model_out = model(xz_slice_model_input, fixed_attributes, yz_slice_model_input)

    if 'vec_fields' in model_output.keys():
        maps = xz_model_out['vec_fields']
        for ith_attri in maps.keys():
            if ith_attri in xz_model_out['model_map'].keys():
                sdf_values = xz_model_out['template'].detach() #xz_model_out['model_map'][ith_attri].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 2]], fixed_attributes)
                #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'templ' in ith_attri:
                sdf_values = xz_model_out['template'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'overall' in ith_attri:
                sdf_values = xz_model_out['template'].detach() #xz_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)



    '''
    3. z
    '''
    xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                 torch.zeros_like(slice_coords_2d[:, :1]),), dim=-1)
    xy_slice_model_input =  xy_slice_coords.to(device)[None, ...]

    xy_model_out = model(xy_slice_model_input,fixed_attributes, yz_slice_model_input)

    if 'vec_fields' in model_output.keys():
        maps = xz_model_out['vec_fields']
        for ith_attri in maps.keys():
            if ith_attri in xy_model_out['model_map'].keys():
                sdf_values = xy_model_out['template'].detach() #xy_model_out['model_map'][ith_attri].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 1]], fixed_attributes)
                #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'templ' in ith_attri:
                sdf_values = xy_model_out['template'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 1]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'overall' in ith_attri:
                sdf_values =xy_model_out['template'].detach() # xy_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 1]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)



def write_deepsdf_dsp_summary(model, model_input, lat_vecs, attributes, gt, model_output, writer, total_steps, prefix='train_', device='cpu'):

    slice_coords_2d = dataio.get_mgrid(128)

    #with torch.no_grad():
    model.eval()

    '''
    1. x
    '''
    yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input =  yz_slice_coords.to(device)[None, ...]

    fixed_attributes = {}
    for ith_attri in attributes.keys():
        #print(attributes[ith_attri][0])
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]#[None, :]
    lat_vec_subset = lat_vecs[0, [0], :].repeat(yz_slice_model_input.shape[1], 1).to(device)[None, :, :]
    yz_model_out = model(yz_slice_model_input,fixed_attributes, lat_vec_subset, )

    if 'vec_fields'  in yz_model_out.keys():

        maps = yz_model_out['vec_fields']
        for ith_attri in maps.keys():
            sdf_values = yz_model_out['template'].detach() #yz_model_out['model_map'][ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            fig = make_displacement(sdf_values, maps[ith_attri][..., [1, 2]], fixed_attributes)
            #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
            writer.add_figure(prefix + 'yz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            if 'overall' in ith_attri:
                sdf_values = yz_model_out['template'].detach() #yz_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [1, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'yz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)


    '''
    2. y
    '''
    xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                 torch.zeros_like(slice_coords_2d[:, :1]),
                                 slice_coords_2d[:,-1:]), dim=-1)
    xz_slice_model_input = xz_slice_coords.to(device)[None, ...]

    #xz_model_out = model(xz_slice_model_input, fixed_attributes, yz_slice_model_input)
    lat_vec_subset = lat_vecs[0, [0], :].repeat(xz_slice_model_input.shape[1], 1).to(device)[None, :, :]
    xz_model_out = model(xz_slice_model_input,fixed_attributes, lat_vec_subset, )


    if 'vec_fields' in model_output.keys():
        maps = xz_model_out['vec_fields']
        for ith_attri in maps.keys():
            sdf_values = xz_model_out['template'].detach() #xz_model_out['model_map'][ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 2]], fixed_attributes)
            #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
            writer.add_figure(prefix + 'xz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            if 'overall' in ith_attri:
                sdf_values = xz_model_out['template'].detach() #xz_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)



    '''
    3. z
    '''
    xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                 torch.zeros_like(slice_coords_2d[:, :1]),), dim=-1)
    xy_slice_model_input =  xy_slice_coords.to(device)[None, ...]

    #xy_model_out = model(xy_slice_model_input,fixed_attributes, yz_slice_model_input)
    lat_vec_subset = lat_vecs[0, [0], :].repeat(yz_slice_model_input.shape[1], 1).to(device)[None, :, :]
    xy_model_out = model(xy_slice_model_input,fixed_attributes, lat_vec_subset, )

    if 'vec_fields' in model_output.keys():
        maps = xz_model_out['vec_fields']
        for ith_attri in maps.keys():
            sdf_values = xy_model_out['template'].detach() #xy_model_out['model_map'][ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 1]], fixed_attributes)
            #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
            writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            if 'overall' in ith_attri:
                sdf_values =xy_model_out['template'].detach() # xy_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 1]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)


def write_deepvfsdf_dsp_summary(model, batchvecs, attributes, gt, model_output, writer, total_steps, prefix='train_',device='cpu'):
    slice_coords_2d = dataio.get_mgrid(128)

    #with torch.no_grad():
    model.eval()

    '''
    1. x
    '''
    yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input =  yz_slice_coords.to(device)[None, ...]

    fixed_attributes = {}
    for ith_attri in attributes.keys():
        #print(attributes[ith_attri][0])
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]#[None, :]
    yz_model_out = model(yz_slice_model_input, fixed_attributes, yz_slice_model_input)

    if 'vec_fields'  in model_output.keys():
        maps = yz_model_out['vec_fields']
        for ith_attri in maps.keys():
            if ith_attri in yz_model_out['model_map'].keys():
                sdf_values = yz_model_out['template'].detach() #yz_model_out['model_map'][ith_attri].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [1, 2]], fixed_attributes)
                #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'yz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'templ' in ith_attri:
                sdf_values = yz_model_out['template'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [1, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'yz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'overall' in ith_attri:
                sdf_values = yz_model_out['template'].detach() #yz_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [1, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'yz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)


    '''
    2. y
    '''
    xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                 torch.zeros_like(slice_coords_2d[:, :1]),
                                 slice_coords_2d[:,-1:]), dim=-1)
    xz_slice_model_input = xz_slice_coords.to(device)[None, ...]

    xz_model_out = model(xz_slice_model_input, fixed_attributes, yz_slice_model_input)

    if 'vec_fields' in model_output.keys():
        maps = xz_model_out['vec_fields']
        for ith_attri in maps.keys():
            if ith_attri in xz_model_out['model_map'].keys():
                sdf_values = xz_model_out['template'].detach() #xz_model_out['model_map'][ith_attri].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 2]], fixed_attributes)
                #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'templ' in ith_attri:
                sdf_values = xz_model_out['template'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'overall' in ith_attri:
                sdf_values = xz_model_out['template'].detach() #xz_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 2]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xz_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)



    '''
    3. z
    '''
    xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                 torch.zeros_like(slice_coords_2d[:, :1]),), dim=-1)
    xy_slice_model_input =  xy_slice_coords.to(device)[None, ...]

    xy_model_out = model(xy_slice_model_input,fixed_attributes, yz_slice_model_input)

    if 'vec_fields' in model_output.keys():
        maps = xz_model_out['vec_fields']
        for ith_attri in maps.keys():
            if ith_attri in xy_model_out['model_map'].keys():
                sdf_values = xy_model_out['template'].detach() #xy_model_out['model_map'][ith_attri].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 1]], fixed_attributes)
                #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'templ' in ith_attri:
                sdf_values = xy_model_out['template'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 1]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'overall' in ith_attri:
                sdf_values =xy_model_out['template'].detach() # xy_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri][..., [0, 1]], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)




def create_mesh_from_implicit_shape(model, attributes, output_type, N, dim, device):
    head = 0
    num_samples = N ** 3
    max_batch = 32 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 2
    voxel_size = 2.0 / (N - 1) * 2

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N#overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]


    while head < num_samples:
        #print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:dim].to(device)
        attributes = {key: value.to(device) for key, value in attributes.items()}
        samples[head : min(head + max_batch, num_samples), -1] = (
            model(sample_subset[None, :, range(dim)], attributes, sample_subset[None, :, range(dim)])[output_type]
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    volume = sdf_values.detach().cpu().numpy()
    verts, faces, normals, values = np.zeros((1, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    import skimage.measure as measure

    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume, level=0., spacing=[voxel_size] * 3)
    except:
        pass

    return verts[None, :, :], faces



def create_mesh_from_deepsdf(model, attributes, lat_vec, output_type, N, dim, device):
    model.eval()
    head = 0
    num_samples = N ** 3
    max_batch = 16 ** dim

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array([-1., -1., -1.]) * 2
    voxel_size = 2.0 / (N - 1) * 2

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N # (overall_index.long() % (N*N)) % N #overall_index % N # (overall_index.long() % (N*N)) % N  #overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N # (overall_index.long() % (N*N)) // N #(overall_index.long() % (N*N)) / N #(overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N#overall_index.long() // (N * N) #((overall_index.long() // N) // N) % N #(overall_index.long() / N) / N #((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0].float() * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1].float() * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2].float() * voxel_size) + voxel_origin[0]
    #samples *=2


    while head < num_samples:
        #print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:dim].to(device)
        if type(lat_vec) == dict:
            lat_vec_subset = {}
            for name, value in lat_vec.items():
                lat_vec_subset[name] = lat_vec[name][0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        else:
            lat_vec_subset  = lat_vec[0, [0], :].repeat(sample_subset.shape[0], 1).to(device)[None, :, :]
        samples[head : min(head + max_batch, num_samples), -1] = (
            model(sample_subset[None, :, range(dim)], attributes, lat_vec_subset)[output_type]
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, -1]
    if dim == 3:
        sdf_values = sdf_values.reshape(N, N, N)
    else:
        sdf_values = sdf_values.reshape(N, N)[None, :, :]
    volume = sdf_values.detach().cpu().numpy()
    verts, faces, normals, values = np.zeros((1, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    import skimage.measure as measure
    voxel_size = 4.0 / (N - 1) * 1.

    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume, level=0., spacing=[voxel_size] * 3,)
    except:
        pass
    verts += voxel_origin
    return verts[None, :, :], faces


def write_sdf_summary(model, model_input, attributes, gt, model_output, writer, total_steps, prefix='train_',device='cpu'):
    slice_coords_2d = dataio.get_mgrid(128)

    #with torch.no_grad():
    #model.eval()

    fixed_attributes = {}
    for ith_attri in attributes.keys():
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]  # [None, :]

    #model_out_3d = model(slice_coords_3d, fixed_attributes, slice_coords_3d)['model_out'].detach().cpu().numpy()
    out_mesh, _ =create_mesh_from_implicit_shape(model, fixed_attributes, output_type='model_out', N=128, dim=3, device=model.device)
    #writer.add_3d('reconstruction', to_dict_batch([out_mesh]), step=total_steps)
    template_mesh, faces =create_mesh_from_implicit_shape(model,  fixed_attributes, output_type='template', N=128, dim=3, device=model.device)
    writer.add_mesh('template', vertices=torch.from_numpy(template_mesh), global_step=total_steps)
    writer.add_mesh('reconstruction', vertices=torch.from_numpy(out_mesh),global_step=total_steps)
    #writer.add_3d('template', to_dict_batch([template_mesh]), step=total_steps)

    '''
    1. x
    '''
    yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input =  yz_slice_coords.to(device)[None, ...]

    fixed_attributes = {}
    for ith_attri in attributes.keys():
        #print(attributes[ith_attri][0])
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]#[None, :]
    yz_model_out = model(yz_slice_model_input, fixed_attributes, yz_slice_model_input)
    sdf_values = yz_model_out['model_out'].detach()
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    #print(sdf_values.shape)
    fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
    writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)


    if 'model_map' in model_output.keys():
        maps = yz_model_out['model_map']
        for ith_attri in maps.keys():
            sdf_values = maps[ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            #print(sdf_values.shape)
            fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
            writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)



    '''
    2. y
    '''
    xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                 torch.zeros_like(slice_coords_2d[:, :1]),
                                 slice_coords_2d[:,-1:]), dim=-1)
    xz_slice_model_input = xz_slice_coords.to(device)[None, ...]

    xz_model_out = model(xz_slice_model_input, fixed_attributes, xz_slice_model_input)
    sdf_values = xz_model_out['model_out'].detach()
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
    writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

    if 'model_map' in model_output.keys():
        maps = xz_model_out['model_map']
        for ith_attri in maps.keys():
            sdf_values = maps[ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            # print(sdf_values.shape)
            fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
            writer.add_figure(prefix + 'xz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)


    '''
    3. z
    '''
    xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                 torch.zeros_like(slice_coords_2d[:, :1]),), dim=-1)
    xy_slice_model_input =  xy_slice_coords.to(device)[None, ...]

    xy_model_out = model(xy_slice_model_input,fixed_attributes, xy_slice_model_input)
    sdf_values = xy_model_out['model_out'].detach()
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
    writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

    min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
    min_max_summary(prefix + 'coords', model_input, writer, total_steps)

    if 'model_map' in model_output.keys():
        maps = xy_model_out['model_map']
        for ith_attri in maps.keys():
            sdf_values = maps[ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            # print(sdf_values.shape)
            fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
            writer.add_figure(prefix + 'xy_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)




def write_deepsdf_summary(model, model_input, lat_vecs,attributes,  gt, model_output, writer, total_steps, prefix='train_',device='cpu'):
    slice_coords_2d = dataio.get_mgrid(128)

    #with torch.no_grad():
    model.eval()
    colors_tensor = torch.as_tensor([
        [255, 0, 0],
        [0, 255, 0]])

    if type(lat_vecs) == dict:
        fixed_lat_vec = {}
        for name, value in lat_vecs.items():
            fixed_lat_vec[name] = lat_vecs[name][0][[0], :].repeat(slice_coords_2d.shape[0], 1)[None, :, :].to(device)
    else:
        fixed_lat_vec = lat_vecs[0]
        fixed_lat_vec = fixed_lat_vec[[0], :].repeat(slice_coords_2d.shape[0], 1)[None, :, :].to(device)

    fixed_attributes = {}
    for ith_attri in attributes.keys():
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]  # [None, :]


    out_mesh, _ = create_mesh_from_deepsdf(model, fixed_attributes, lat_vecs, output_type='model_out', N=128, dim=3, device=model.device)

    pred_color = colors_tensor[None, 0].repeat( out_mesh.shape[-2], 1)[None, :, :].to(device)
    gt_color = colors_tensor[None, 1].repeat(model_input.shape[1], 1)[None, :, :].to(device)
    arr_samples_comp = torch.cat((torch.from_numpy(out_mesh).to(device), model_input[0].reshape(-1, 3)[None, :, :]), dim=-2)
    arr_color_comp = torch.cat((pred_color, gt_color), dim=-2)
    writer.add_mesh('comparison: ' + str(gt['id'][0]), vertices=arr_samples_comp, global_step=total_steps, colors=arr_color_comp)
    writer.add_mesh('reconstruction: ' + str(gt['id'][0]), vertices = torch.from_numpy(out_mesh).to(device), global_step=total_steps, colors=pred_color)
    writer.add_mesh('gt: '+ str(gt['id'][0]), vertices=model_input[0].reshape(-1, 3)[None, :, :], global_step=total_steps, colors=gt_color)

    '''
    1. x
    '''
    yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input =  yz_slice_coords.to(device)[None, ...]

    yz_model_out = model(yz_slice_model_input, fixed_attributes, fixed_lat_vec)
    sdf_values = yz_model_out['model_out'].detach()
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    #print(sdf_values.shape)
    fig = make_contour_plot(sdf_values, gt['id'][0], mode='lin')
    writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)


    if 'model_map' in model_output.keys():
        maps = yz_model_out['model_map']
        for ith_attri in maps.keys():
            sdf_values = maps[ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            #print(sdf_values.shape)
            fig = make_contour_plot(sdf_values,  gt['id'][0], mode='lin')
            writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)



    '''
    2. y
    '''
    xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                 torch.zeros_like(slice_coords_2d[:, :1]),
                                 slice_coords_2d[:,-1:]), dim=-1)
    xz_slice_model_input = xz_slice_coords.to(device)[None, ...]

    xz_model_out = model(xz_slice_model_input, fixed_attributes, fixed_lat_vec)
    sdf_values = xz_model_out['model_out'].detach()
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values,  gt['id'][0], mode='lin')
    writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

    if 'model_map' in model_output.keys():
        maps = xz_model_out['model_map']
        for ith_attri in maps.keys():
            sdf_values = maps[ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            # print(sdf_values.shape)
            fig = make_contour_plot(sdf_values,  gt['id'][0], mode='lin')
            writer.add_figure(prefix + 'xz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)


    '''
    3. z
    '''
    xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                 torch.zeros_like(slice_coords_2d[:, :1]),), dim=-1)
    xy_slice_model_input =  xy_slice_coords.to(device)[None, ...]

    xy_model_out = model(xy_slice_model_input, fixed_attributes, fixed_lat_vec)
    sdf_values = xy_model_out['model_out'].detach()
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values, gt['id'][0], mode='lin')
    writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

    min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
    min_max_summary(prefix + 'coords', model_input, writer, total_steps)

    if 'model_map' in model_output.keys():
        maps = xy_model_out['model_map']
        for ith_attri in maps.keys():
            sdf_values = maps[ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            # print(sdf_values.shape)
            fig = make_contour_plot(sdf_values, gt['id'][0], mode='lin')
            writer.add_figure(prefix + 'xy_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)



def write_2dsdf_summary(model, model_input, attributes, gt, model_output, writer, total_steps, prefix='train_',device='cpu'):
    slice_coords_2d = dataio.get_mgrid(128)

    #with torch.no_grad():
    model.eval()
    '''
    1. x
    '''
    #yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input =  slice_coords_2d.to(device)[None, ...]

    #yz_slice_model_input = slice_coords_2d.to(device)
    fixed_attributes = {}
    for ith_attri in attributes.keys():
        #print(attributes[ith_attri][0])
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]#[None, :]
    yz_model_out = model(yz_slice_model_input, fixed_attributes)
    sdf_values = yz_model_out['model_out'].detach()
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    #print(sdf_values.shape)
    fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
    writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)


    if 'model_map' in model_output.keys():
        maps = yz_model_out['model_map']
        for ith_attri in maps.keys():
            sdf_values = maps[ith_attri].detach()
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            #print(sdf_values.shape)
            fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
            writer.add_figure(prefix + 'xy_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)

def write_2ddsp_summary(model, model_input, attributes, gt, model_output, writer, total_steps, prefix='train_',device='cpu'):
    slice_coords_2d = dataio.get_mgrid(128)

    #with torch.no_grad():
    model.eval()
    '''
    1. x
    '''
    #yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input =  slice_coords_2d.to(device)[None, ...]

    fixed_attributes = {}
    for ith_attri in attributes.keys():
        #print(attributes[ith_attri][0])
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]#[None, :]
    xy_model_out = model(yz_slice_model_input, fixed_attributes)

    if 'vec_fields'  in model_output.keys():
        maps = xy_model_out['vec_fields']
        for ith_attri in maps.keys():
            if ith_attri in xy_model_out['model_map'].keys():
                sdf_values = xy_model_out['template'].detach() #yz_model_out['model_map'][ith_attri].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri], fixed_attributes)
                #writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'templ' in ith_attri:
                sdf_values = xy_model_out['template'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)
            elif 'overall' in ith_attri:
                sdf_values = xy_model_out['template'].detach() #yz_model_out['model_out'].detach()
                sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_displacement(sdf_values, maps[ith_attri], fixed_attributes)
                # writer.add_figure(prefix + 'yz_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)
                writer.add_figure(prefix + 'xy_sdf_slice_displm_for_' + str(ith_attri), fig, global_step=total_steps)



def write_1dsdf_summary(model, model_input, attributes, gt, model_output, writer, total_steps, prefix='train_',device='cpu'):
    slice_coords_1d = torch.linspace(0, 1, 200)[:, None] #dataio.get_mgrid(128)
    #slice_coords =
    #with torch.no_grad():
    model.eval()
    '''
    1. x
    '''
    #yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    x_slice_model_input =  slice_coords_1d.to(device)

    #yz_slice_model_input = slice_coords_2d.to(device)
    fixed_attributes = {}
    for ith_attri in attributes.keys():
        #print(attributes[ith_attri][0])
        fixed_attributes[ith_attri] = attributes[ith_attri][[0]]#[None, :]
    list_attributes = []
    for ith_attri in fixed_attributes.keys():
        #list_attributes.append(fixed_attributes[ith_attri][:, None])
        #fixed_attributes = torch.cat(list_attributes, dim=-1)
        fixed_attributes[ith_attri] = fixed_attributes[ith_attri].repeat(x_slice_model_input.shape[0], ).float()

    x_model_out = model(x_slice_model_input, fixed_attributes)
    sdf_values = x_model_out['model_out'].detach()
    #sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    #print(sdf_values.shape)
    #fig = make_contour_plot(sdf_values, fixed_attributes, mode='lin')
    fig = make_curve(sdf_values, fixed_attributes, model='lin')
    writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)


    if 'model_map' in model_output.keys():
        maps = x_model_out['model_map']
        for ith_attri in maps.keys():
            sdf_values = maps[ith_attri].detach()
            #sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            #print(sdf_values.shape)
            fig = make_curve(sdf_values, fixed_attributes, model='lin')
            writer.add_figure(prefix + 'xy_sdf_slice_attr_map_for_' + str(ith_attri), fig, global_step=total_steps)




def hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    with torch.no_grad():
        hypo_parameters, embedding = model.get_hypo_net_weights(model_input)

        for name, param in hypo_parameters.items():
            writer.add_histogram(prefix + name, param.cpu(), global_step=total_steps)

        writer.add_histogram(prefix + 'latent_code', embedding.cpu(), global_step=total_steps)


