import numpy as np
import torch
import torch.nn.functional as F

import naisr.diff_operators as diff_operators
import naisr.modules as modules


def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}


def image_l1(mask, model_output, gt):
    if mask is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['img']).mean()}
    else:
        return {'img_loss': (mask * torch.abs(model_output['model_out'] - gt['img'])).mean()}


def image_mse_TV_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}


def image_mse_FH_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    img_hessian, status = diff_operators.hessian(rand_output['model_out'],
                                                 rand_output['model_in'])
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}


def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def image_hypernetwork_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_mse(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}


def function_mse(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt['gradients']).pow(2).sum(-1))
    return {'gradients_loss': gradients_loss}


def gradients_color_mse(model_output, gt):
    # compute gradients on the model
    gradients_r = diff_operators.gradient(model_output['model_out'][..., 0], model_output['model_in'])
    gradients_g = diff_operators.gradient(model_output['model_out'][..., 1], model_output['model_in'])
    gradients_b = diff_operators.gradient(model_output['model_out'][..., 2], model_output['model_in'])
    gradients = torch.cat((gradients_r, gradients_g, gradients_b), dim=-1)
    # compare them with the ground-truth
    weights = torch.tensor([1e1, 1e1, 1., 1., 1e1, 1e1]).cuda()
    gradients_loss = torch.mean((weights * (gradients[0:2] - gt['gradients']).pow(2)).sum(-1))
    return {'gradients_loss': gradients_loss}


def laplace_mse(model_output, gt):
    # compute laplacian on the model
    laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    # compare them with the ground truth
    laplace_loss = torch.mean((laplace - gt['laplace']) ** 2)
    return {'laplace_loss': laplace_loss}


def wave_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']
    x = model_output['model_in']  # (meta_batch_size, num_points, 3)
    y = model_output['model_out']  # (meta_batch_size, num_points, 1)
    squared_slowness = gt['squared_slowness']
    dirichlet_mask = gt['dirichlet_mask']
    batch_size = x.shape[1]

    du, status = diff_operators.jacobian(y, x)
    dudt = du[..., 0]

    if torch.all(dirichlet_mask):
        diff_constraint_hom = torch.Tensor([0])
    else:
        hess, status = diff_operators.jacobian(du[..., 0, :], x)
        lap = hess[..., 1, 1, None] + hess[..., 2, 2, None]
        dudt2 = hess[..., 0, 0, None]
        diff_constraint_hom = dudt2 - 1 / squared_slowness * lap

    dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]
    neumann = dudt[dirichlet_mask]

    return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 1e1,
            'neumann': torch.abs(neumann).sum() * batch_size / 1e2,
            'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}


def helmholtz_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']

    if 'rec_boundary_values' in gt:
        rec_boundary_values = gt['rec_boundary_values']

    wavenumber = gt['wavenumber'].float()
    x = model_output['model_in']  # (meta_batch_size, num_points, 2)
    y = model_output['model_out']  # (meta_batch_size, num_points, 2)
    squared_slowness = gt['squared_slowness'].repeat(1, 1, y.shape[-1] // 2)
    batch_size = x.shape[1]

    full_waveform_inversion = False
    if 'pretrain' in gt:
        pred_squared_slowness = y[:, :, -1] + 1.
        if torch.all(gt['pretrain'] == -1):
            full_waveform_inversion = True
            pred_squared_slowness = torch.clamp(y[:, :, -1], min=-0.999) + 1.
            squared_slowness_init = torch.stack((torch.ones_like(pred_squared_slowness),
                                                 torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.stack((pred_squared_slowness, torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.where((torch.abs(x[..., 0, None]) > 0.75) | (torch.abs(x[..., 1, None]) > 0.75),
                                           squared_slowness_init, squared_slowness)
        y = y[:, :, :-1]

    du, status = diff_operators.jacobian(y, x)
    dudx1 = du[..., 0]
    dudx2 = du[..., 1]

    a0 = 5.0

    # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
    Lpml = 0.5
    dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
    dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
    dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
    dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

    sx = wavenumber * a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
    sy = wavenumber * a0 * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]

    ex = torch.cat((torch.ones_like(sx), -sx / wavenumber), dim=-1)
    ey = torch.cat((torch.ones_like(sy), -sy / wavenumber), dim=-1)

    A = modules.compl_div(ey, ex).repeat(1, 1, dudx1.shape[-1] // 2)
    B = modules.compl_div(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)
    C = modules.compl_mul(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)

    a, _ = diff_operators.jacobian(modules.compl_mul(A, dudx1), x)
    b, _ = diff_operators.jacobian(modules.compl_mul(B, dudx2), x)

    a = a[..., 0]
    b = b[..., 1]
    c = modules.compl_mul(modules.compl_mul(C, squared_slowness), wavenumber ** 2 * y)

    diff_constraint_hom = a + b + c
    diff_constraint_on = torch.where(source_boundary_values != 0.,
                                     diff_constraint_hom - source_boundary_values,
                                     torch.zeros_like(diff_constraint_hom))
    diff_constraint_off = torch.where(source_boundary_values == 0.,
                                      diff_constraint_hom,
                                      torch.zeros_like(diff_constraint_hom))
    if full_waveform_inversion:
        data_term = torch.where(rec_boundary_values != 0, y - rec_boundary_values, torch.Tensor([0.]).cuda())
    else:
        data_term = torch.Tensor([0.])

        if 'pretrain' in gt:  # we are not trying to solve for velocity
            data_term = pred_squared_slowness - squared_slowness[..., 0]

    return {'diff_constraint_on': torch.abs(diff_constraint_on).sum() * batch_size / 1e3,
            'diff_constraint_off': torch.abs(diff_constraint_off).sum(),
            'data_term': torch.abs(data_term).sum() * batch_size / 1}


def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords, )[:, :, 0:3]

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------

    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e3,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e2}  # 1e1      # 5e1




def loss_anglytical_sdfvf(model_output, gt, stage='train', dict_losses={}):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''

    whether_eikonal = dict_losses['whether_eikonal'] if 'whether_eikonal' in dict_losses.keys() else False
    whether_vec =  dict_losses['whether_vec'] if 'whether_vec' in dict_losses.keys() else False
    whether_jacobian = dict_losses['whether_jacobian'] if 'whether_jacobian' in dict_losses.keys() else False
    whether_hyperelastic = dict_losses['whether_hyperelastic'] if 'whether_hyperelastic' in dict_losses.keys() else False
    whether_bendingenergy = dict_losses['whether_bendingenergy'] if 'whether_bendingenergy' in dict_losses.keys() else False

    #if stage == 'train':
    '''
    get ground-truth of template and target shapes
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_templ_sdf = gt['templ_sdf']
    gt_templ_normals = gt['templ_normals']
    coords_shape = model_output['model_in']
    '''
    get input of the network, which is 3D coordinates with covariates
    '''
    if 'all_input' in model_output.keys():
        coords = model_output['all_input'] # 3D coordinates with covariates
    else:
        coords = model_output['model_in'] # 3D coordinates

    '''
    get predicted data
    '''
    pred_sdf = model_output['model_out'] # predicted sdf value of target shape
    pred_template = model_output['template'] # predicted
    pred_vect_fields = model_output['vec_fields'] # predicted sdf value of the template shape
    pred_evol = model_output['model_map']
    '''
    1. SDF for target shape
    '''
    #l2_loss = torch.nn.MSELoss()
    #sdf_constraint_target = l2_loss(pred_sdf, gt_sdf)
    bce_loss =torch.nn.HingeEmbeddingLoss(margin=0.1)# torch.nn.BCEWithLogitsLoss() #torch.nn.HingeEmbeddingLoss(margin=0.1)#torch.nn.BCEWithLogitsLoss()#
    sdf_constraint_target = bce_loss(pred_sdf, (gt_sdf > 0).float() * (-2)+1) #torch.relu(-1 * pred_sdf * torch.sign(gt_sdf)).mean()  #bce_loss(pred_sdf, (gt_sdf > 0).float() * (-2)+1)
    #bce_loss = torch.nn.BCELoss()
    #sdf_constraint_target = bce_loss(pred_sdf / 2 + 0.5, (gt_sdf > 0).float())
    #loss = torch.nn.MarginRankingLoss(margin=0.01)
    #shuffle_idx = np.random.randint(0, pred_sdf.shape[1], size=pred_sdf.shape[1])
    #sdf_constraint_target += loss(pred_sdf[:, shuffle_idx], pred_sdf, torch.sign((gt_sdf[:, shuffle_idx] > gt_sdf).float()-0.5))

    '''
    2. SDF for template shape
    '''
    #sdf_constraint_template = l2_loss(pred_template, gt_templ_sdf)
    #sdf_constraint_template = bce_loss(pred_template/2+0.5, (gt_templ_sdf > 0).float())
    sdf_constraint_template = bce_loss(pred_template, (gt_templ_sdf > 0).float() * (-2) + 1) #l2_loss(pred_template, gt_templ_sdf) #bce_loss(pred_template, (gt_templ_sdf > 0).float() * (-2) + 1)
    #sdf_constraint_template = torch.relu(-1 * pred_template * torch.sign(gt_templ_sdf)).mean()
    #loss = torch.nn.MarginRankingLoss(margin=0.01)
    #shuffle_idx = np.random.randint(0, pred_sdf.shape[1], size=pred_sdf.shape[1])
    #sdf_constraint_template += loss(pred_template[:, shuffle_idx], pred_template, torch.sign((gt_templ_sdf[:, shuffle_idx] > gt_templ_sdf).float()-0.5))
    #sdf_constraint_template = bce_loss(pred_template, (gt_templ_sdf>0).float())
    #sdf_constraint_template = torch.relu(-1 * pred_template * torch.sign(gt_templ_sdf)) #bce_loss(pred_template, (gt_templ_sdf > 0).float() * (-2)+1)

    '''
    3. evolution
    '''

    sdf_evo = 0
    #evo_loss = torch.nn.BCELoss()
    for i_key in pred_evol.keys():
        if '_grad' not in i_key and 'initial' not in i_key :
            #print(i_key)
            #sdf_evo += l2_loss(pred_evol[i_key], gt_sdf)
            sdf_evo += bce_loss(pred_evol[i_key],  (gt_sdf > 0).float() * (-2)+1)
            #sdf_evo += bce_loss(pred_evol[i_key], (gt_templ_sdf > 0).float() * (-2) + 1)
            #sdf_evo += bce_loss(pred_evol[i_key]/2+0.5, (gt_sdf > 0).float())
    '''
    3. Vector Field constraint: boundary condition
    '''
    templ_constraint = 0
    for i_key in pred_vect_fields.keys():
        if 'templ' in i_key:
            current_vec_field = pred_vect_fields[i_key]
            current_templ_constraint = current_vec_field.norm(dim=-1)
            templ_constraint += current_templ_constraint.mean()

    # -----------------
    losses = {'sdf': sdf_constraint_target.mean() *1e3,
              'sdf_templ': sdf_constraint_template.mean()*1e3,
              'templ_constraint': templ_constraint*1e1,
                }
    if sdf_evo != 0:
        losses.update({'sdf_evo': sdf_evo*1e2})
    if 'loss_lip' in model_output.keys():
        losses.update({'loss_lip': model_output['loss_lip']  * 1e-3})
        #losses.update({'loss_lip_initial': model_output['loss_lip_initial'] * 1e-3})
              #'templ_constraint': templ_constraint}
              #'sdf_templ': sdf_constraint_template.mean()*1e3,}



    if whether_eikonal:
        '''
        4. Vector Field constraint: Eikonal Constraint
        '''
        # PDE constraints
        def eikonal_constraint(gradient):
            return (gradient.norm(dim=-1) - 1.) ** 2
        def get_unsigned_distance_from_vec_field(vec_field):
            df_of_vec_field = vec_field.norm(dim=-1)
            return df_of_vec_field

        def get_velocity(df_of_vec_field, coords, dim):
            grad = diff_operators.gradient(df_of_vec_field, coords)[:, :, 0:dim]
            vec = grad(df_of_vec_field, coords)[:, :, 0:dim]
            return vec
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_norm_constraint = {}
    
        for i_key in pred_vect_fields.keys():
            #if '_templ' not in i_key:
            if 'overall' in i_key:
                current_vec_field = pred_vect_fields[i_key]
                df_of_vec_field = get_unsigned_distance_from_vec_field(current_vec_field)
                current_velocity = get_velocity(df_of_vec_field, coords, dim=model_output['model_in'].shape[-1])
                grad_constraint = eikonal_constraint(current_velocity).unsqueeze(-1)
                velocity_norm_constraint[i_key] = grad_constraint.mean()
        '''
        gradient = diff_operators.gradient(pred_sdf, coords_shape)[:, :, 0:coords_shape.shape[-1]]
        gradient_templ = diff_operators.gradient(pred_template, coords_shape)[:, :, 0:coords_shape.shape[-1]]
        eikonal_loss = {}
        eikonal_loss['eikonal'] = eikonal_constraint(gradient).mean() #* 0.01
        eikonal_loss['eikonal_templ'] = eikonal_constraint(gradient_templ).mean()
        losses.update(eikonal_loss)
            #'v_mean': velocity_mean_constraint,
            #'v_norm': velocity_norm_constraint}
            #'normal_constraint':normal_constraint.mean()}  # 1e1      # 5e1

    if whether_vec:
        '''
        5. Vector Field constraint: Velocity Constraint
        '''
        def velocity_field_constraint(predicted_sdf, coords):
            velocity = diff_operators.gradient(predicted_sdf, coords)**2#.index_select(-1, torch.arange(coords_shape.shape[-1]).to(coords_shape.device))** 2
            velocity_field_constraint = torch.sum(velocity, dim=-1).mean()
            # vec = diff_operators.laplace(predicted_sdf, coords)[:, :, 0:dim]
            return velocity_field_constraint

        pred_vect_fields = model_output['vec_fields']
        velocity_vec_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key and 'implicit' not in i_key:
                current_constraint = 0
                for i in range(pred_vect_fields[i_key].shape[-1]):
                    current_constraint  += \
                        velocity_field_constraint(pred_vect_fields[i_key][..., i], coords)
                velocity_vec_constraint['gradient_' + i_key] = current_constraint*100
        losses.update(velocity_vec_constraint)


    if whether_jacobian:
        '''
        6. Vector Field constraint: Jacobian Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_jacobian_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_jacobian_loss(coords, pred_vect_fields[i_key], batch_size=coords.shape[0])
                velocity_jacobian_constraint['jacobian_' + i_key] = current_constraint*10
        losses.update(velocity_jacobian_constraint)

    if whether_hyperelastic:
        '''
        7. Vector Field constraint: HyperElastic Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_HE_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_hyper_elastic_loss(coords, pred_vect_fields[i_key], batch_size=coords.shape[0], alpha_l=1, alpha_a=1, alpha_v=1)
                velocity_HE_constraint['hyper_elastic_' + i_key] = current_constraint
        losses.update(velocity_HE_constraint)

    if whether_bendingenergy:
        '''
        8. Vector Field constraint: Bending Energy Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_BE_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_bending_energy(coords, pred_vect_fields[i_key], batch_size=coords.shape[0])
                velocity_BE_constraint['bending_energy_' + i_key] = current_constraint
        losses.update(velocity_BE_constraint)

    return losses



def loss_pointcloud_sdf(model_output, gt, batch_vecs, epoch, dict_losses={}):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    whether_small_mean_deform = dict_losses['whether_small_mean_deform'] if 'whether_small_mean_deform' in dict_losses.keys() else False
    whether_small_ind_deform=  dict_losses['whether_small_ind_deform'] if 'whether_small_ind_deform' in dict_losses.keys() else False
    whether_eikonal = dict_losses['whether_eikonal'] if 'whether_eikonal' in dict_losses.keys() else False
    whether_vec =  dict_losses['whether_vec'] if 'whether_vec' in dict_losses.keys() else False
    whether_jacobian = dict_losses['whether_jacobian'] if 'whether_jacobian' in dict_losses.keys() else False
    whether_hyperelastic = dict_losses['whether_hyperelastic'] if 'whether_hyperelastic' in dict_losses.keys() else False
    whether_bendingenergy = dict_losses['whether_bendingenergy'] if 'whether_bendingenergy' in dict_losses.keys() else False
    whether_code_regularization = dict_losses['whether_code_regularization'] if 'whether_code_regularization' in dict_losses.keys() else False
    whether_vad = dict_losses['whether_vad'] if 'whether_vad' in dict_losses.keys() else False
    whether_disentangle = dict_losses['whether_disentangle'] if 'whether_disentangle' in dict_losses.keys() else False
    whether_inv= dict_losses['whether_inv'] if 'whether_inv' in dict_losses.keys() else False
    whether_inv_l2= dict_losses['whether_inv_l2'] if 'whether_inv_l2' in dict_losses.keys() else False

    '''
    get ground-truth of template and target shapes
    '''
    gt_sdf = gt['sdf']
    #if 'normal' in list(gt.keys()):
    gt_normals = gt['normal']

    '''
    get input of the network, which is 3D coordinates with covariates
    '''
    if 'all_input' in model_output.keys():
        model_input = model_output['all_input'] # 3D coordinates with covariates
    else:
        model_input = model_output['model_in'] # 3D coordinates
    coords = model_output['model_in']

    '''
    get predicted data
    '''
    pred_sdf = model_output['model_out'] # predicted sdf value of target shape

    '''
    1. SDF for target shape
    '''
    l1_loss = torch.nn.L1Loss(reduction="mean")
    gradient = diff_operators.gradient(pred_sdf, model_input)[:, :, 0:(coords.shape[-1])] #-1)]
    #sdf_constraint_target = l1_loss(pred_sdf[:, pred_sdf.shape[1]//6::, :], gt_sdf[:, pred_sdf.shape[1]//6::, :])  #torch.where(gt_sdf > -9, torch.abs(pred_sdf - gt_sdf), torch.zeros_like(pred_sdf))
    sdf_constraint_target = l1_loss(pred_sdf[torch.abs(gt_sdf)<1e-3], gt_sdf[torch.abs(gt_sdf)<1e-3])  # torch.where(gt_sdf > -9, torch.abs(pred_sdf - gt_sdf), torch.zeros_like(pred_sdf))

    normal_constraint = ( 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None])[torch.abs(gt_sdf[:,:,0])<1e-3, :]

    inter_constraint = l1_loss(pred_sdf[torch.abs(gt_sdf) >1e-3], gt_sdf[torch.abs(gt_sdf) > 1e-3])
    # -----------------
    losses = {'sdf': sdf_constraint_target * 3e2,
              'normal_constraint': normal_constraint.mean() * 1e2,
              'inter_constraint': inter_constraint.mean() * 1e3,
              #'neg_constraint': neg_constraint.mean() * 1e1
              }

    if whether_code_regularization:
        l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=-1)**2) / batch_vecs.shape[-1]
        reg_loss = (1e-4 * min(1, epoch / 100) * l2_size_loss * 1e4
                   ) / pred_sdf.shape[0] / pred_sdf.shape[1]*2
        losses.update({'code_reg': reg_loss})

    if whether_eikonal:
        '''
        4. Vector Field constraint: Eikonal Constraint
        '''
        # PDE constraints
        def eikonal_constraint(gradient):
            l1_eik = torch.nn.L1Loss(reduction='mean')
            gradient_norm = gradient.norm(dim=-1)
            return l1_eik(gradient_norm, torch.ones_like(gradient_norm).float()) #(gradient.norm(dim=-1) - 1).norm(dim=-1) #(gradient.norm(dim=-1) - 1.) ** 2

        eikonal_loss = {}
        #gradient = diff_operators.gradient(pred_sdf, model_input)[:, :, 0:(coords.shape[-1])] #-1)]
        eikonal_loss['eikonal'] = eikonal_constraint(gradient)  * 1e2
        losses.update(eikonal_loss )

    #if whether_vad:
    #    z_var = torch.full_like(batch_vecs, args['log_var'])
    #    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + latents[idxes] ** 2 - 1. - z_var) / batch_size
    #    total_loss = loss + args['ratio_kl'] * kl_loss

    if whether_jacobian:
        '''
        4. Vector Field constraint: Jacobian Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_jacobian_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_jacobian_loss(coords, pred_vect_fields[i_key], batch_size=coords.shape[0])
                velocity_jacobian_constraint['jacobian_' + i_key] = current_constraint
        losses.update(velocity_jacobian_constraint)

    if whether_vec:
        '''
        5. Vector Field constraint: Velocity Constraint
        '''

        def velocity_field_constraint(predicted_sdf, coords):
            velocity = diff_operators.gradient(predicted_sdf, coords)** 2
            velocity_field_constraint = torch.sum(velocity, dim=-1)
            # vec = diff_operators.laplace(predicted_sdf, coords)[:, :, 0:dim]
            return velocity_field_constraint

        pred_vec_fields = model_output['vec_fields']
        names = pred_vec_fields.keys()
        velocity_vec_constraint = {}
        velocity_cov_constraint = {}
        velocity_embd_constraint = {}

        for name, arr_vf in pred_vec_fields.items():

            if  'overall' not in name and ('padding' not in name):
                current_constraint_vec = 0

                for i in range(arr_vf.shape[-1]):
                    current_constraint_vec += velocity_field_constraint(arr_vf[..., i], model_input)
                velocity_vec_constraint['gradient_vec_' + name] = current_constraint_vec[torch.abs(gt_sdf[:,:,0])>0.1, :]* 1e1
                #velocity_cov_constraint['gradient_' + name] = current_constraint_cov

        '''
        current_constraint_cov = 0
        for i in range(3):
            current_constraint_cov += velocity_field_constraint(pred_vec_fields['overall'], model_output['covaraites'])
        velocity_cov_constraint['gradient_cov_overall'] = current_constraint_cov[:, pred_sdf.shape[1]//6::]
        '''
        '''
        current_constraint_vec_embd = 0
        for i in range(3):
            current_constraint_vec_embd  += torch.abs(diff_operators.gradient(pred_vec_fields['overall'][..., i], model_output['embedding']))
            #current_constraint_embd += velocity_field_constraint(pred_vec_fields['overall'], model_output['covaraites'])
        current_constraint_sdf_embd = torch.abs(diff_operators.gradient(pred_sdf ,model_output['embedding']))

        velocity_embd_constraint['vec_gradient_embd_overall'] = current_constraint_vec_embd[:, pred_sdf.shape[1] // 4::] * 1e2
        velocity_embd_constraint['sdf_gradient_embd_overall'] = current_constraint_sdf_embd[:, pred_sdf.shape[1] // 4::] * 1e2
        '''
        losses.update(velocity_vec_constraint)
        #losses.update(velocity_embd_constraint)
        #losses.update(velocity_cov_constraint)
        '''
        dict_vec_field = model_output['vec_fields']
        dict_loss_vf = {}
        for name, arr_pred in dict_vec_field.items():
            if ('c_padding' in name) and ('overall' not in name):
                dict_loss_vf['c_padding_' + name] = (arr_pred.norm(dim=-1)**2).mean()
        losses.update(dict_loss_vf)
        '''
        #velocity_cov_constraint = torch.mean(torch.sum(diff_operators.gradient(pred_vec_fields['v'], model_output['covaraites'])**2, dim=-1))
        #losses.update({'velocity_cov_constraint': velocity_cov_constraint*1e2,})



    if whether_vad:
        batch_vecs_mean = batch_vecs['mean']
        batch_vecs_log_var = batch_vecs['log_var']
        kl_loss = -0.5 * torch.sum(-torch.exp(batch_vecs_log_var) - batch_vecs_mean ** 2 + 1. +batch_vecs_log_var)
        kl_loss /= batch_vecs_mean.shape[0]
        kl_loss /= batch_vecs_mean.shape[1]
        losses.update({'kl_loss': kl_loss * 0.1})

    if whether_small_ind_deform:
        '''
        3. Vector Field constraint: boundary condition
        '''

        dict_vec_field = model_output['vec_fields']
        dict_loss_vf = {}
        for name, arr_pred in dict_vec_field.items():
            if ('padding' not in name) and ('overall' not in name):
                dict_loss_vf['vf_size_' + name] = (arr_pred.norm(dim=-1)**2).mean() #* 1e2
        losses.update(dict_loss_vf)

    if whether_small_mean_deform:
        '''
        3. Vector Field constraint: boundary condition
        '''
        dict_vec_field = model_output['vec_fields']
        dict_loss_vf = {}
        for name, arr_pred in dict_vec_field.items():
            if ('padding' not in dict_vec_field):
                dict_loss_vf[name] = arr_pred.norm(dim=-1).mean() #* 1e1
        losses.update(dict_loss_vf)



    if whether_disentangle:
        pred_sdf_z_padding = model_output['model_out_z_padding']
        l1_loss = torch.nn.L1Loss(reduction="mean")
        gradient_z_padding = diff_operators.gradient(pred_sdf_z_padding, model_input)[:, :, 0:(coords.shape[-1])]  # -1)]
        sdf_constraint_z_padding = l1_loss(pred_sdf_z_padding[torch.abs(gt_sdf)<1e-3], gt_sdf[torch.abs(gt_sdf)<1e-3])  # torch.where(gt_sdf > -9, torch.abs(pred_sdf - gt_sdf), torch.zeros_like(pred_sdf))
        normal_constraint_z_padding = (1 - F.cosine_similarity(gradient_z_padding, gt_normals, dim=-1)[..., None])[torch.abs(gt_sdf[:,:,0])<1e-3, :]
        inter_constraint_z_padding = l1_loss(pred_sdf_z_padding[torch.abs(gt_sdf)> 1e-3], gt_sdf[torch.abs(gt_sdf)> 1e-3])
        losses.update({'sdf_z_padding': sdf_constraint_z_padding * 3e2,
                       'normal_constraint_z_padding': normal_constraint_z_padding.mean() * 1e2,
                       'inter_constraint_z_padding': inter_constraint_z_padding.mean() * 1e3,
                      })
        losses.update({'eikonal_z_padding': eikonal_constraint(gradient_z_padding) *1e2})
        '''
        pred_disentangle = model_output['disentangle']
        loss_per_co = {}
        
        for name, value in pred_disentangle.items():
            loss_per_co['sdf_disentangle_' + name] = l1_loss(value[gt_sdf<0.05], gt_sdf[gt_sdf<0.05]).mean() * 3e1
            loss_per_co['intercon_' + name] =  torch.relu(0.05 - value[gt_sdf > 0.05])
            #gradient_per_co = diff_operators.gradient(model_output['disentangle'][name], model_input)[:, :, 0:(coords.shape[-1])]  # -1)]
        
            #loss_per_co['normvec_disentangle_' + name] = (1 - F.cosine_similarity(gradient_per_co, gt_normals, dim=-1)[..., None])[ :, pred_sdf.shape[1] // 2::, :]
            #loss_per_co['eik_'] += eikonal_constraint(gradient_per_co[:, pred_sdf.shape[1] // 2::, :])
        losses.update(loss_per_co)
        '''


    if whether_inv:
        '''
        5. inv
        '''
        '''
        def gradient(disp, time):
            list_of_gradients_xyz = []
            for i in range(disp.shape[-1]):
                velocity = diff_operators.gradient(disp[..., i], time)
                list_of_gradients_xyz.append(velocity)
            return list_of_gradients_xyz
        '''


        pred_vec_fields = model_output['vec_fields']
        velocity_vec_constraint = {}
        l1_loss =torch.nn.L1Loss() # torch.nn.MSELoss() #torch.nn.L1Loss()
        for name, arr_vf in pred_vec_fields.items():
            if '_inv' in name:
                current = l1_loss(pred_vec_fields[name][torch.abs(gt_sdf[:,:,0])<0.01, :], -pred_vec_fields[name[0:-4]][torch.abs(gt_sdf[:,:,0])<0.01, :])
                #current = (1 - F.cosine_similarity(pred_vec_fields[name][torch.abs(gt_sdf[:,:,0])<0.01, :], pred_vec_fields[name[0:-4]][torch.abs(gt_sdf[:,:,0])<0.01, :], dim=-1)[..., None]).mean()

                velocity_vec_constraint['disp_' + name] = current * 10
        losses.update(velocity_vec_constraint)

        '''
        for name, arr_vf in pred_vec_fields.items():
            if '_inv' in name:
                gradient(arr_vf, )
                current = l1_loss(pred_vec_fields[name][torch.abs(gt_sdf[:,:,0])<0.01, :], -pred_vec_fields[name[0:-4]][torch.abs(gt_sdf[:,:,0])<0.01, :])
                velocity_vec_constraint['v' + name] = current*10
        losses.update(velocity_vec_constraint)
        '''

    if whether_inv_l2:
        '''
        5. inv
        '''
        pred_vec_fields = model_output['vec_fields']
        velocity_vec_constraint = {}
        for name, arr_vf in pred_vec_fields.items():
            if '_inv' in name:
                current = torch.norm(pred_vec_fields[name][torch.abs(gt_sdf[:,:,0])<0.01, :] + pred_vec_fields[name[0:-4]][torch.abs(gt_sdf[:,:,0])<0.01, :], dim=-1).mean()
                #current = (1 - F.cosine_similarity(pred_vec_fields[name][torch.abs(gt_sdf[:,:,0])<0.01, :], pred_vec_fields[name[0:-4]][torch.abs(gt_sdf[:,:,0])<0.01, :], dim=-1)[..., None]).mean()

                velocity_vec_constraint['disp_' + name] = current * 10
        losses.update(velocity_vec_constraint)

    return losses


def loss_pointcloud_puresdf(model_output, gt, batch_vecs, epoch, dict_losses={}):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    whether_small_mean_deform = dict_losses[
        'whether_small_mean_deform'] if 'whether_small_mean_deform' in dict_losses.keys() else False
    whether_small_ind_deform = dict_losses[
        'whether_small_ind_deform'] if 'whether_small_ind_deform' in dict_losses.keys() else False
    whether_eikonal = dict_losses['whether_eikonal'] if 'whether_eikonal' in dict_losses.keys() else False
    whether_vec = dict_losses['whether_vec'] if 'whether_vec' in dict_losses.keys() else False
    whether_jacobian = dict_losses['whether_jacobian'] if 'whether_jacobian' in dict_losses.keys() else False
    whether_code_regularization = dict_losses[
        'whether_code_regularization'] if 'whether_code_regularization' in dict_losses.keys() else False
    whether_disentangle = dict_losses['whether_disentangle'] if 'whether_disentangle' in dict_losses.keys() else False
    '''
    get ground-truth of template and target shapes
    '''
    gt_sdf = gt['sdf']

    '''
    get input of the network, which is 3D coordinates with covariates
    '''
    if 'all_input' in model_output.keys():
        model_input = model_output['all_input']  # 3D coordinates with covariates
    else:
        model_input = model_output['model_in']  # 3D coordinates
    coords = model_output['model_in']

    '''
    get predicted data
    '''
    pred_sdf = model_output['model_out']  # predicted sdf value of target shape

    '''
    1. SDF for target shape
    '''
    l1_loss = torch.nn.L1Loss(reduction="mean")
    sdf_constraint_target = l1_loss(pred_sdf, gt_sdf)
    # -----------------
    losses = {'sdf': sdf_constraint_target,}

    if whether_code_regularization:
        l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=-1) ** 2) / batch_vecs.shape[-1]
        reg_loss = (1e-4 * min(1, epoch / 100) * l2_size_loss * 1e4
                    ) / pred_sdf.shape[0] / pred_sdf.shape[1] * 2
        losses.update({'code_reg': reg_loss})

    if whether_jacobian:
        '''
        4. Vector Field constraint: Jacobian Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_jacobian_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_jacobian_loss(coords, pred_vect_fields[i_key], batch_size=coords.shape[0])
                velocity_jacobian_constraint['jacobian_' + i_key] = current_constraint
        losses.update(velocity_jacobian_constraint)

    if whether_vec:
        '''
        5. Vector Field constraint: Velocity Constraint
        '''

        def velocity_field_constraint(predicted_sdf, coords):
            velocity = diff_operators.gradient(predicted_sdf, coords) ** 2
            velocity_field_constraint = torch.sum(velocity, dim=-1)
            # vec = diff_operators.laplace(predicted_sdf, coords)[:, :, 0:dim]
            return velocity_field_constraint

        pred_vec_fields = model_output['vec_fields']
        names = pred_vec_fields.keys()
        velocity_vec_constraint = {}
        velocity_cov_constraint = {}
        velocity_embd_constraint = {}

        for name, arr_vf in pred_vec_fields.items():

            if 'overall' not in name and ('padding' not in name):
                current_constraint_vec = 0

                for i in range(arr_vf.shape[-1]):
                    current_constraint_vec += velocity_field_constraint(arr_vf[..., i], model_input)
                velocity_vec_constraint['gradient_vec_' + name] = current_constraint_vec[...,
                                                                  pred_sdf.shape[1] // 6::] * 1e1
                # velocity_cov_constraint['gradient_' + name] = current_constraint_cov
        '''
        current_constraint_vec_embd = 0
        for i in range(3):
            current_constraint_vec_embd += torch.abs(
                diff_operators.gradient(pred_vec_fields['overall'][..., i], model_output['embedding']))
            # current_constraint_embd += velocity_field_constraint(pred_vec_fields['overall'], model_output['covaraites'])
        current_constraint_sdf_embd = torch.abs(diff_operators.gradient(pred_sdf, model_output['embedding']))

        velocity_embd_constraint['vec_gradient_embd_overall'] = current_constraint_vec_embd[:,
                                                                pred_sdf.shape[1] // 4::] * 1e2
        velocity_embd_constraint['sdf_gradient_embd_overall'] = current_constraint_sdf_embd[:,
                                                                pred_sdf.shape[1] // 4::] * 1e2
        '''
        losses.update(velocity_vec_constraint)


    if whether_small_ind_deform:
        '''
        3. Vector Field constraint: boundary condition
        '''

        dict_vec_field = model_output['vec_fields']
        dict_loss_vf = {}
        for name, arr_pred in dict_vec_field.items():
            if ('padding' not in name) and ('overall' not in name):
                dict_loss_vf['vf_size_' + name] = (arr_pred.norm(dim=-1) ** 2).mean()  # * 1e2
        losses.update(dict_loss_vf)

    if whether_small_mean_deform:
        '''
        3. Vector Field constraint: boundary condition
        '''
        dict_vec_field = model_output['vec_fields']
        dict_loss_vf = {}
        for name, arr_pred in dict_vec_field.items():
            if ('padding' not in dict_vec_field):
                dict_loss_vf[name] = arr_pred.norm(dim=-1).mean()  # * 1e1
        losses.update(dict_loss_vf)

    if whether_disentangle:
        pred_sdf_z_padding = model_output['model_out_z_padding']
        l1_loss = torch.nn.L1Loss(reduction="mean")
        sdf_constraint_z_padding = l1_loss(pred_sdf_z_padding[:, pred_sdf.shape[1] // 6::, :],
                                           gt_sdf[:, pred_sdf.shape[1] // 6::,
                                           :])
        inter_constraint_z_padding = torch.relu(0.05 - pred_sdf_z_padding[gt_sdf > 0.05])
        losses.update({'sdf_z_padding': sdf_constraint_z_padding * 3e2,
                       'inter_constraint_z_padding': inter_constraint_z_padding.mean() * 1e2,
                       })

    return losses

def loss_pointcloud_sdfvf_autotempl(model_output, gt, stage='train', dict_losses={}):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    whether_small_mean_deform = dict_losses['whether_small_mean_deform'] if 'whether_small_mean_deform' in dict_losses.keys() else False
    whether_small_ind_deform=  dict_losses['whether_small_ind_deform'] if 'whether_small_ind_deform' in dict_losses.keys() else False
    whether_eikonal = dict_losses['whether_eikonal'] if 'whether_eikonal' in dict_losses.keys() else False
    whether_vec =  dict_losses['whether_vec'] if 'whether_vec' in dict_losses.keys() else False
    whether_jacobian = dict_losses['whether_jacobian'] if 'whether_jacobian' in dict_losses.keys() else False
    whether_hyperelastic = dict_losses['whether_hyperelastic'] if 'whether_hyperelastic' in dict_losses.keys() else False
    whether_bendingenergy = dict_losses['whether_bendingenergy'] if 'whether_bendingenergy' in dict_losses.keys() else False

    '''
    get ground-truth of template and target shapes
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normal']
    #gt_template_sdf = gt['template_sdf']
    #gt_template_normals = gt['template_normal']
    #gt_templ_sdf = gt['templ_sdf']

    '''
    get input of the network, which is 3D coordinates with covariates
    '''
    if 'all_input' in model_output.keys():
        coords = model_output['all_input'] # 3D coordinates with covariates
    else:
        coords = model_output['model_in'] # 3D coordinates
    #coords_shape = model_output['model_in']
    template_coords = model_output['template_in']

    '''
    get predicted data
    '''
    pred_sdf = model_output['model_out'] # predicted sdf value of target shape
    pred_template = model_output['template'] # predicted
    pred_vect_fields = model_output['vec_fields'] # predicted sdf value of the template shape

    '''
    1. SDF for target shape
    '''
    gradient = diff_operators.gradient(pred_sdf, coords)[:, :, 0:(template_coords.shape[-1]-1)]
    gradient_with_temp = diff_operators.gradient(pred_template, template_coords)[:, :, 0:(template_coords.shape[-1] -1)]

    sdf_constraint_target = torch.where(gt_sdf != -1, torch.abs(pred_sdf), torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(- 1e1* torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))

    template_sdf_constraint_target = torch.where(gt_sdf != -1, torch.abs(pred_template), torch.zeros_like(pred_template))
    template_inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_template), torch.exp(- 1e1* torch.abs(pred_template)))
    template_normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient_with_temp, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))


    '''
    2. SDF for template shape
    '''
    #sdf_constraint_template = torch.where(gt_templ_sdf != -1, pred_template, torch.zeros_like(pred_template))


    # -----------------
    losses = {'sdf': sdf_constraint_target.mean() * 3e2,  #*1e2,
              'inter_consistent': inter_constraint.mean() * 1e2,
              'normal_constraint': normal_constraint.mean() * 1e2,
              'templ_sdf': template_sdf_constraint_target.mean() * 3e2,  # *1e2,
              'templ_inter_consistent': template_inter_constraint.mean() * 1e2,
              'templ_normal_constraint': template_normal_constraint.mean() * 1e2,

              #'templ_constraint': templ_constraint*1e3,
              #'sdf_templ': sdf_constraint_template.mean()*1e3,
              }


    if whether_eikonal:
        '''
        4. Vector Field constraint: Eikonal Constraint
        '''
        # PDE constraints
        def eikonal_constraint(gradient):
            return torch.abs(gradient.norm(dim=-1) - 1.) #(gradient.norm(dim=-1) - 1).norm(dim=-1) #(gradient.norm(dim=-1) - 1.) ** 2


        eikonal_loss = {}
        gradient_eik = diff_operators.gradient(pred_sdf, coords)[:, :, 0:(template_coords.shape[-1]-1)]
        eikonal_loss['eikonal'] = eikonal_constraint(gradient_eik).mean()  * 5e1
        gradient_with_temp = diff_operators.gradient(pred_template,template_coords)[:, :, 0:(template_coords.shape[-1]-1)]
        eikonal_loss['eikonal_templ'] = eikonal_constraint(gradient_with_temp).mean()  * 5e1
        #eikonal_loss['template_normal'] = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient_with_temp[...,[0,1]], gt_normals, dim=-1)[..., None],
        #                                torch.zeros_like(gradient[..., :1]))
        losses.update(eikonal_loss )

    if whether_small_ind_deform:
        '''
        3. Vector Field constraint: boundary condition
        '''
        vec_constraint = 0
        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_vec_field = pred_vect_fields[i_key]
                current_vec_constraint = current_vec_field.norm(dim=-1)
                vec_constraint += current_vec_constraint.mean()
        losses.update({'vec_field': vec_constraint  })


    if whether_small_mean_deform:
        '''
        3. Vector Field constraint: boundary condition
        '''
        vec_constraint = 0
        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_vec_field = pred_vect_fields[i_key]
                current_vec_constraint = torch.mean(current_vec_field, dim=[0, 1]).norm(dim=-1)
                vec_constraint += current_vec_constraint.mean()
        losses.update({'mean_vec_field': vec_constraint })

    if whether_jacobian:
        '''
        4. Vector Field constraint: Jacobian Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_jacobian_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_jacobian_loss(coords, pred_vect_fields[i_key], batch_size=coords.shape[0])
                velocity_jacobian_constraint['jacobian_' + i_key] = current_constraint
        losses.update(velocity_jacobian_constraint)

    if whether_vec:
        '''
        5. Vector Field constraint: Velocity Constraint
        '''
        def velocity_field_constraint(predicted_sdf, coords):
            velocity = diff_operators.gradient(predicted_sdf, coords).index_select(-1, torch.arange(template_coords.shape[-1]-1).to(template_coords.device))** 2
            velocity_field_constraint = torch.sum(velocity, dim=-1).mean()
            # vec = diff_operators.laplace(predicted_sdf, coords)[:, :, 0:dim]
            return velocity_field_constraint

        pred_vect_fields = model_output['vec_fields']
        velocity_vec_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key and 'implicit' not in i_key:
                current_constraint = 0
                for i in range(pred_vect_fields[i_key].shape[-1]):
                    current_constraint  += \
                        velocity_field_constraint(pred_vect_fields[i_key][..., i], coords)
                velocity_vec_constraint['gradient_' + i_key] = current_constraint * 1e1
        losses.update(velocity_vec_constraint)

    return losses





def loss_pointcloud_sdfvf(model_output, gt, stage='train', dict_losses={}):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    whether_small_mean_deform = dict_losses['whether_small_mean_deform'] if 'whether_small_mean_deform' in dict_losses.keys() else False
    whether_small_ind_deform=  dict_losses['whether_small_ind_deform'] if 'whether_small_ind_deform' in dict_losses.keys() else False
    whether_eikonal = dict_losses['whether_eikonal'] if 'whether_eikonal' in dict_losses.keys() else False
    whether_vec =  dict_losses['whether_vec'] if 'whether_vec' in dict_losses.keys() else False
    whether_jacobian = dict_losses['whether_jacobian'] if 'whether_jacobian' in dict_losses.keys() else False
    whether_hyperelastic = dict_losses['whether_hyperelastic'] if 'whether_hyperelastic' in dict_losses.keys() else False
    whether_bendingenergy = dict_losses['whether_bendingenergy'] if 'whether_bendingenergy' in dict_losses.keys() else False

    '''
    get ground-truth of template and target shapes
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normal']
    gt_template_sdf = gt['template_sdf']
    gt_template_normals = gt['template_normal']
    #gt_templ_sdf = gt['templ_sdf']

    '''
    get input of the network, which is 3D coordinates with covariates
    '''
    if 'all_input' in model_output.keys():
        coords = model_output['all_input'] # 3D coordinates with covariates
    else:
        coords = model_output['model_in'] # 3D coordinates
    #coords_shape = model_output['model_in']
    template_coords = model_output['template_in']

    '''
    get predicted data
    '''
    pred_sdf = model_output['model_out'] # predicted sdf value of target shape
    pred_template = model_output['template'] # predicted
    pred_vect_fields = model_output['vec_fields'] # predicted sdf value of the template shape

    '''
    1. SDF for target shape
    '''
    gradient_with_temp = diff_operators.gradient(pred_template, template_coords)[:, :, 0:3] # -1)]
    template_sdf_constraint_target = torch.abs(pred_template -  gt_template_sdf)[:, 0:5000, :] #torch.where(gt_sdf > -9, torch.abs(pred_template -  gt_template_sdf), torch.zeros_like(pred_template))
    template_normal_constraint = (1 - F.cosine_similarity(gradient_with_temp, gt_template_normals, dim=-1)[..., None])[:, 5000:10000, :] #torch.where(gt_template_sdf ==0, 1 - F.cosine_similarity(gradient_with_temp, gt_template_normals, dim=-1)[..., None],torch.zeros_like(gradient[..., :1]))

    '''
    1. SDF for target shape
    '''
    gradient = diff_operators.gradient(pred_sdf, coords)[:, :, 0:3] #-1)]
    sdf_constraint_target = torch.abs(pred_sdf - gt_sdf)[:, 0:5000, :]  #torch.where(gt_sdf > -9, torch.abs(pred_sdf - gt_sdf), torch.zeros_like(pred_sdf))
    normal_constraint = ( 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None])[:, 5000:10000, :]

    # -----------------
    losses = {'sdf': sdf_constraint_target.mean() * 3e2,
              'normal_constraint': normal_constraint.mean() * 1e1,
              'templ_sdf': template_sdf_constraint_target.mean() * 3e2,
              'templ_normal_constraint': template_normal_constraint.mean() * 1e1,
              }

    '''
    2. SDF for template shape
    '''
    #sdf_constraint_template = torch.where(gt_templ_sdf != -1, pred_template, torch.zeros_like(pred_template))



    # -----------------
    '''
    losses = {'sdf': sdf_constraint_target.mean() * 3e3,  #*1e2,
              #'inter_consistent': inter_constraint.mean() * 1e3,
              'normal_constraint': normal_constraint.mean() * 1e3,
              'templ_sdf': template_sdf_constraint_target.mean() * 3e3,  # *1e2,
              #'templ_inter_consistent': template_inter_constraint.mean() * 1e3,
              'templ_normal_constraint': template_normal_constraint.mean() * 1e3,

              #'templ_constraint': templ_constraint*1e3,
              #'sdf_templ': sdf_constraint_template.mean()*1e3,
              }
    '''
    if whether_eikonal:
        '''
        4. Vector Field constraint: Eikonal Constraint
        '''
        # PDE constraints
        def eikonal_constraint(gradient):
            return torch.abs(gradient.norm(dim=-1) - 1.) #(gradient.norm(dim=-1) - 1).norm(dim=-1) #(gradient.norm(dim=-1) - 1.) ** 2


        eikonal_loss = {}
        #gradient_eik = diff_operators.gradient(pred_sdf, coords)[:, :, 0:(template_coords.shape[-1]-1)]
        #eikonal_loss['eikonal'] = eikonal_constraint(gradient_eik).mean()  * 5e1
        gradient_with_temp = diff_operators.gradient(pred_template,template_coords)[:, :, 0:(template_coords.shape[-1])] #-1)]
        eikonal_loss['eikonal_templ'] = eikonal_constraint(gradient_with_temp).mean()  * 5
        #eikonal_loss['template_normal'] = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient_with_temp[...,[0,1]], gt_normals, dim=-1)[..., None],
        #                                torch.zeros_like(gradient[..., :1]))
        losses.update(eikonal_loss )

    if whether_small_ind_deform:
        '''
        3. Vector Field constraint: boundary condition
        '''
        vec_constraint = 0
        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_vec_field = pred_vect_fields[i_key]
                current_vec_constraint = current_vec_field.norm(dim=-1)
                vec_constraint += current_vec_constraint.mean()
        losses.update({'vec_field': vec_constraint * 1e2 })


    if whether_small_mean_deform:
        '''
        3. Vector Field constraint: boundary condition
        '''
        vec_constraint = 0
        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_vec_field = pred_vect_fields[i_key]
                current_vec_constraint = torch.mean(current_vec_field, dim=[0, 1]).norm(dim=-1)
                vec_constraint += current_vec_constraint.mean()
        losses.update({'mean_vec_field': vec_constraint })

    if whether_jacobian:
        '''
        4. Vector Field constraint: Jacobian Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_jacobian_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_jacobian_loss(coords, pred_vect_fields[i_key], batch_size=coords.shape[0])
                velocity_jacobian_constraint['jacobian_' + i_key] = current_constraint
        losses.update(velocity_jacobian_constraint)

    if whether_vec:
        '''
        5. Vector Field constraint: Velocity Constraint
        '''
        def velocity_field_constraint(predicted_sdf, coords):
            velocity = diff_operators.gradient(predicted_sdf, coords).index_select(-1, torch.arange(template_coords.shape[-1]).to(template_coords.device))** 2
            velocity_field_constraint = torch.sum(velocity, dim=-1).mean()
            # vec = diff_operators.laplace(predicted_sdf, coords)[:, :, 0:dim]
            return velocity_field_constraint

        pred_vect_fields = model_output['vec_fields']
        velocity_vec_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key and 'implicit' not in i_key:
                current_constraint = 0
                for i in range(pred_vect_fields[i_key].shape[-1]):
                    current_constraint  += \
                        velocity_field_constraint(pred_vect_fields[i_key][..., i], coords)
                velocity_vec_constraint['gradient_' + i_key] = current_constraint * 1e2
        losses.update(velocity_vec_constraint)

    return losses




def loss_anglytical_sdf(model_output, gt, stage='train'):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    if stage == 'train':
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        if 'all_input' in model_output.keys():
            coords = model_output['all_input']
        else:
            coords = model_output['model_in']
        pred_sdf = model_output['model_out']

        gradient = diff_operators.gradient(pred_sdf, coords, )[:, :, 0:model_output['model_in'].shape[-1]]
        #normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
        #                                torch.zeros_like(gradient[..., :1]))

        l2_loss = torch.nn.MSELoss()
        sdf_constraint = l2_loss(pred_sdf, gt_sdf)

        #bce_loss = torch.nn.BCELoss()
        #m = torch.nn.Sigmoid()
        #sdf_constraint = bce_loss(m(pred_sdf), (gt_sdf > 0).float())

        #gradient_loss =  torch.sqrt(torch.sum(torch.square(gradient), dim=-1)).mean()
        # PDE constraints
        #def eikonal_constraint(gradient):
        #    return (gradient.norm(dim=-1) - 1.) ** 2
        #grad_constraint = eikonal_constraint(gradient).unsqueeze(-1)

        # Exp      # Lapl
        # -----------------
        return {'sdf': sdf_constraint.mean()*10,}
                #'normal_constraint':normal_constraint.mean()}  # 1e1      # 5e1
    else:
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        coords = model_output['model_in']
        pred_sdf = model_output['model_out']#

        l2_loss = torch.nn.MSELoss()
        sdf_constraint = l2_loss(pred_sdf, gt_sdf)

        #bce_loss = torch.nn.BCELoss()
        #m = torch.nn.Sigmoid()
        #sdf_constraint = bce_loss(m(pred_sdf), (gt_sdf > 0).float())

        # Exp      # Lapl
        # -----------------
        return {'sdf': sdf_constraint.mean(), }  # 1e1      # 5e1



def loss_anglytical_1dsdf(model_output, gt, stage='train'):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''

    if 'all_input' in model_output.keys():
        coords = model_output['all_input']
    else:
        coords = model_output['model_in']

    if stage == 'train':
        gt_sdf = gt['sdf'][:, None]

        pred_sdf = model_output['model_out']
        l2_loss = torch.nn.MSELoss()
        sdf_constraint = l2_loss(pred_sdf, gt_sdf)
        gradient = diff_operators.gradient(pred_sdf, coords, )[:, [0]]
        grad_constraint_z = gradient.norm(dim=-1)
        return {'sdf': sdf_constraint,}
                #'grad_constraint': grad_constraint_z.mean()*0.1}
    else:
        gt_sdf = gt['sdf'][:, None]
        pred_sdf = model_output['model_out']#
        l2_loss = torch.nn.MSELoss()
        sdf_constraint = l2_loss(pred_sdf, gt_sdf)
        gradient = diff_operators.gradient(pred_sdf, coords, )[:, [0]]
        grad_constraint_z = gradient.norm(dim=-1)
        return {'sdf': sdf_constraint,}
                #'grad_constraint': grad_constraint_z.mean()*0.1}  # 1e1      # 5e1


def loss_anglytical_probsdf(model_output, gt, stage='train'):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    if stage == 'train':
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        if 'all_input' in model_output.keys():
            coords = model_output['all_input']
        else:
            coords = model_output['model_in']
        pred_sdf = model_output['model_out']

        gradient = diff_operators.gradient(pred_sdf, coords, )[:, :, 0:3]
        normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                       torch.zeros_like(gradient[..., :1]))

        l2_loss = torch.nn.MSELoss()
        sdf_constraint = l2_loss(pred_sdf, gt_sdf)
        #import numpy as np
        #a= np.sqrt(2 * np.pi)
        #prob_pred_sdf = torch.exp(-pred_sdf**2)
        #prob_gt_sdf = torch.exp( -gt_sdf ** 2)

        loss_existance = l2_loss(model_output['model_existence'], torch.abs(gt_sdf))# + l2_loss(model_output['model_existence'], prob_pred_sdf)
        # Exp      # Lapl
        # -----------------
        return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,
                'normal_constraint': normal_constraint.mean() * 1e3,
                'existance': loss_existance * 1e3}  # 1e1      # 5e1
    else:
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        coords = model_output['model_in']
        pred_sdf = model_output['model_out']



        l2_loss = torch.nn.MSELoss()
        sdf_constraint = l2_loss(pred_sdf, gt_sdf)

        # Exp      # Lapl
        # -----------------
        return {'sdf': torch.abs(sdf_constraint).mean(), }  # 1e1      # 5e1

# inter = 3e3 for ReLU-PE


def loss_anglytical_sdfvf_bs(model_output, gt, stage='train', dict_losses={}):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''

    '''
    get ground-truth of template and target shapes
    '''
    gt_sdf = gt['sdf']
    coords_shape = model_output['model_in']
    '''
    get input of the network, which is 3D coordinates with covariates
    '''
    if 'all_input' in model_output.keys():
        coords = model_output['all_input']  # 3D coordinates with covariates
    else:
        coords = model_output['model_in']  # 3D coordinates

    '''
    get predicted data
    '''
    pred_sdf = model_output['model_out']

    '''
    1. SDF for target shape
    '''
    #l2_loss = torch.nn.MSELoss()
    #sdf_constraint_target = l2_loss(pred_sdf, gt_sdf)
    bce_loss = torch.nn.HingeEmbeddingLoss(
        margin=0.1)  # torch.nn.BCEWithLogitsLoss() #torch.nn.HingeEmbeddingLoss(margin=0.1)#torch.nn.BCEWithLogitsLoss()#
    sdf_constraint_target = bce_loss(pred_sdf, (gt_sdf > 0).float() * (-2) + 1)  # torch.relu(-1 * pred_sdf * torch.sign(gt_sdf)).mean()  #bce_loss(pred_sdf, (gt_sdf > 0).float() * (-2)+1)
    # bce_loss = torch.nn.BCELoss()
    # sdf_constraint_target = bce_loss(pred_sdf / 2 + 0.5, (gt_sdf > 0).float())
    # loss = torch.nn.MarginRankingLoss(margin=0.01)
    # shuffle_idx = np.random.randint(0, pred_sdf.shape[1], size=pred_sdf.shape[1])
    # sdf_constraint_target += loss(pred_sdf[:, shuffle_idx], pred_sdf, torch.sign((gt_sdf[:, shuffle_idx] > gt_sdf).float()-0.5))

    # -----------------
    losses = {'sdf': sdf_constraint_target.mean() * 1e3,}
    return losses



import torch


def compute_hyper_elastic_loss(
    input_coords, output, batch_size=None, alpha_l=1, alpha_a=1, alpha_v=1
):
    """Compute the hyper-elastic regularization loss."""

    grad_u = compute_jacobian_matrix(input_coords, output, add_identity=False)
    grad_y = compute_jacobian_matrix(input_coords, output, add_identity=True)  # This is slow, faster to infer from grad_u

    # Compute length loss
    length_loss = torch.linalg.norm(grad_u, dim=(-2, -1))
    length_loss = torch.pow(length_loss, 2)
    length_loss = torch.mean(length_loss)
    length_loss = 0.5 * alpha_l * length_loss

    # Compute cofactor matrices for the area loss
    cofactors = torch.zeros(batch_size, input_coords.shape[1], output.shape[-1], output.shape[-1])
    if output.shape[-1] == 3:
        # Compute elements of cofactor matrices one by one (Ugliest solution ever?)
        cofactors[:, :, 0, 0] = torch.det(grad_y[:, :, 1:, 1:])
        cofactors[:, :, 0, 1] = torch.det(grad_y[:, :, 1:, 0::2])
        cofactors[:, :, 0, 2] = torch.det(grad_y[:, :, 1:, :2])
        cofactors[:, :, 1, 0] = torch.det(grad_y[:, :, 0::2, 1:])
        cofactors[:, :, 1, 1] = torch.det(grad_y[:, :, 0::2, 0::2])
        cofactors[:, :, 1, 2] = torch.det(grad_y[:, :, 0::2, :2])
        cofactors[:, :, 2, 0] = torch.det(grad_y[:, :, :2, 1:])
        cofactors[:, :, 2, 1] = torch.det(grad_y[:, :, :2, 0::2])
        cofactors[:, :, 2, 2] = torch.det(grad_y[:, :, :2, :2])
    elif output.shape[-1] == 2:
        cofactors[:, :, 0, 0] = grad_y[:, :, 1, 1]
        cofactors[:, :, 0, 1] = grad_y[:, :, 1, 0]
        cofactors[:, :, 1, 0] = grad_y[:, :, 0, 1]
        cofactors[:, :, 1, 1] = grad_y[:, :, 0, 0]

    # Compute area loss
    area_loss = torch.pow(cofactors, 2)
    area_loss = torch.sum(area_loss, dim=-2)
    area_loss = area_loss - 1
    area_loss = torch.maximum(area_loss, torch.zeros_like(area_loss))
    area_loss = torch.pow(area_loss, 2)
    area_loss = torch.mean(area_loss)  # sum over dimension 1 and then 0
    area_loss = alpha_a * area_loss

    loss = length_loss + area_loss
    if output.shape[-1] == 3:
        # Compute volume loss
        volume_loss = torch.det(grad_y)
        volume_loss = torch.mul(torch.pow(volume_loss - 1, 4), torch.pow(volume_loss, -2))
        volume_loss = torch.mean(volume_loss)
        volume_loss = alpha_v * volume_loss

        # Compute total loss
        loss += volume_loss

    return loss #/ batch_size


def compute_bending_energy(input_coords, output, batch_size=None):
    """Compute the bending energy."""

    jacobian_matrix = compute_jacobian_matrix(input_coords, output, add_identity=False)

    dx_xyz = torch.zeros(input_coords.shape[0], input_coords.shape[1], output.shape[-1],  output.shape[-1])
    dy_xyz = torch.zeros(input_coords.shape[0], input_coords.shape[1],  output.shape[-1],  output.shape[-1])
    if output.shape[-1] == 3:
        dz_xyz = torch.zeros(input_coords.shape[0], input_coords.shape[1],  output.shape[-1],  output.shape[-1])
    for i in range(output.shape[-1]):
        dx_xyz[:, :, i, :] = gradient(input_coords, jacobian_matrix[:, :, i, 0])[..., 0:output.shape[-1]]
        dy_xyz[:, :, i, :] = gradient(input_coords, jacobian_matrix[:, :, i, 1])[..., 0:output.shape[-1]]
        if output.shape[-1] == 3:
            dz_xyz[:, :, i, :] = gradient(input_coords, jacobian_matrix[:, :, i, 2])[..., 0:output.shape[-1]]

    dx_xyz = torch.square(dx_xyz)
    dy_xyz = torch.square(dy_xyz)
    loss = (
        torch.mean(dx_xyz[:, :, :, 0])
        + torch.mean(dy_xyz[:, :, :, 1])
    )
    loss += (2 * torch.mean(dx_xyz[:, :, 1]))

    if output.shape[-1] == 3:
        dz_xyz = torch.square(dz_xyz)
        loss += torch.mean(dz_xyz[:, :, :, 2])
        loss += (2 * torch.mean(dx_xyz[:, :, 2])+ 2* torch.mean(dy_xyz[:, :, 2]))
    return loss * 1/ 8


def compute_jacobian_loss(input_coords, output, batch_size=None):
    """Compute the jacobian regularization loss."""

    # Compute Jacobian matrices
    jac = compute_jacobian_matrix(input_coords, output)

    # Compute determinants and take norm
    loss = torch.det(jac) - 1
    loss = torch.linalg.norm(loss, 1)

    return loss / batch_size


def compute_jacobian_matrix(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix of the output wrt the input."""

    jacobian_matrix = torch.zeros(input_coords.shape[0], input_coords.shape[1], output.shape[-1], output.shape[-1])
    for i in range(output.shape[-1]):
        jacobian_matrix[:, :, i, :] = gradient(input_coords, output[..., i])[..., 0:output.shape[-1]]
        if add_identity:
            jacobian_matrix[..., i, i] += torch.ones_like(jacobian_matrix[..., i, i])
    return jacobian_matrix


def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad


def loss_anglytical_sdfic(model_output, gt, stage='train', dict_losses={}):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''

    whether_eikonal = dict_losses['whether_eikonal'] if 'whether_eikonal' in dict_losses.keys() else False
    whether_vec = dict_losses['whether_vec'] if 'whether_vec' in dict_losses.keys() else False
    whether_jacobian = dict_losses['whether_jacobian'] if 'whether_jacobian' in dict_losses.keys() else False
    whether_hyperelastic = dict_losses[
        'whether_hyperelastic'] if 'whether_hyperelastic' in dict_losses.keys() else False
    whether_bendingenergy = dict_losses[
        'whether_bendingenergy'] if 'whether_bendingenergy' in dict_losses.keys() else False

    # if stage == 'train':
    '''
    get ground-truth of template and target shapes
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_templ_sdf = gt['templ_sdf']
    gt_templ_normals = gt['templ_normals']
    coords_shape = model_output['model_in']
    '''
    get input of the network, which is 3D coordinates with covariates
    '''
    if 'all_input' in model_output.keys():
        coords = model_output['all_input']  # 3D coordinates with covariates
    else:
        coords = model_output['model_in']  # 3D coordinates

    '''
    get predicted data
    '''
    pred_sdf = model_output['model_out']  # predicted sdf value of target shape
    pred_template = model_output['template']  # predicted
    pred_vect_fields = model_output['vec_fields']  # predicted sdf value of the template shape
    pred_evol = model_output['model_map']
    '''
    1. SDF for target shape
    '''
    # l2_loss = torch.nn.MSELoss()
    # sdf_constraint_target = l2_loss(pred_sdf, gt_sdf)
    bce_loss = torch.nn.HingeEmbeddingLoss(
        margin=0.1)  # torch.nn.BCEWithLogitsLoss() #torch.nn.HingeEmbeddingLoss(margin=0.1)#torch.nn.BCEWithLogitsLoss()#
    sdf_constraint_target = bce_loss(pred_sdf, (gt_sdf > 0).float() * (
        -2) + 1)  # torch.relu(-1 * pred_sdf * torch.sign(gt_sdf)).mean()  #bce_loss(pred_sdf, (gt_sdf > 0).float() * (-2)+1)
    # bce_loss = torch.nn.BCELoss()
    # sdf_constraint_target = bce_loss(pred_sdf / 2 + 0.5, (gt_sdf > 0).float())
    # loss = torch.nn.MarginRankingLoss(margin=0.01)
    # shuffle_idx = np.random.randint(0, pred_sdf.shape[1], size=pred_sdf.shape[1])
    # sdf_constraint_target += loss(pred_sdf[:, shuffle_idx], pred_sdf, torch.sign((gt_sdf[:, shuffle_idx] > gt_sdf).float()-0.5))

    '''
    2. SDF for template shape
    '''
    # sdf_constraint_template = l2_loss(pred_template, gt_templ_sdf)
    # sdf_constraint_template = bce_loss(pred_template/2+0.5, (gt_templ_sdf > 0).float())
    sdf_constraint_template = bce_loss(pred_template, (gt_templ_sdf > 0).float() * (
        -2) + 1)  # l2_loss(pred_template, gt_templ_sdf) #bce_loss(pred_template, (gt_templ_sdf > 0).float() * (-2) + 1)
    # sdf_constraint_template = torch.relu(-1 * pred_template * torch.sign(gt_templ_sdf)).mean()
    # loss = torch.nn.MarginRankingLoss(margin=0.01)
    # shuffle_idx = np.random.randint(0, pred_sdf.shape[1], size=pred_sdf.shape[1])
    # sdf_constraint_template += loss(pred_template[:, shuffle_idx], pred_template, torch.sign((gt_templ_sdf[:, shuffle_idx] > gt_templ_sdf).float()-0.5))
    # sdf_constraint_template = bce_loss(pred_template, (gt_templ_sdf>0).float())
    # sdf_constraint_template = torch.relu(-1 * pred_template * torch.sign(gt_templ_sdf)) #bce_loss(pred_template, (gt_templ_sdf > 0).float() * (-2)+1)

    '''
    3. evolution
    '''

    sdf_evo = 0
    # evo_loss = torch.nn.BCELoss()
    for i_key in pred_evol.keys():
        if '_grad' not in i_key and 'initial' not in i_key:
            # print(i_key)
            # sdf_evo += l2_loss(pred_evol[i_key], gt_sdf)
            sdf_evo += bce_loss(pred_evol[i_key], (gt_sdf > 0).float() * (-2) + 1)
            # sdf_evo += bce_loss(pred_evol[i_key], (gt_templ_sdf > 0).float() * (-2) + 1)
            # sdf_evo += bce_loss(pred_evol[i_key]/2+0.5, (gt_sdf > 0).float())


    '''
    4. inverse consistency
    '''

    pred_inverse = model_output['inverse_to_target']
    ic = torch.nn.MSELoss()
    ic_loss = ic(pred_inverse, coords_shape)

    '''
    3. Vector Field constraint: boundary condition
    '''
    templ_constraint = 0
    for i_key in pred_vect_fields.keys():
        if 'templ' in i_key:
            current_vec_field = pred_vect_fields[i_key]
            current_templ_constraint = current_vec_field.norm(dim=-1)
            templ_constraint += current_templ_constraint.mean()

    # -----------------
    losses = {'sdf': sdf_constraint_target.mean() * 1e3,
              'sdf_templ': sdf_constraint_template.mean() * 1e3,
              'templ_constraint': templ_constraint * 1e1,
              'ic_loss': ic_loss * 1e3,
              }
    if sdf_evo != 0:
        losses.update({'sdf_evo': sdf_evo * 1e2})
    if 'loss_lip' in model_output.keys():
        losses.update({'loss_lip': model_output['loss_lip'] * 1e-3})
        # losses.update({'loss_lip_initial': model_output['loss_lip_initial'] * 1e-3})
        # 'templ_constraint': templ_constraint}
        # 'sdf_templ': sdf_constraint_template.mean()*1e3,}

    if whether_eikonal:
        '''
        4. Vector Field constraint: Eikonal Constraint
        '''

        # PDE constraints
        def eikonal_constraint(gradient):
            return (gradient.norm(dim=-1) - 1.) ** 2

        def get_unsigned_distance_from_vec_field(vec_field):
            df_of_vec_field = vec_field.norm(dim=-1)
            return df_of_vec_field

        def get_velocity(df_of_vec_field, coords, dim):
            grad = diff_operators.gradient(df_of_vec_field, coords)[:, :, 0:dim]
            vec = grad(df_of_vec_field, coords)[:, :, 0:dim]
            return vec

        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_norm_constraint = {}

        for i_key in pred_vect_fields.keys():
            #if '_templ' not in i_key:
            if 'overall' in i_key:
                current_vec_field = pred_vect_fields[i_key]
                df_of_vec_field = get_unsigned_distance_from_vec_field(current_vec_field)
                current_velocity = get_velocity(df_of_vec_field, coords, dim=model_output['model_in'].shape[-1])
                grad_constraint = eikonal_constraint(current_velocity).unsqueeze(-1)
                velocity_norm_constraint[i_key] = grad_constraint.mean()
        '''
        gradient = diff_operators.gradient(pred_template, coords_shape)[:, :, 0:coords_shape.shape[-1]]
        eikonal_loss = {}
        eikonal_loss['eikonal'] = eikonal_constraint(gradient).mean()  # * 0.01
        losses.update(eikonal_loss)
        # 'v_mean': velocity_mean_constraint,
        # 'v_norm': velocity_norm_constraint}
        # 'normal_constraint':normal_constraint.mean()}  # 1e1      # 5e1





    if whether_vec:
        '''
        5. Vector Field constraint: Velocity Constraint
        '''

        def velocity_field_constraint(predicted_sdf, coords):
            velocity = diff_operators.gradient(predicted_sdf,
                                               coords) ** 2  # .index_select(-1, torch.arange(coords_shape.shape[-1]).to(coords_shape.device))** 2
            velocity_field_constraint = torch.sum(velocity, dim=-1).mean()
            # vec = diff_operators.laplace(predicted_sdf, coords)[:, :, 0:dim]
            return velocity_field_constraint

        pred_vect_fields = model_output['vec_fields']
        velocity_vec_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key and 'implicit' not in i_key:
                current_constraint = 0
                for i in range(pred_vect_fields[i_key].shape[-1]):
                    current_constraint += \
                        velocity_field_constraint(pred_vect_fields[i_key][..., i], coords)
                velocity_vec_constraint['gradient_' + i_key] = current_constraint * 10
        losses.update(velocity_vec_constraint)

    if whether_jacobian:
        '''
        6. Vector Field constraint: Jacobian Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_jacobian_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_jacobian_loss(coords, pred_vect_fields[i_key], batch_size=coords.shape[0])
                velocity_jacobian_constraint['jacobian_' + i_key] = current_constraint * 10
        losses.update(velocity_jacobian_constraint)

    if whether_hyperelastic:
        '''
        7. Vector Field constraint: HyperElastic Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_HE_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_hyper_elastic_loss(coords, pred_vect_fields[i_key],
                                                                batch_size=coords.shape[0], alpha_l=1, alpha_a=1,
                                                                alpha_v=1)
                velocity_HE_constraint['hyper_elastic_' + i_key] = current_constraint
        losses.update(velocity_HE_constraint)

    if whether_bendingenergy:
        '''
        8. Vector Field constraint: Bending Energy Regularity
        '''
        pred_vect_fields = model_output['vec_fields']
        velocity_BE_constraint = {}

        for i_key in pred_vect_fields.keys():
            if '_templ' not in i_key:
                current_constraint = compute_bending_energy(coords, pred_vect_fields[i_key], batch_size=coords.shape[0])
                velocity_BE_constraint['bending_energy_' + i_key] = current_constraint
        losses.update(velocity_BE_constraint)

    return losses


