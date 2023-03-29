import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np

import deep_sdf
import naisr.workspace as ws
import naisr
import torch.utils.data as data_utils

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    device=1,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).to(device)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).to(device)

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        #sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
        #    test_sdf, num_samples
        #).cuda()
        xyz = test_sdf[:, 0:3]
        sdf_gt = test_sdf[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).to(device)

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.item())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.item()

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='examples/pediatric_airway/naivf_2d_template.json',
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )

    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )

    arg_parser.add_argument(
        "--seed",
        dest="seed",
        default=10,
        help="random seed",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=256,
        help="Marching cube resolution.",
    )

    skip = False

    use_octree_group = arg_parser.add_mutually_exclusive_group()
    use_octree_group.add_argument(
        '--octree',
        dest='use_octree',
        action='store_true',
        help='Use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )
    use_octree_group.add_argument(
        '--no_octree',
        dest='use_octree',
        action='store_false',
        help='Don\'t use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )


    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).to(device)
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var


    experiment_directory = args.experiment_directory
    specs_filename = os.path.join(args.experiment_directory)

    logging.info("running " + experiment_directory)

    # backup code
    #now = datetime.datetime.now()
    #code_bk_path = os.path.join(
    #    experiment_directory, 'code_bk_%s.tar.gz' % now.strftime('%Y_%m_%d_%H_%M_%S'))
    # ws.create_code_snapshot('./', code_bk_path,
    #                         extensions=('.py', '.json', '.cpp', '.cu', '.h', '.sh'),
    #                         exclude=('examples', 'third-party', 'bin'))

    specs = ws.load_experiment_specifications(experiment_directory)

    #
    device = specs['Device']

    # save and log
    model_dir = os.path.join(specs['LoggingRoot'], specs['ExperimentName'])

    # if os.path.exists(model_dir):
    #    val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
    #    if val == 'y':
    #        shutil.rmtree(model_dir)

    # os.makedirs(model_dir)

    # dsp_summary_fn = utils.write_dsp_summary

    '''
    load dataset
    '''
    data_source = specs["DataSource"]
    split_file = specs["Split"]

    '''
    load training parameters
    '''
    num_epochs = specs["NumEpochs"]
    epochs_til_checkpoint = specs['EpochsTilCkpt']
    steps_til_summary = specs['StepsTilSummary']
    use_lbfgs = specs['UseLBFGS']
    #lr_schedules = get_learning_rate_schedules(specs)
    double_precision = specs['DoublePrecision']
    shapetype = specs["Class"]
    template_attributes = specs["TemplateAttributes"]

    # logging.info("Experiment description: \n" + specs["Description"])

    # data_source = specs["DataSource"]
    train_split_file = specs["Split"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.info(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]


    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    #decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            specs["ExperimentName"], ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.to(device)

    #with open(specs["Split"], "r") as f:
    #    split = json.load(f)

    #npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)

    # random.shuffle(npz_filenames)
    #npz_filenames = sorted(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        specs["ExperimentName"], ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    clamping_function = None
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        # clamping_function = lambda x: x * specs["ClampingDistance"]
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    # init dataloader
    #if shapetype == 'Airway':
    train_sdf_dataset = naisr.PediatricAirway3DShapeDataset_1(
        filename_datasource=data_source,
        filename_split=split_file,
        attributes=specs['Attributes'],
        split='all')
    template_dataset = naisr.PediatricAirway3DShapeDataset_1(
        filename_datasource=data_source,
        filename_split=split_file,
        attributes=specs['Attributes'],
        split='all')

    test_sdf_dataset = naisr.PediatricAirway3DShapeDataset_1(
        filename_datasource=data_source,
        filename_split=split_file,
        attributes=specs['Attributes'],
        split='test')



    batch_size = 1
    num_data_loader_threads = specs["DataLoaderThreads"]


    train_dataloader = data_utils.DataLoader(
        train_sdf_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    val_dataloader = data_utils.DataLoader(
        test_sdf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    #for ii, npz in enumerate(npz_filenames):
    for step, (model_input, template, attributes, gt, indices) in enumerate(train_dataloader):

        model_input = model_input.to(device)
        attributes = {key: value.to(device) for key, value in attributes.items()}
        for key, value in gt.items():
            if not isinstance(value, list):
                gt[key] = value.to(device)
        indices = indices.to(device)

        #full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
        subj_id = str(gt["id"][0].cpu().numpy())
        logging.debug("loading {}".format(gt['id']))

        data_sdf = model_input #deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, subj_id + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, subj_id + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, subj_id)
                latent_filename = os.path.join(
                    reconstruction_codes_dir, subj_id+ ".pth"
                )

            if (
                skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(subj_id))

            data_sdf = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            #data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            if not os.path.isfile(latent_filename):
                err, latent = reconstruct(
                    decoder,
                    int(args.iterations),
                    latent_size,
                    data_sdf,
                    0.01,  # [emp_mean,emp_var],
                    0.1,
                    device=device,
                    num_samples=data_sdf.shape[0],
                    lr=5e-3,
                    l2reg=True,
                )
                logging.info("reconstruct time: {}".format(time.time() - start))
                logging.info("reconstruction error: {}".format(err))
                err_sum += err
                # logging.info("current_error avg: {}".format((err_sum / (ii + 1))))
                # logging.debug(ii)

                # logging.debug("latent: {}".format(latent.detach().cpu().numpy()))
            else:
                logging.info("loading from " + latent_filename)
                latent = torch.load(latent_filename).squeeze(0)

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    if args.use_octree:
                        deep_sdf.mesh.create_mesh_octree(
                            decoder, latent, mesh_filename, N=args.resolution, max_batch=int(2 ** 17),
                            clamp_func=clamping_function
                        )
                    else:
                        deep_sdf.mesh.create_mesh(
                            decoder, latent, mesh_filename, N=args.resolution, max_batch=int(2 ** 17),
                        )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)