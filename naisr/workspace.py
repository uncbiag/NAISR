#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import torch

model_params_subdir = 'checkpoints'
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "latentcodes"
logs_filename = "Logs.pth"
recon_testset_subdir = "Results_recon_testset"
inter_testset_subdir = "Results_inter_testset"
generation_subdir = "Results_generation"
recon_testset_ttt_subdir = "Results_recon_testset_ttt"
inter_testset_ttt_subdir = "Results_inter_testset_ttt"
generation_ttt_subdir = "Results_generation_ttt"
reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
reconstruction_models_subdir = "Models"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
training_meshes_subdir = "TrainingMeshes"

reconstructions_subdir = "Reconstructions_0312"
transport_subdir = "Tranports"
transport_meshes_subdir = "Meshes"
transport_codes_subdir = "Codes"
transport_models_subdir = "Models"

transport_general_subdir = "GeneralTranports"
transport_general_meshes_subdir = "Meshes"
transport_general_codes_subdir = "Codes"
transport_general_models_subdir = "Models"

reconstructions_inv_subdir = "InverseReconstruction"
reconstruction_inv_meshes_subdir = "Meshes"
reconstruction_inv_codes_subdir = "Codes"
reconstruction_inv_models_subdir = "Models"

reconstructions_withcov_subdir = "ReconstructionsWithCov_0312"
reconstruction_withcov_meshes_subdir = "Meshes"
reconstruction_withcov_codes_subdir = "Codes"
reconstruction_withcov_models_subdir = "Models"

vis_comp_subdir='VisComp'
vis_comp_meshes_subdir = "Meshes"


tensorboard_log_subdir = "TensorboardLogs"
from torch.utils.tensorboard import SummaryWriter


def save_tensorboard_logs(saver, step, **kargs):
    if step % 10 == 0:
        for ln in kargs.keys():
            saver.add_scalar(ln, kargs[ln], step)



def get_tensorboard_logs_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, tensorboard_log_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def create_tensorboard_saver(experiment_dir):
    return SummaryWriter(get_tensorboard_logs_dir(experiment_dir, True))

def load_experiment_specifications(experiment_directory):

    filename = experiment_directory #os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            .format(experiment_directory)
        )

    return json.load(open(filename))


def load_model_parameters(experiment_directory, checkpoint, decoder):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    decoder.load_state_dict(data["model_state_dict"])

    return decoder, data["epoch"]

# def build_decoder(experiment_directory, experiment_specs):

#     arch = __import__(
#         "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
#     )

#     latent_size = experiment_specs["CodeLength"]

#     decoder = arch.Decoder(latent_size, **experiment_specs["NetworkSpecs"]).cuda()

#     return decoder


# def load_decoder(
#     experiment_directory, experiment_specs, checkpoint, data_parallel=True
# ):

#     decoder = build_decoder(experiment_directory, experiment_specs)

#     if data_parallel:
#         decoder = torch.nn.DataParallel(decoder)

#     epoch = load_model_parameters(experiment_directory, checkpoint, decoder)

#     return (decoder, epoch)


#def load_latent_vectors(experiment_directory, checkpoint):
#
#    filename = os.path.join(
#        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
#    )
#
#    if not os.path.isfile(filename):
#        raise Exception(
#            "The experiment directory ({}) does not include a latent code file"
#            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
#        )
#
#    data = torch.load(filename)
#
#    if isinstance(data["latent_codes"], torch.Tensor):
#
#        num_vecs = data["latent_codes"].size()[0]
#
#        lat_vecs = []
#        for i in range(num_vecs):
#            lat_vecs.append(data["latent_codes"][i].cuda())
#
#        return lat_vecs
#
#    else:
#
#        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape
#
#        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)
#
#        lat_vecs.load_state_dict(data["latent_codes"])
#
#        return lat_vecs.weight.data.detach()


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_recon_testset_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        recon_testset_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )

def get_recon_testset_ttt_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        recon_testset_ttt_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )



def get_recon_testset_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        recon_testset_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        instance_name + ".pth",
    )


def get_inter_testset_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        inter_testset_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )

def get_inter_testset_ttt_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        inter_testset_ttt_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )


def get_inter_testset_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        inter_testset_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        instance_name + ".pth",
    )


def get_generation_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        generation_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )

def get_generation_ttt_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        generation_ttt_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )


def get_generation_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        generation_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        instance_name + ".npz",
    )






def build_decoder(experiment_directory, experiment_specs):

    arch = __import__(
        "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
    )

    latent_size = experiment_specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **experiment_specs["NetworkSpecs"]).cuda()

    return decoder


def load_decoder(
    experiment_directory, experiment_specs, checkpoint, data_parallel=True
):

    decoder = build_decoder(experiment_directory, experiment_specs)

    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)

    epoch = load_model_parameters(experiment_directory, checkpoint, decoder)

    return (decoder, epoch)


def load_pre_trained_latent_vectors(experiment_directory, checkpoint):

    filename = os.path.join(
        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
        )

    data = torch.load(filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        num_vecs = data["latent_codes"].size()[0]

        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i].cuda())

        return lat_vecs

    else:

        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)

        lat_vecs.load_state_dict(data["latent_codes"])

        return lat_vecs.weight.data.detach()



def load_latent_vectors(experiment_directory,  name, device):

    filename = os.path.join(
        experiment_directory, latent_codes_subdir, name + ".pth"
    )
    print(filename)
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, name)
        )

    data = torch.load(filename, map_location=torch.device(device))

    if isinstance(data["latent_codes"], torch.Tensor):

        num_vecs = data["latent_codes"].size()[0]

        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i].cuda())

        return lat_vecs

    else:

        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)

        lat_vecs.load_state_dict(data["latent_codes"])

        return lat_vecs.weight.data.detach()

def load_transport_vectors(experiment_directory,  name, device):

    filename = os.path.join(
        experiment_directory, name + ".pth"
    )
    print(filename)
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, name)
        )

    data = torch.load(filename, map_location=torch.device(device))

    return data


def load_inferred_covariates(experiment_directory,  name, device):

    filename = os.path.join(
        experiment_directory, name + "_covariate.pth"
    )
    print(filename)
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, name)
        )

    data = torch.load(filename, map_location=torch.device(device))
    for name, value in data.items():
        data[name] = value.to(device)
    return data

def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name,
        instance_name + ".ply",
    )


def get_reconstructed_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )






def create_code_snapshot(root, dst_path, extensions=(".py", ".json"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path

    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)
