#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
from torch.nn import functional as F

import signal
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import time
import argparse

import naisr
from naisr.utils import *
import naisr.workspace as ws
import naisr.loss_funcs as loss_functions
import utils
import math

def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))



def save_model(model_dir, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(model_dir, True)
    torch.save({"epoch": epoch, "model_state_dict": decoder.state_dict()},os.path.join(model_params_dir, filename),)
    #torch.save(decoder.state_dict(), os.path.join(model_params_dir, 'model_epoch_%04d.pth' % epoch))


def save_optimizer(model_dir, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(model_dir, True)
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(model_dir, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(model_dir), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)
    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(model_dir, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(model_dir, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(model_dir, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(model_dir), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    model_dir,
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
        os.path.join(model_dir, ws.logs_filename),
    )


def load_logs(model_dir):

    full_filename = os.path.join(model_dir, ws.logs_filename)

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


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def load_checkpoints(continue_from, ws, model_dir, decoder, optimizer_all):

    logging.info('continuing from "{}"'.format(continue_from))

    decoder, model_epoch = ws.load_model_parameters(
        model_dir, continue_from, decoder
    )
    optimizer_all, optimizer_epoch = load_optimizer(
        ws, model_dir, continue_from + ".pth", optimizer_all
    )

    loss_log, lr_log, timing_log, log_epoch = load_logs(
        ws, model_dir
    )

    if not log_epoch == model_epoch:
        loss_log, lr_log, timing_log = clip_logs(
            loss_log, lr_log, timing_log, model_epoch
        )

    if not (model_epoch == optimizer_epoch and model_epoch == model_epoch):
        raise RuntimeError(
            "epoch mismatch: {} vs {} vs {} vs {}".format(
                model_epoch, optimizer_epoch, model_epoch, log_epoch
            )
        )

    start_epoch = model_epoch + 1

    return  decoder, optimizer_all, start_epoch, loss_log, lr_log, timing_log, start_epoch


def main_function(experiment_directory, clip_grad=True, loss_schedules=None):



    # load specs
    specs = ws.load_experiment_specifications(experiment_directory)

    # load model
    model = naisr.DeepNAIGSR(
                 template_attributes=specs['TemplateAttributes'],
                 in_features=specs['InFeatures'],
                 hidden_features=specs['HiddenFeatures'],
                 hidden_layers=specs['HidenLayers'],
                 out_features=specs['OutFeatures'],
                 device=specs['Device'],
                 backbone=specs['Backbone'],
                 outermost_linear=False,
                 pos_enc=specs['PosEnc'],
    latent_size=specs["CodeLength"])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The total number of parameters: ' + str(pytorch_total_params))

    # Define the loss
    loss_fn = loss_functions.loss_pointcloud_sdf
    dsp_summary_fn = utils.write_deepsdf_dsp_summary
    #
    #dsp_summary_fn = utils.write_dsp_summary
    #
    device = specs['Device']
    model.to(device)

    # save and log
    model_dir = os.path.join(specs['LoggingRoot'], specs['ExperimentName'])

    def save_latest(epoch):
        save_model(model_dir, "latest.pth", model, epoch)
        save_optimizer(model_dir, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(model_dir, "latest.pth", lat_vecs, epoch)


    def save_final(epoch):
        save_model(model_dir, "final.pth", model, epoch)
        save_optimizer(model_dir, "final.pth", optimizer_all, epoch)
        save_latent_vectors(model_dir, "final.pth", lat_vecs, epoch)

    def save_current(epoch):
        save_model(model_dir, 'epoch_%04d.pth' % epoch, model, epoch)
        save_optimizer(model_dir,  'epoch_%04d.pth' % epoch, optimizer_all, epoch)
        save_latent_vectors(model_dir,  'epoch_%04d.pth' % epoch, lat_vecs, epoch)


    #if os.path.exists(model_dir):
    #    val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
    #    if val == 'y':
    #        shutil.rmtree(model_dir)

    #os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)
    summary_fn = utils.write_deepsdf_summary


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
    lr_schedules = get_learning_rate_schedules(specs)
    double_precision = specs['DoublePrecision']
    shapetype = specs["Class"]
    template_attributes = specs["TemplateAttributes"]
    num_samp_per_scene = specs["SamplesPerScene"]
    # init dataloader
    if shapetype == 'Airway':
        train_sdf_dataset = naisr.PediatricAirway3DShapeDataset(
            filename_datasource=data_source,
            filename_split=split_file,
            attributes=specs['Attributes'],
            split='train')

        test_sdf_dataset = naisr.PediatricAirway3DShapeDataset(
            filename_datasource=data_source,
            filename_split=split_file,
            attributes=specs['Attributes'],
            split='test')



    batch_size = specs["BatchSize"]
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

    if specs["Articulation"] == True:
        num_scenes = len(train_sdf_dataset)
    logging.info("There are {} cases".format(num_scenes))

    # embedding
    latent_size = specs["CodeLength"]
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=1).to(device)
    #lat_vecs.requires_grad_ = True

    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0)/ math.sqrt(latent_size),

    )
    logging.info(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    #  optimizer

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )


    '''
    get named parameters
    '''
    named_params = []
    #  optimizer
    named_params += list(model.named_parameters())
    list_composers = []
    for named_param in named_params:
        if "composer" in named_param[0]:
            list_composers.append(named_param)


    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    total_steps = 0
    with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
        train_losses = []

        for epoch in range(num_epochs):

            if not epoch % epochs_til_checkpoint and epoch:
                #torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch), np.array(train_losses))
                save_current(epoch)
                save_latest(epoch)


            for step, (model_input, attributes, gt, indices) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = model_input.to(device) #{key: value.cuda() for key, value in model_input.items()}
                #template_input = template_input.to(device)

                attributes = {key: value.to(device) for key, value in attributes.items()}
                for key, value in gt.items():
                    if not isinstance(value, list):
                        gt[key] = value.to(device)

                gt['sdf'] = torch.clamp(gt['sdf'], -1., 1.)
                #gt['template_sdf'] = torch.clamp(gt['template_sdf'],-1., 1.)

                indices =  indices[:, None].repeat(1, num_samp_per_scene).to(device) #indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1).to(device)
                batch_vecs = lat_vecs(indices)

                if double_precision:
                    model_input = model_input.to(device).double()
                    #template_input = template_input.to(device).double()
                    attributes = {key: value.to(device).double() for key, value in attributes.items()}
                    gt = {key: value.to(device).double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optimizer_all.zero_grad()
                        model_output = model(model_input, attributes, batch_vecs)
                        losses = loss_fn(model_output, gt, batch_vecs, epoch, dict_losses=specs['Loss']) #, whether_vec=True)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean()
                        train_loss.backward()
                        return train_loss

                    optimizer_all.step(closure)

                model_output = model(model_input, attributes, batch_vecs, training=True)
                model_output['model_out'] = torch.clamp(model_output['model_out'],-1., 1.)
                #print(model_output)
                losses = loss_fn(model_output, gt, batch_vecs, epoch, dict_losses=specs['Loss']) #, whether_vec=True)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    #train_loss += single_loss
                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                writer.add_scalar("total_train_loss", train_loss, total_steps)
                #writer.add_scalar("composer", model.composer, total_steps)
                #for ith_composer in range(len(list_composers)):
                #    writer.add_scalar(str(ith_composer) + '_' + list_composers[ith_composer][0],
                #                      list_composers[ith_composer][1],  total_steps)

                if not use_lbfgs:
                    optimizer_all.zero_grad()
                    train_loss.backward()
                    optimizer_all.step()

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)


                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)


                pbar.update(1)

                if not total_steps % steps_til_summary:
                    model.eval()
                    summary_fn(model, model_input, batch_vecs, attributes, gt, model_output, writer, total_steps, prefix='train_',device=device)
                    dsp_summary_fn(model, model_input, batch_vecs, attributes, gt, model_output, writer, total_steps,
                                   prefix='train_', device=device)
                    model.train()

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (
                    epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        lat_vecs.eval()
                        #with torch.no_grad():
                        val_losses = []
                        for (model_input, attributes, gt, indices) in val_dataloader:
                            model_input = model_input.to(device)  # {key: value.cuda() for key, value in model_input.items()}
                            #template_input = template_input.to(device)

                            #indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)
                            #batch_vecs = lat_vecs(indices)

                            indices = indices[:, None].repeat(1, num_samp_per_scene).to(
                                device)  # indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1).to(device)
                            batch_vecs = lat_vecs(indices)

                            attributes = {key: value.to(device) for key, value in attributes.items()}
                            for key, value in gt.items():
                                if not isinstance(value, list):
                                    gt[key] = value.to(device)

                            gt['sdf'] = torch.clamp(gt['sdf'],-1., 1.)

                            if double_precision:
                                model_input = model_input.to(device).double()
                                #template_input = template_input.to(device).double()
                                attributes = {key: value.to(device).double() for key, value in attributes.items()}
                                gt = {key: value.to(device).double() for key, value in gt.items()}


                            model_output = model(model_input,attributes, batch_vecs)
                            model_output['model_out']  =torch.clamp(model_output['model_out'], -1., 1.)


                            val_loss = loss_fn(model_output, gt, batch_vecs, epoch, dict_losses=specs['Loss']) #, whether_vec=True)
                            val_losses.append(val_loss['sdf'].detach().cpu().numpy())

                        writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        summary_fn(model, model_input, batch_vecs, attributes, gt, model_output, writer, total_steps,  prefix='test_',device=device)
                        dsp_summary_fn(model, model_input, batch_vecs, attributes, gt, model_output, writer,
                                       total_steps,
                                       prefix='test_', device=device)

                        model.train()
                        lat_vecs.train()

                total_steps += 1

        save_final(epoch)
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train a DeepConvSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='examples/pediatric_airway/naivf_deepnaigsr_l2.json',
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )


    args = arg_parser.parse_args()
    main_function(args.experiment_directory, clip_grad=False)
