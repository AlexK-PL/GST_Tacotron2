# ONE EPOCH = one forward pass and one backward pass of all the training examples.
#
# BATCH SIZE = the number of training examples in one forward/backward pass. The
# higher the batch size, the more memory space you'll need.
#
# NUMBER OF ITERATIONS = number of passes, each pass using [batch size] number of
# examples. To be clear, one pass = one forward pass + one backward pass.
#
# Example: if you have 1000 training examples, and your batch size is 500, then
# it will take 2 iterations to complete 1 epoch.

import os
import time
import math

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from numpy import finfo

from Tacotron2 import tacotron_2
from fp16_optimizer import FP16_Optimizer
from distributed import apply_gradient_allreduce
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger


def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def reduce_tensor(tensor, n_gpus):
    # this function is recorded in the computation graph. Gradients propagating to the cloned tensor will propagate to
    # the original tensor
    rt = tensor.clone()
    # Each rank has a tensor and all_reduce sums up all tensors from different ranks to all ranks. Computes the average
    # of the tensor results of all ranks (a rank is a gpu as far as I understood):
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
        #  logger = None
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def init_distributed(hyper_params, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA"
    print("Initializing distributed")
    # Set CUDA device so everything is done on the right GPU
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    torch.distributed.init_process_group(backend=hyper_params['dist_backend'], rank=rank, world_size=n_gpus,
                                         init_method=hyper_params['dist_url'], group_name=group_name)

    print("Initializing distributed: Done")


def load_model(hyper_params):
    # according to the documentation, it is recommended to move a model to GPU before constructing the optimizer
    model = tacotron_2(hyper_params).cuda()
    if hyper_params['fp16_run']:  # converts everything into half type (16 bits)
        model = batchnorm_to_float(model.half())
        model.decoder.attention_layer.score_mask_value = float(finfo('float16').min)

    if hyper_params['distributed_run']:
        model = apply_gradient_allreduce(model)

    return model


def validate(model, criterion, valset, iteration, batch_size, n_gpus, collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""

    # We change to eval() because this is an evaluation stage and not a training
    model.eval()
    # temporarily set all the requires_grad flag to false
    with torch.no_grad():
        # Sampler that restricts data loading to a subset of the dataset. Distributed sampler for distributed batch.
        # Which samples take (randomization?)
        val_sampler = DistributedSampler(valset) if distributed_run else None
        # data loader wraper to the validation data (same as for the training data)
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1, shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            _, _, _, _, gst_scores = y_pred
            if i == 0:
                validation_gst_scores = gst_scores
            else:
                validation_gst_scores = torch.cat((validation_gst_scores, gst_scores), 0)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()  # gets the pure float value with item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)  # Averaged val_loss from all batches

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f} ".format(iteration, val_loss))  # I changed this
        # print("GST scores of the validation set: {}".format(validation_gst_scores.shape))
        logger.log_validation(reduced_val_loss, model, y, y_pred, validation_gst_scores, iteration)


# ------------------------------------------- MAIN TRAINING METHOD -------------------------------------------------- #

def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, rank, group_name,
          hyper_params, train_loader, valset, collate_fn):
    """Training and validation method with logging results to tensorboard and stdout

    :param output_directory (string): directory to save checkpoints
    :param log_directory (string): directory to save tensorboard logs
    :param checkpoint_path (string): checkpoint path
    :param n_gpus (int): number of gpus
    :param rank (int): rank of current gpu
    :param hyper_params (object dictionary): dictionary with all hyper parameters
    """

    # Check whether is a distributed running
    if hyper_params['distributed_run']:
        init_distributed(hyper_params, n_gpus, rank, group_name)

    # set the same fixed seed to reproduce same results everytime we train
    torch.manual_seed(hyper_params['seed'])
    torch.cuda.manual_seed(hyper_params['seed'])

    model = load_model(hyper_params)
    learning_rate = hyper_params['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hyper_params['weight_decay'])

    if hyper_params['fp16_run']:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=hyper_params['dynamic_loss_scaling'])

    # Define the criterion of the loss function. The objective.
    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(output_directory, log_directory, rank)
    # logger = ''

    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            # Re-start the model from the last checkpoint if we save the parameters and don't want to start from 0
            model = warm_start_model(checkpoint_path, model)
        else:
            # CHECK THIS OUT!!!
            model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
            if hyper_params['use_saved_learning_rate']:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    # Set this to make all modules and regularization aware this is the training stage:
    model.train()

    # MAIN LOOP
    for epoch in range(epoch_offset, hyper_params['epochs']):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            # CHECK THIS OUT!!!
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            input_data, output_target = model.parse_batch(batch)
            output_predicted = model(input_data)

            loss = criterion(output_predicted, output_target)

            if hyper_params['distributed_run']:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hyper_params['fp16_run']:
                optimizer.backward(loss)  # transformed optimizer into fp16 type
                grad_norm = optimizer.clip_fp32_grads(hyper_params['grad_clip_thresh'])
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hyper_params['grad_clip_thresh'])

            # Performs a single optimization step (parameter update)
            optimizer.step()
            # This boolean controls overflow when running in fp16 optimizer
            overflow = optimizer.overflow if hyper_params['fp16_run'] else False

            # If overflow is True, it will not enter. If isnan is True, it will not enter neither.
            if not overflow and not math.isnan(reduced_loss) and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grand Norm {:.6f} {:.2f}s/it".format(iteration, reduced_loss,
                                                                                 grad_norm, duration))
                # logs training information of the current iteration
                logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)

            # Every iters_per_checkpoint steps there is a validation of the model and its updated parameters
            if not overflow and (iteration % hyper_params['iters_per_checkpoint'] == 0):
                validate(model, criterion, valset, iteration, hyper_params['batch_size'], n_gpus, collate_fn,
                         logger, hyper_params['distributed_run'], rank)
                if rank == 0:
                    checkpoint_path = os.path.join(output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

            iteration += 1
