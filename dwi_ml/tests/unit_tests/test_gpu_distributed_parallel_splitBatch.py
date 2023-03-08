#!/usr/bin/env python

"""
Based on tutorial from: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
"""
import logging
import os

import numpy as np
import requests.exceptions
import torch
import torch.multiprocessing as mp

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.tests.utils.data_and_models_for_tests import (
    create_test_batch_sampler, create_batch_loader, fetch_testing_data,
    ModelForTest)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# backend: The backend to use. Depending on build-time configurations,
# valid values include mpi, gloo, nccl, and ucc.
backend = "gloo"
batch_size = 50
batch_size_units = 'nb_streamlines'


def manual_test_multi_gpu():
    try:
        data_dir = fetch_testing_data()
        hdf5_filename = os.path.join(data_dir, 'hdf5_file.hdf5')
    except requests.exceptions.ConnectionError:
        # This test is meant to be used on multi-gpu. In our case, cluster does
        # not allow fetch test data. Copy ~/.scilpy/hdf5_file.hdf5 and use.
        hdf5_filename = './hdf5_file.hdf5'

    # Initializing dataset
    dataset = MultiSubjectDataset(hdf5_filename, lazy=False)
    dataset.load_data()

    # Initializing main classes
    batch_sampler, batch_loader = _create_sampler_and_loader(dataset)

    nb_gpus = torch.cuda.device_count()
    dicts, batches, batch_loader = load_and_split_batches(
        nb_gpus, batch_sampler, batch_loader)

    model = ModelForTest()

    mp.spawn(compute_inputs_and_train,
             args=(nb_gpus, dicts, batches, batch_loader, model),
             nprocs=nb_gpus,
             join=True)


def compute_inputs_and_train(gpu_id, nb_gpus, dicts, batches, batch_loader,
                             model):
    batch_ids = dicts[gpu_id]
    batch_streamlines = batches[gpu_id]

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=gpu_id, world_size=nb_gpus)

    model.move_to(gpu_id)
    print(model.device)

    model = DDP(model, device_ids=[gpu_id])
    print(model.device)

    print("GPU ID: {}/ {}. Batch size: {}"
          .format(gpu_id, nb_gpus, len(batch_streamlines)))

    batch_streamlines = [s.to(gpu_id, non_blocking=True,
                              dtype=torch.float)
                         for s in batch_streamlines]

    batch_inputs = batch_loader.load_batch_inputs(
        model.module, batch_streamlines, batch_ids)
    model_outputs = model(batch_inputs)
    print("SUCESS. Model outputs:", len(model_outputs))


def load_and_split_batches(nb_gpus, batch_sampler, batch_loader):
    # Loading a batch of streamlines (on CPU, like in DataLoader).
    batch_sampler.set_context('training')
    batch_loader.set_context('training')
    batch_id_generator = batch_sampler.__iter__()
    batch_idx_tuples = next(batch_id_generator)

    batch_streamlines, batch_idx_dict = \
        batch_loader.load_batch_streamlines(batch_idx_tuples)

    # Currently, our batch sampler samples ~the same number for each subject.

    if nb_gpus > 1:
        nb_subjs = len(batch_idx_tuples)
        if nb_subjs == 1:
            nb_streamlines = len(batch_streamlines)

            print("One subject. Splitting the {} streamlines amongt GPUs."
                  .format(nb_streamlines))

            # We can just divide its streamlines
            subj_id = list(batch_idx_dict.keys())[0]
            group_ids = np.split(np.asarray(range(nb_streamlines)), nb_gpus)

            list_of_dicts = [{subj_id: slice(0, len(group_ids[i]))}
                             for i in range(nb_gpus)]
            list_of_streamlines = [[batch_streamlines[i] for i in
                                   group_ids[g]] for g in range(nb_gpus)]
            print("Sub-batch sizes: {}"
                  .format([len(sub_batch)
                           for sub_batch in list_of_streamlines]))
        elif nb_subjs % nb_gpus == 0:
            # We can just divide by subjects
            all_subjs = np.asarray(list(batch_idx_dict.keys()))
            print("Found {} subjects in the batch, to be devided into {} "
                  "GPUs. Unclear how to optimize this, we will try our best."
                  .format(len(all_subjs), nb_gpus))
            groups_of_subjs = np.split(all_subjs, nb_gpus)

            list_of_streamlines = []
            list_of_dicts = []
            for g in groups_of_subjs:
                nb_total = 0
                group_dict = {}
                streamlines = []
                for subj in g:
                    streamlines += batch_streamlines[batch_idx_dict[subj]]
                    nb = batch_idx_dict[subj].stop - batch_idx_dict[subj].start
                    group_dict.update({subj: slice(nb_total, nb_total + nb)})
                    nb_total += nb
                list_of_dicts.append(group_dict)
                list_of_streamlines.append(streamlines)

            print("Sub-batch sizes: {}"
                  .format([len(sub_batch)
                           for sub_batch in list_of_streamlines]))
        else:
            # Not sure what to do.
            # split subjects until all passed? (some GPUs will have much more
            # data than others)
            # split streamlines and don't take into consideration the subjects?
            # (some volumes could be loaded twice in different GPUs).
            raise NotImplementedError
    else:
        list_of_dicts = [batch_idx_dict]
        list_of_streamlines = [batch_streamlines]

    return list_of_dicts, list_of_streamlines, batch_loader


def _create_sampler_and_loader(dataset):

    # Initialize batch sampler
    logging.debug('\nInitializing sampler...')
    batch_sampler = create_test_batch_sampler(
        dataset, batch_size=batch_size,
        batch_size_units='nb_streamlines', log_level=logging.WARNING)

    batch_loader = create_batch_loader(dataset, log_level=logging.WARNING)

    return batch_sampler, batch_loader


if __name__ == '__main__':
    manual_test_multi_gpu()
