#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
import logging
from os import path

from torch.utils.data.dataloader import DataLoader

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment.batch_samplers import (BatchStreamlinesSamplerWithInputs)


def parse_args():
    """
    Convert "raw" streamlines from a hdf5 to torch data or to PackedSequences
    via our MultiSubjectDataset class. Test some properties.
    The MultiSubjectDataset is used during training, in the trainer_abstract.
    """
    p = argparse.ArgumentParser(description=str(parse_args.__doc__),
                                formatter_class=RawTextHelpFormatter)
    p.add_argument('hdf5_filename',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects.')
    return p.parse_args()


def test_sampler(fake_dataset, chunk_size, batch_size, step_size,
                 compress=False):
    # Initialize batch sampler
    print('Initializing sampler...')
    training_set = fake_dataset.training_set

    batch_sampler = BatchStreamlinesSamplerWithInputs(
        training_set, 'streamlines', 'input', chunk_size=chunk_size,
        max_batch_size=batch_size, rng=1234,
        step_size=step_size, compress=compress, nb_subjects_per_batch=1,
        cycles=1, neighborhood_type=None, neighborhood_radius=None,
        split_ratio=0, noise_gaussian_size=0, noise_gaussian_variability=0,
        reverse_ratio=0, wait_for_gpu=True, normalize_directions=True,
        nb_previous_dirs=1)

    # Use it in the dataloader
    # it says error, that the collate_fn should receive a list, but it is
    # supposed to work with dict too.
    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/1918
    print('Initializing DataLoader')
    dataloader = DataLoader(training_set, batch_sampler=batch_sampler,
                            collate_fn=batch_sampler.load_batch)
    print(dataloader)

    batch_generator = batch_sampler.__iter__()

    # Loop on batches
    i = 0
    total_sampled_ids_sub0 = []
    for batch in batch_generator:
        subj0 = batch[0]
        (subj0_id, subj0_streamlines) = subj0
        print('Batch # {}: nb sampled streamlines subj 0 was {}'
              .format(i, len(subj0_streamlines)))
        total_sampled_ids_sub0.append(batch[0])
        i = i + 1
        if i > 3:
            break


def test_non_lazy():
    print("\n\n========= NON-LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, lazy=False,
                                       experiment_name='test',
                                       taskman_managed=True, cache_size=None)
    fake_dataset.load_data()

    print('\n=============================Test with batch size 1000')
    test_sampler(fake_dataset, 256, 1000, None)

    logging.getLogger().setLevel('INFO')

    print('\n=============================Test with batch size 1000000')
    test_sampler(fake_dataset, 256, 100000, None)

    print('\n===========================Test with batch size 10000 + resample')
    test_sampler(fake_dataset, 256, 10000, 0.5)

    print('\n===========================Test with batch size 10000 + compress'
          '(Batch size should be equal to chunk size: 256)')
    test_sampler(fake_dataset, 256, 10000, None, True)


def test_lazy():
    print("\n\n========= LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, lazy=True,
                                       experiment_name='test',
                                       taskman_managed=True, cache_size=1)
    fake_dataset.load_data()

    print('\n=============================Test with batch size 1000')
    test_sampler(fake_dataset, 256, 1000, None)

    print('\n===========================Test with batch size 10000 + resample')
    test_sampler(fake_dataset, 256, 10000, 0.5)

    print('\n===========================Test with batch size 10000 + compress'
          '(Batch size should be equal to chunk size: 256)')
    test_sampler(fake_dataset, 256, 10000, None, True)


if __name__ == '__main__':
    args = parse_args()

    if not path.exists(args.hdf5_filename):
        raise ValueError("The hdf5 file ({}) was not found!"
                         .format(args.hdf5_filename))

    logging.basicConfig(level='DEBUG')

    test_non_lazy()
    print('\n\n')

    logging.basicConfig(level='DEBUG')

    test_lazy()
