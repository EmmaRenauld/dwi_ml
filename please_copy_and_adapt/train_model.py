#!/usr/bin/env python
# -*- coding: utf-8 -*-

###########################
# Remove or add parameters to fit your needs. You should change your yaml file
# accordingly.
# Change DWIMLAbstractSequences for an implementation of your own model.
# It should be a child of this abstract class.
#
# Choose the batch sampler that fits your model
# So far, we have implemented one version, where
#     x: input = volume group named "input"
#     y: target = the streamlines
############################

""" Train a model for my favorite experiment"""

import argparse
import logging
from os import path

import torch
import yaml

from dwi_ml.data.dataset.multi_subject_containers import (
    LazyMultiSubjectDataset, MultiSubjectDataset)
from dwi_ml.training.checks_for_experiment_parameters import (
    check_all_experiment_parameters, check_logging_level)
from dwi_ml.training.trainer_abstract import DWIMLTrainerAbstractSequences

# This is model-dependant. Choose the best batch sampler for you:
from dwi_ml.model.batch_samplers import BatchSequencesSamplerOneInputVolume


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('hdf5_filename',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects.')
    p.add_argument('parameters_filename',
                   help='Experiment configuration YAML filename. See '
                        'please_copy_and_adapt/training_parameters.yaml for '
                        'an example.')
    p.add_argument('--experiment_name', default=None,
                   help='If given, name for the experiment. Else, model will '
                        'decide the name to give')

    arguments = p.parse_args()

    return arguments


def init_dataset(lazy, hdf5_filename):
    """
    Choose the class used to represent your data. (We hope that our
    MultisubjectDataset may fit everyone's needs. If it is not the case, you
    are welcome to contribute to dwi_ml. Or you may change these for
    your own implementations or for a child class.)

    These classes should both contain the following methods:
      load_training_data()
      toDo load_validation_data()
    """
    if lazy:
        dataset_cls = LazyMultiSubjectDataset
    else:
        dataset_cls = MultiSubjectDataset

    dataset = dataset_cls(hdf5_filename)
    return dataset


def init_batch_sampler(dataset, batch_size, rng_seed, max_n_subjects, cycles,
                       step_size, neighborhood_type, neighborhood_radius,
                       split_ratio, noise_size, noise_variability,
                       reverse_ratio, avoid_cpu, device, normalize_directions,
                       num_previous_dirs):
    """
    Choose the class used to represent your batch sampler. (This is model-
    dependant. You should choose the best sampler from our samplers in
    dwi_ml.model.batch_samplers. If none fits your needs, use your own
    implementation or a child class.)

    It should contain the following methods:
       ??
    """
    sampler_cls = BatchSequencesSamplerOneInputVolume

    sampler = sampler_cls(dataset, streamline_group_name='streamlines',
                          input_group_name='input', batch_size=batch_size,
                          rng=rng_seed, n_subjects_per_batch=max_n_subjects,
                          cycles=cycles, step_size=step_size,
                          neighborhood_type=neighborhood_type,
                          neighborhood_radius_vox=neighborhood_radius,
                          split_streamlines_ratio=split_ratio,
                          noise_gaussian_size=noise_size,
                          noise_gaussian_variability=noise_variability,
                          reverse_streamlines_ratio=reverse_ratio,
                          avoid_cpu_computations=avoid_cpu, device=device,
                          normalize_directions=normalize_directions,
                          nb_previous_dirs=num_previous_dirs)

    return dataset


def main():
    args = parse_args()

    # Check that all files exist
    if not path.exists(args.hdf5_filename):
        raise FileNotFoundError("The hdf5 file ({}) was not found!"
                                .format(args.hdf5_filename))
    if not path.exists(args.parameters_filename):
        raise FileNotFoundError("The Yaml parameters file was not found: {}"
                                .format(args.parameters_filename))

    # Load parameters
    with open(args.parameters_filename) as f:
        yaml_parameters = yaml.safe_load(f.read())

    # Initialize logger
    logging_level = check_logging_level(yaml_parameters['logging']['level'])
    logging.basicConfig(level=logging_level)
    logging.info(yaml_parameters)

    # Perform checks
    # We have decided to use yaml for a more rigorous way to store parameters,
    # compared, say, to bash. However, no argparser is used so we need to
    # make our own checks.
    (step_size, normalize_directions, noise_size, noise_variability,
     split_ratio, reverse_ratio, neighborhood_type, neighborhood_radius,
     nb_previous_dirs, max_epochs, patience, batch_size, n_subjects_per_batch,
     cycles, lazy, cache_manager, avoid_cpu_computations, num_cpu_workers,
     worker_interpolation, taskman_managed,
     rng_seed) = check_all_experiment_parameters(yaml_parameters)

    # Set device
    if avoid_cpu_computations and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Instantiate dataset class
    dataset = init_dataset(lazy, args.hdf5_filename)

    # Instantiate batch sampler class
    batch_sampler = init_batch_sampler(
        dataset, batch_size, rng_seed, n_subjects_per_batch, cycles,
        step_size, neighborhood_type, neighborhood_radius, split_ratio,
        noise_size, noise_variability, reverse_ratio, avoid_cpu_computations,
        device, normalize_directions, nb_previous_dirs)

    # Instantiate trainer class
    # (Change DWIMLAbstractSequences for your class.)
    # Then load dataset, build model, train and save
    experiment = DWIMLTrainerAbstractSequences()

    # Run the experiment
    experiment.load_dataset()
    experiment.build_model()
    experiment.train()
    experiment.save()


if __name__ == '__main__':
    main()
