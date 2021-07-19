#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
           Train a model for your favorite experiment

Remove or add parameters to fit your needs. You should change your yaml file
accordingly.

- Change init_dataset() if the MultiSubjectDataset doesn't fit your needs
- Change init_sampler() for the BatchSampler version that fits your needs
- Implement build_model()
- Change the DWIMLTrainer if it doesn't fit your needs.
"""

import argparse
import logging
import os
from os import path

import yaml

from dwi_ml.experiment.monitoring import EarlyStoppingError
from dwi_ml.training.checks_for_experiment_parameters import (
    check_all_experiment_parameters, check_logging_level)
from dwi_ml.training.trainer_abstract import (DWIMLTrainer,
                                              load_params_from_checkpoint)

# These are model-dependant. Choose the best classes and functions for you
# Change this init_dataset function if you prefer! This loads either a
# MultiSubjectDataset or a LazyMultiSubjectDataset.
from dwi_ml.data.dataset.multi_subject_containers import init_dataset
# Change this init_batch_sampler if you prefer!
# This is one possibility, others could be implemented.
from dwi_ml.model.batch_samplers import (
    BatchSequencesSamplerOneInputVolume as ChosenBatchSampler)
# Implement this:
from dwi_ml.model.models import init_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiment_path',
                   help='Path where to save your experiment. Complete path '
                        'will be experiment_path/experiment_name.')
    p.add_argument('--hdf5_filename',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects. If a '
                        'checkpoint exists, this information is already '
                        'contained in the checkpoint and is not necessary.')
    p.add_argument('--parameters_filename',
                   help='Experiment configuration YAML filename. See '
                        'please_copy_and_adapt/training_parameters.yaml for '
                        'an example. If a checkpoint exists, this information '
                        'is already contained in the checkpoint and is not '
                        'necessary.')
    p.add_argument('--experiment_name',
                   help='If given, name for the experiment. Else, model will '
                        'decide the name to give based on time of day.')
    p.add_argument('--override_checkpoint_patience', type=int,
                   help='If a checkpoint exists, patience can be increased '
                        'to allow experiment to continue if the allowed '
                        'number of bad epochs has been previously reached.')

    arguments = p.parse_args()

    return arguments


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

    # Verify if a checkpoint has been saved. Else create an experiment.
    if path.exists(os.path.join(args.experiment_path, args.experiment_name,
                                "checkpoint")):
        # Loading checkpoint
        print("Experiment checkpoint folder exists, resuming experiment!")
        checkpoint_state = load_params_from_checkpoint(args.experiment_path)
        if args.parameters_filename:
            logging.warning('Resuming experiment from checkpoint. Yaml file '
                            'option was not necessary and will not be used!')
        if args.hdf5_filename:
            logging.warning('Resuming experiment from checkpoint. hdf5 file '
                            'option was not necessary and will not be used!')

        # Stop now if early stopping was triggered.
        if (args.override_checkpoint_patience is None and
                checkpoint_state['early_stopping_state']['n_bad_epochs'] >=
                checkpoint_state['early_stopping_state']['patience']):
            raise EarlyStoppingError("Resumed experiment was stopped because "
                                     "of early stopping, increase patience in "
                                     "order to resume training!")

        # Instantiate dataset classes
        training_dataset = \
            init_dataset(**checkpoint_state['train_data_params'])
        validation_dataset = \
            init_dataset(**checkpoint_state['valid_data_params'])

        # Instantiate batch sampler classes
        training_batch_sampler = ChosenBatchSampler(
            training_dataset,
            **checkpoint_state['train_sampler_params'])
        validation_batch_sampler = ChosenBatchSampler(
            validation_dataset,
            **checkpoint_state['valid_sampler_params'])

        # Instantiate model
        # Remember that you have training_batch_sampler.compute_feature_sizes()
        model = init_model()

        # Instantiate trainer
        experiment = DWIMLTrainer.init_from_checkpoint(
            training_batch_sampler, validation_batch_sampler, model,
            checkpoint_state)
    else:
        # Perform checks
        # We have decided to use yaml for a more rigorous way to store
        # parameters, compared, say, to bash. However, no argparser is used so
        # we need to make our own checks.
        all_params = check_all_experiment_parameters(yaml_parameters)

        # Instantiate dataset classes
        data_params = all_params
        data_params['subjs_set'] = 'training_subjs'
        training_dataset = init_dataset(**data_params)
        data_params['subjs_set'] = 'validation_subjs'
        validation_dataset = init_dataset(**data_params)

        # Instantiate batch sampler classes
        training_batch_sampler = ChosenBatchSampler(training_dataset,
                                                    **all_params)
        validation_batch_sampler = ChosenBatchSampler(validation_dataset,
                                                      **all_params)

        # Instantiate model
        # Remember that you have training_batch_sampler.compute_feature_sizes()
        model = init_model(**all_params)

        # Instantiate trainer
        experiment = DWIMLTrainer(
            training_batch_sampler, validation_batch_sampler, model,
            args.experiment_path, args.experiment_name,
            **all_params)

    #####
    # Run (or continue) the experiment
    #####
    try:
        experiment.train()
    except EarlyStoppingError as e:
        print(e)

    experiment.save_model()

    print("Script terminated successfully. Saved experiment in folder : ")
    print(experiment.experiment_dir)


if __name__ == '__main__':
    main()
