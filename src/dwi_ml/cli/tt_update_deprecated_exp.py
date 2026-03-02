#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copies an existing experiment to another folder, updating deprecated values.

Useful for Emmanuelle's work! :)

"""
import argparse
import json
import logging
import os
import shutil

import torch
from dwi_ml.io_utils import verify_which_model_in_path
from dwi_ml.models.projects.transformer_models import find_transformer_class
from dwi_ml.training.projects.transformer_trainer import TransformerTrainer
from scilpy.io.utils import add_verbose_arg

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiment_path',
                   help='Name for the experiment.')
    p.add_argument('out_experiment',
                   help="Name of the fixed experiment.")
    p.add_argument('--new_hdf5',
                   help="Required only if previous hdf5 has been moved.")

    add_verbose_arg(p)

    return p


def _replace(params, old_key, new_key):
    if old_key in params:
        logging.warning("Replacing old key {} with new key {}"
                        .format(old_key, new_key))
        params[new_key] = params[old_key]
        del params[old_key]
    else:
        logging.warning("Expected to find deprecated key {} (to be replaced "
                        "by {}), but did not find it. Skipping."
                        .format(old_key, new_key))
    return params


def fix_deprecated_transformers_params(params):
    # deleted start_from_copy_prev
    if 'start_from_copy_prev' in params:
        logging.warning("Deleting option start_from_copy_prev")
        del params['start_from_copy_prev']

    # deleted options compress_loss, weight_loss_with_angle (in dg)
    if 'compress_loss' in params['dg_args']:
        logging.warning("Deleting option compress_loss")
        del params['dg_args']['compress_loss']
    if 'compress_eps' in params['dg_args']:
        logging.warning("Deleting option compress_eps")
        del params['dg_args']['compress_eps']
    if 'weight_loss_with_angle' in params['dg_args']:
        logging.warning("Deleting option weight_loss_with_angle")
        del params['dg_args']['weight_loss_with_angle']

    return params


def fix_other_checkpoint_params(checkpoint_state):
    """
    Updating non-specific params (trainer, batch loader, etc)
    """
    # Nothing to do for Transformer, I had not used it before updates.
    # See l2t_update_deprecated_exp if I have issues.

    return checkpoint_state


def fix_model_parameters_json_file(args):
    """
    Updating this specific model's params (Transformers)
    """

    # 1) Loading params from checkpoint's latest model
    model_dir = os.path.join(args.experiment_path, 'checkpoint', 'model')
    model_type = verify_which_model_in_path(model_dir)
    print("Model's class: {}".format(model_type))
    cls = find_transformer_class(model_type)
    params = cls._load_params(model_dir)

    # 2) Loading params from best model
    model_dir = os.path.join(args.experiment_path, 'best_model')
    params2 = cls._load_params(model_dir)

    # Verifying that they fit
    assert params == params2, ("Unexpected error. Parameters in the "
                               "checkpoint dir and in the best_model dir "
                               "should be the same. Did you modify the "
                               "parameters.json files?\n"
                               "Checkpoint params: \n"
                               "{}\n"
                               "--------------------"
                               "Best model params: \n"
                               "{}"
                               .format(format_dict_to_str(params),
                                       format_dict_to_str(params2)))
    del params2
    logging.debug("Loaded params:\n{}".format(format_dict_to_str(params)))

    # 3) Fixing params
    params = fix_deprecated_transformers_params(params)
    print("\n\n----------------Fixed the model parameters ----------------\n"
          "Reformated model's params:\n " + format_dict_to_str(params))

    # 4) Save fixed params in both parameters files
    fixed_checkpoint_model_dir = os.path.join(
        args.out_experiment, "checkpoint", "model")
    params_in_checkpoint = os.path.join(
        fixed_checkpoint_model_dir, "parameters.json")
    with open(params_in_checkpoint, 'w') as json_file:
        json_file.write(json.dumps(params, indent=4, separators=(',', ': ')))
    fixed_best_model_dir = os.path.join(args.out_experiment, "best_model")
    params_in_best_model = os.path.join(fixed_best_model_dir,
                                        "parameters.json")
    with open(params_in_best_model, 'w') as json_file:
        json_file.write(json.dumps(params, indent=4, separators=(',', ': ')))

    # Verify that both models can be loaded
    _ = cls.load_model_from_params_and_state(
        fixed_checkpoint_model_dir)
    model = cls.load_model_from_params_and_state(
        fixed_best_model_dir)

    return model


def fix_checkpoint(args, model):
    # Fixing trainer

    # Loading checkpoint
    experiments_path, experiment_name = os.path.split(args.experiment_path)
    checkpoint_state = TransformerTrainer.load_params_from_checkpoint(
        experiments_path, experiment_name)

    # Verify hdf5
    dataset_params = checkpoint_state['dataset_params']['training set']
    if not os.path.isfile(dataset_params['hdf5_file']):
        if args.new_hdf5 is None:
            raise ValueError("hdf5 file has been deleted or moved ({})\n"
                             "Please set a path to a new hdf5.")
        else:
            # Get the hdf5
            dataset = prepare_multisubjectdataset(
                argparse.Namespace(**{'hdf5_file': args.new_hdf5,
                                      'lazy': True,
                                      'cache_size': 1}))
            # Compare all values
            for k, v in dataset_params.items():
                if k not in ['set_name', 'hdf5_file', 'lazy']:
                    assert dataset.training_set.__getattribute__(k) == v, \
                        ("Value {} in old hdf5 (training set) was {} but is "
                         "{} in the new one!"
                         .format(k, v,
                                 dataset.training_set.__getattribute__(k)))
                    assert dataset.validation_set.__getattribute__(k) == v, \
                        ("Value {} in old hdf5 (validation set) was {} but is "
                         "{} in the new one!"
                         .format(k, v,
                                 dataset.training_set.__getattribute__(k)))

    elif args.new_hdf5 is not None:
        raise ValueError("We already have all required information from the "
                         "hdf5 at {}. We do not need a --new_hdf5.")
    else:
        # Ensure it was lazy
        dataset_params['lazy'] = True
        dataset = prepare_multisubjectdataset(
            argparse.Namespace(**dataset_params))

    # Fixing checkpoint
    checkpoint_state = fix_other_checkpoint_params(checkpoint_state)
    checkpoint_dir = os.path.join(args.out_experiment, "checkpoint")
    torch.save(checkpoint_state,
               os.path.join(checkpoint_dir, "checkpoint_state.pkl"))

    # Init stuff will succeed if ok.
    batch_sampler = DWIMLBatchIDSampler.init_from_checkpoint(
        dataset, checkpoint_state['batch_sampler_params'])
    batch_loader = DWIMLBatchLoaderOneInput.init_from_checkpoint(
            dataset, model, checkpoint_state['batch_loader_params'])
    experiments_path, experiment_name = os.path.split(args.out_experiment)
    _ = TransformerTrainer.init_from_checkpoint(
        model, experiments_path, experiment_name,
        batch_sampler, batch_loader,
        checkpoint_state, new_patience=None, new_max_epochs=None,
        log_level='WARNING')


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # General logging (ex, scilpy: Warning)
    logging.getLogger().setLevel(level=logging.WARNING)

    if not os.path.exists(args.experiment_path):
        raise FileNotFoundError("Experiment not found ({})."
                                .format(args.experiment_path))
    if os.path.exists(args.out_experiment):
        raise FileExistsError("Out experiment already exists! ({})."
                              .format(args.out_experiment))
    shutil.copytree(args.experiment_path, args.out_experiment)

    model = fix_model_parameters_json_file(args)

    if args.experiment_path != args.out_experiment:
        fix_checkpoint(args, model)

    print("Out experiment {} should now be usable!"
          .format(args.out_experiment))


if __name__ == '__main__':
    main()
