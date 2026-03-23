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

import numpy as np

from dwi_ml.general.training.utils.monitoring import BatchHistoryMonitor
import torch
from dwi_ml.io_utils import verify_which_model_in_path
from dwi_ml.general.models.projects import find_transformer_class
from dwi_ml.projects.Transformers.transformer_trainer import TransformerTrainer
from scilpy.io.utils import add_verbose_arg

from dwi_ml.general.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.general.experiment_utils.prints import format_dict_to_str
from dwi_ml.general.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.general.training.batch_samplers import DWIMLBatchIDSampler


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiment_path',
                   help='Name for the experiment.')
    p.add_argument('out_experiment',
                   help="Name of the fixed experiment.")
    p.add_argument('--hdf5_path',
                   help="Required only if previous hdf5 has been moved.")

    add_verbose_arg(p)

    return p


def _replace(params, old_key, new_key):
    if old_key in params:
        logging.warning("Replacing old key {} with new key {}"
                        .format(old_key, new_key))
        params[new_key] = params[old_key]
        del params[old_key]
    return params


def _remove(params, old_key):
    if old_key in params:
        logging.warning("Deleting option {}".format(old_key))
        del params[old_key]
    return params


def fix_model_params(params):
    print("\n--------------- In model params:")

    # deleted start_from_copy_prev
    params = _remove(params, 'start_from_copy_prev')

    # deleted options compress_loss, weight_loss_with_angle (in dg)
    params['dg_args'] = _remove(params['dg_args'], 'compress_loss')
    params['dg_args'] = _remove(params['dg_args'], 'compress_eps')
    params['dg_args'] = _remove(params['dg_args'], 'weight_loss_with_angle')

    # ------
    # Even older models:
    # ------
    params = _replace(params, 'embedding_key_x', 'input_embedding_key')
    params = _replace(params, 'd_model', 'input_embedded_size')  # NOT the same if TTST or TTO model, but I don't think I have any...

    # Added cnn options
    if 'nb_cnn_filters' not in params:
        logging.warning("Adding options for CNN")
        params['nb_cnn_filters'] = None
    if 'kernel_size' not in params:
        logging.warning("Adding options for CNN")
        params['kernel_size'] = None

    # Neighborhood management modified
    r = params['neighborhood_radius']
    if isinstance(r, list):
        logging.warning("Updating neighborhood radius management")
        params['neighborhood_radius'] = len(r)
        params['neighborhood_resolution'] = r[0]
        if len(r) > 1:
            if not np.all(np.diff(r) == r[0]):
                raise ValueError("Now, neighborhood must have the same "
                                 "resolution between each layer of "
                                 "neighborhood. But got: {}".format(r))

    return params

def fix_model_state(cls, model_dir):
    print("\n--------------- In model state ():".format(model_dir))
    model_state = cls._load_state(model_dir)
    model_state = _replace(model_state, 'embedding_layer_x.linear.weight', 
                           'input_embedding_layer.linear.weight')
    model_state = _replace(model_state, 'embedding_layer_x.linear.bias', 
                           'input_embedding_layer.linear.bias')
    torch.save(model_state, os.path.join(model_dir, "model_state.pkl"))


def fix_checkpoint_params(checkpoint_state):
    """
    Updating non-specific params (trainer, batch loader, etc)
    """
    # Nothing to do for Transformer, I had not used it before updates.
    # See l2t_update_deprecated_exp if I have issues.
    print("\n--------------- In checkpoint params:")
    # 1) Dataset params: better use a --hdf5_path to fix.

    # 2) Trainer params : nb_steps --> nb_segments, no more lr_decrease_params
    checkpoint_state['params_for_init'] = _replace(
        checkpoint_state['params_for_init'], 'tracking_phase_nb_steps_init',
        'tracking_phase_nb_segments_init')
    checkpoint_state['params_for_init'] = _remove(
        checkpoint_state['params_for_init'], 'lr_decrease_params')

    # 3) Monitors: new var ever_max, ever_min
    for k in checkpoint_state['current_states'].keys():
        if isinstance(checkpoint_state['current_states'][k], dict) and \
                'average_per_epoch' in checkpoint_state['current_states'][k].keys():
            # Found a monitor
            if 'ever_min' not in checkpoint_state['current_states'][k]:
                logging.warning("Setting false min, max for monitor {}. "
                                # But not sure this is ever used.... TODO
                                .format(k))
                checkpoint_state['current_states'][k]['ever_min'] = -np.inf
                checkpoint_state['current_states'][k]['ever_max'] = np.inf
    
    # New monitor
    if 'unclipped_grad_norm_monitor_state' not in checkpoint_state['current_states']:
        logging.warning("Adding a new monitor (unclipped_grad_norm)")
        monitor = BatchHistoryMonitor('unclipped_grad_norm_monitor', 
                                       weighted=False)
        checkpoint_state['current_states'][monitor.name + '_state'] = monitor.get_state()

    return checkpoint_state

def fix_checkpoint_rng_state(checkpoint_state):
    print("\n--------------- In checkpoint state:")

    assert 'torch_cuda_state' in checkpoint_state['current_states']
    checkpoint_state['current_states']['torch_cuda_state'] = None
    return checkpoint_state


def load_both_models_checkpoint_best_and_fix(args):
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
    logging.info("Loaded params:\n{}".format(format_dict_to_str(params)))

    # 3) Fixing params
    params = fix_model_params(params)
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

    # 5. Fixing model state
    fix_model_state(cls, fixed_checkpoint_model_dir)
    fix_model_state(cls, fixed_best_model_dir)


    # Verify that both models can be loaded
    _ = cls.load_model_from_params_and_state(
        fixed_checkpoint_model_dir)
    model = cls.load_model_from_params_and_state(
        fixed_best_model_dir)

    return model


def load_checkpoint_and_fix(args, model):
    # Fixing trainer

    # Loading checkpoint
    experiments_path, experiment_name = os.path.split(args.experiment_path)
    checkpoint_state = TransformerTrainer.load_params_from_checkpoint(
        experiments_path, experiment_name)

    # Verify hdf5
    dataset_params = checkpoint_state['dataset_params']['training set']
    if args.hdf5_path is None and not os.path.isfile(dataset_params['hdf5_file']):
        raise ValueError("hdf5 file has been deleted or moved ({})\n"
                         "Please set a path to a new hdf5.")
    elif args.hdf5_path is not None:
        # Get the hdf5
        dataset = prepare_multisubjectdataset(
            argparse.Namespace(**{'hdf5_file': args.hdf5_path,
                                  'lazy': True,
                                  'cache_size': 1}))
    else:
        dataset_params['lazy'] = True
        dataset = prepare_multisubjectdataset(
            argparse.Namespace(**dataset_params))

    # Fixing checkpoint
    checkpoint_state = fix_checkpoint_params(checkpoint_state)
    checkpoint_dir = os.path.join(args.out_experiment, "checkpoint")
    torch.save(checkpoint_state,
               os.path.join(checkpoint_dir, "checkpoint_state.pkl"))

    # Init stuff will succeed if ok.
    batch_sampler = DWIMLBatchIDSampler.init_from_checkpoint(
        dataset, checkpoint_state['batch_sampler_params'])
    batch_loader = DWIMLBatchLoaderOneInput.init_from_checkpoint(
            dataset, model, checkpoint_state['batch_loader_params'])
    experiments_path, experiment_name = os.path.split(args.out_experiment)
    if experiments_path == '':
        experiments_path = '/'

    try:
        _ = TransformerTrainer.init_from_checkpoint(
            model, experiments_path, experiment_name,
            batch_sampler, batch_loader,
            checkpoint_state, new_patience=None, new_max_epochs=None,
            log_level='WARNING')
    except RuntimeError as e:
        if 'RNG state' in str(e):
            logging.warning("RNG error in the checkpoint, due to pytorch "
                            "version, probably. Will ignore the RNG state. "
                            "Will probably not really influence anything")
            checkpoint_path = os.path.join(
                experiments_path, experiment_name, "checkpoint",
                "checkpoint_state.pkl")
            checkpoint_state = torch.load(checkpoint_path, weights_only=False)
            checkpoint_state = fix_checkpoint_rng_state(checkpoint_state)

            print("Saving new state as ", checkpoint_path)
            torch.save(checkpoint_state, checkpoint_path)
            checkpoint_state = torch.load(checkpoint_path, weights_only=False)
    
            _ = TransformerTrainer.init_from_checkpoint(
                    model, experiments_path, experiment_name,
                    batch_sampler, batch_loader,
                    checkpoint_state, new_patience=None, new_max_epochs=None,
                    log_level='WARNING')
        else:
            raise RuntimeError(e)


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # General logging (ex, scilpy: Warning)
    logging.getLogger().setLevel(args.verbose)

    if not os.path.exists(args.experiment_path):
        raise FileNotFoundError("Experiment not found ({})."
                                .format(args.experiment_path))
    if os.path.exists(args.out_experiment):
        raise FileExistsError("Out experiment already exists! ({})."
                              .format(args.out_experiment))
    shutil.copytree(args.experiment_path, args.out_experiment)

    model = load_both_models_checkpoint_best_and_fix(args)
    load_checkpoint_and_fix(args, model)

    print("Out experiment {} should now be usable!"
          .format(args.out_experiment))


if __name__ == '__main__':
    main()
