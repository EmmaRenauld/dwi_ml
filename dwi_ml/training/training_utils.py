# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment.timer import Timer
from dwi_ml.models.batch_samplers import BatchStreamlinesSampler1IPV
from dwi_ml.utils import format_dict_to_str


def parse_args_train_model():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--print_description', action='store_true',
                   help="Print the complete description will all parameters "
                        "in the yaml_parameters.")
    p.add_argument('--experiment_path', default='./', metavar='p',
                   help='Path where to save your experiment. \nComplete path '
                        'will be experiment_path/experiment_name. Default: ./')
    p.add_argument('--hdf5_file', metavar='h',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training and validation subjects.\n'
                        '**If a checkpoint exists, this information is '
                        'already contained in the \ncheckpoint and is not '
                        'necessary. Else, mandatory.')
    p.add_argument('--yaml_parameters', metavar='p',
                   help='Experiment configuration YAML filename. \n'
                        '    - See '
                        'please_copy_and_adapt/training_parameters.yaml for '
                        'an example.\n'
                        '    - Use --print_description for more information.\n'
                        '**If a checkpoint exists, this information '
                        'is already contained in the \ncheckpoint and is not '
                        'necessary. Else, mandatory.')
    p.add_argument('--experiment_name', metavar='n',
                   help='If given, name for the experiment. Else, model will '
                        'decide the name to \ngive based on time of day.')
    p.add_argument('--override_checkpoint_patience', type=int, metavar='new_p',
                   help='If a checkpoint exists, patience can be increased '
                        'to allow experiment \nto continue if the allowed '
                        'number of bad epochs has been previously reached.')
    p.add_argument('--override_checkpoint_max_epochs', type=int,
                   metavar='new_max',
                   help='If a checkpoint exists, max_epochs can be increased '
                        'to allow experiment \nto continue if the allowed '
                        'number of epochs has been previously reached.')
    p.add_argument('--logging', choices=['error', 'warning', 'info', 'debug'],
                   default='info',
                   help="Logging level. One of ['error', 'warning', 'info', "
                        "'debug']. Default: Info.")
    p.add_argument('--comet_workspace', metavar='w',
                   help='Your comet workspace. If not set, comet.ml will not '
                        'be used. See our \ndocs/Getting Started for more '
                        'information on comet and its API key.')
    p.add_argument('--comet_project',
                   help='Send your experiment to a specific comet.ml project. '
                        'If not set, it will \nbe sent to Uncategorized '
                        'Experiments.')

    return p


def check_unused_args_for_checkpoint(args, project_specific_args):
    all_args = ['hdf5_file', 'yaml_parameters'].extend(project_specific_args)
    for s in all_args:
        if args[s]:
            logging.warning('Resuming experiment from checkpoint. {} '
                            'option was not necessary and will not be used!'
                            .format(s))


def prepare_data(dataset_params):
    """
    Instantiate a MultiSubjectDataset and load data.
    """
    with Timer("\n\nPreparing testing and validation sets",
               newline=True, color='blue'):
        dataset = MultiSubjectDataset(**dataset_params)
        dataset.load_data()

        logging.info("\n\nDataset attributes: \n" +
                     format_dict_to_str(dataset.attributes))

    return dataset


def prepare_batch_sampler_1i_pv(dataset, train_sampler_params,
                                valid_sampler_params):
    """
    Instantiate a BatchStreamlinesSampler1IPV batch sampler

    (1i_pv = one input + previous_dirs)
    """
    # Instantiate batch
    # In this example, using one input volume, the first one.
    volume_group_name = train_sampler_params['input_group_name']
    streamline_group_name = train_sampler_params['streamline_group_name']
    with Timer("\n\nPreparing batch samplers with volume: '{}' and "
               "streamlines '{}'"
                       .format(volume_group_name, streamline_group_name),
               newline=True, color='green'):
        # Batch samplers could potentially be set differently between training
        # and validation. Modify the code if you wish.
        training_batch_sampler = BatchStreamlinesSampler1IPV(
            dataset.training_set, **train_sampler_params)
        validation_batch_sampler = BatchStreamlinesSampler1IPV(
            dataset.validation_set, **valid_sampler_params)
        logging.info("\n\nTraining batch sampler attributes: \n" +
                     format_dict_to_str(training_batch_sampler.attributes))
        logging.info("\n\nValidation batch sampler attributes: \n" +
                     format_dict_to_str(training_batch_sampler.attributes))

    return training_batch_sampler, validation_batch_sampler
