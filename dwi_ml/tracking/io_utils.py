# -*- coding: utf-8 -*-
import logging
import os
from argparse import ArgumentParser

from dipy.io.stateful_tractogram import (Space, Origin, set_sft_logger_level,
                                         StatefulTractogram)
from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np

from scilpy.tracking.seed import SeedGenerator

from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.io_utils import add_arg_existing_experiment_path, add_memory_args
from dwi_ml.testing.utils import add_args_testing_subj_hdf5
from dwi_ml.tracking.tracking_mask import TrackingMask
from dwi_ml.tracking.tracker import DWIMLAbstractTracker

ALWAYS_VOX_SPACE = Space.VOX
ALWAYS_CORNER = Origin('corner')


def add_tracking_options(p: ArgumentParser):
    add_arg_existing_experiment_path(p)
    add_args_testing_subj_hdf5(p, optional_hdf5=True,
                               ask_input_group=True)

    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')
    p.add_argument('seeding_mask_group',
                   help="Seeding mask's volume group in the hdf5.")

    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument('--algo', choices=['det', 'prob'], default='det',
                         help="Tracking algorithm (det or prob). Must be "
                              "implemented in the chosen model. \n[det]")
    track_g.add_argument('--step_size', type=float,
                         help='Step size in mm. Default: using the step size '
                              'saved in the model parameters.')
    track_g.add_argument('--track_forward_only', action='store_true',
                         help="If set, tracks in one direction only (forward) "
                              "given the initial seed.")
    track_g.add_argument('--mask_interp', default='nearest',
                         choices=['nearest', 'trilinear'],
                         help="Mask interpolation: nearest-neighbor or "
                              "trilinear. [%(default)s]")
    track_g.add_argument('--data_interp', default='trilinear',
                         choices=['nearest', 'trilinear'],
                         help="Input data interpolation: nearest-neighbor or "
                              "trilinear. [%(default)s]")

    stop_g = p.add_argument_group("Stopping criteria")
    stop_g.add_argument('--min_length', type=float, default=10.,
                        metavar='m',
                        help='Minimum length of a streamline in mm. '
                             '[%(default)s]')
    stop_g.add_argument('--max_length', type=float, default=300.,
                        metavar='M',
                        help='Maximum length of a streamline in mm. '
                             '[%(default)s]')
    stop_g.add_argument('--tracking_mask_group', metavar='key',
                        help="Tracking mask's volume group in the hdf5.")
    stop_g.add_argument('--theta', metavar='t', type=float, default=90,
                        help="Stopping criterion during propagation: "
                             "tracking is stopped when a direction is \nmore "
                             "than an angle theta from preceding direction. "
                             "[%(default)s]")
    stop_g.add_argument('--eos_stop', metavar='prob',
                        help="Stopping criterion if a EOS value was learned "
                             "during training. For all models, \ncan be a "
                             "probability (default 0.5). For classification "
                             "models, can also be the \nkeyword 'max', which "
                             "will stop the propagation if the EOS class is "
                             "the class \nwith maximal probability, no matter "
                             "its value.")
    stop_g.add_argument(
        '--discard_last_point', action='store_true',
        help="If set, discard the last point (once out of the tracking mask) "
             "of the \nstreamline. Default: do not discard them; append them. "
             "This is the default in \nDipy too. Note that points obtained "
             "after an invalid direction (based on the \npropagator's "
             "definition of invalid; ex when angle is too sharp or "
             "sh_threshold \nis not reached) are never added.")

    r_g = p.add_argument_group('  Random seeding options')
    r_g.add_argument('--rng_seed', type=int,
                     help='Initial value for the random number generator. '
                          '[%(default)s]')
    r_g.add_argument(
        '--skip', type=int, default=0,
        help="Skip the first N random numbers. Useful if you want to create "
             "new streamlines to \nadd to a tractogram previously created "
             "with a fixed --rng_seed. Ex: If \ntractogram_1 was created "
             "with -nt 1,000,000, you can create tractogram_2 with \n"
             "--skip 1,000,000.")

    # Memory options:
    m_g = add_memory_args(p, add_lazy_options=True,
                          add_multiprocessing_option=True,
                          add_rng=True)
    m_g.add_argument('--simultaneous_tracking', type=int, default=1,
                     metavar='nb',
                     help='Track n streamlines at the same time. Intended for '
                          'GPU usage. Default = 1 \n(no simultaneous '
                          'tracking).')

    return track_g


def prepare_seed_generator(parser, args, hdf_handle):
    """
    Prepares a SeedGenerator from scilpy's library. Returns also some header
    information to allow verifications.
    """
    if args.subj_id not in hdf_handle:
        raise ValueError("Subject {} not found in the HDF5 file."
                         .format(args.subj_id))
    if args.seeding_mask_group not in hdf_handle[args.subj_id]:
        raise ValueError("Seeding mask {} not found the subject's HDF group."
                         .format(args.seeding_mask_group))
    seeding_group = hdf_handle[args.subj_id][args.seeding_mask_group]
    seed_data = np.array(seeding_group['data'], dtype=np.float32)
    seed_res = np.array(seeding_group.attrs['voxres'], dtype=np.float32)
    affine = np.array(seeding_group.attrs['affine'], dtype=np.float32)
    ref = nib.Nifti1Image(seed_data, affine)

    seed_generator = SeedGenerator(seed_data, seed_res, space=ALWAYS_VOX_SPACE,
                                   origin=ALWAYS_CORNER)

    if len(seed_generator.seeds_vox_corner) == 0:
        parser.error('Seed mask "{}" does not have any voxel with value > 0.'
                     .format(args.in_seed))

    if args.npv:
        # Note. Not really nb seed per voxel, just in average.
        nbr_seeds = len(seed_generator.seeds_vox_corner) * args.npv
    elif args.nt:
        nbr_seeds = args.nt
    else:
        # Setting npv = 1.
        nbr_seeds = len(seed_generator.seeds_vox_corner)

    seed_header = nib.Nifti1Image(seed_data, affine).header

    return seed_generator, nbr_seeds, seed_header, ref


def prepare_tracking_mask(hdf_handle, tracking_mask_group, subj_id,
                          mask_interp):
    """
    Prepare the tracking mask as a DataVolume from scilpy's library. Returns
    also some header information to allow verifications.
    """
    if subj_id not in hdf_handle:
        raise KeyError("Subject {} not found in {}. Possible subjects are: {}"
                       .format(subj_id, hdf_handle, list(hdf_handle.keys())))
    if tracking_mask_group not in hdf_handle[subj_id]:
        raise KeyError("HDF group '{}' not found for subject {} in hdf file {}"
                       .format(tracking_mask_group, subj_id, hdf_handle))
    tm_group = hdf_handle[subj_id][tracking_mask_group]
    mask_data = np.array(tm_group['data'], dtype=np.float64).squeeze()
    # mask_res = np.array(tm_group.attrs['voxres'], dtype=np.float32)
    affine = np.array(tm_group.attrs['affine'], dtype=np.float32)
    ref = nib.Nifti1Image(mask_data, affine)

    mask = TrackingMask(mask_data.shape, mask_data, mask_interp)

    return mask, ref


def track_and_save(tracker: DWIMLAbstractTracker, args, ref):
    if args.save_seeds:
        name, ext = os.path.splitext(args.out_tractogram)
        if ext != '.trk':
            raise ValueError("Cannot save seeds! (data per streamline not "
                             "saved with extension {}). Please change out "
                             "filename to .trk".format(ext))

    with Timer("\nTracking...", newline=True, color='blue'):
        streamlines, seeds = tracker.track()

        logging.debug("Tracked {} streamlines (out of {} seeds). Now saving..."
                      .format(len(streamlines), tracker.nbr_seeds))

    if len(streamlines) == 0:
        logging.warning("No streamlines created! Not saving tractogram!")
        return

    # save seeds if args.save_seeds is given
    # Seeds must be saved in voxel space (ok!), but origin: center, if we want
    # to use scripts such as scil_compute_seed_density_map.
    if args.save_seeds:
        print("Saving seeds in data_per_streamline.")
        seeds = [np.asarray(seed) - 0.5 for seed in seeds]  # to_center
        data_per_streamline = {'seeds': seeds}
    else:
        data_per_streamline = {}

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    logging.info("Saving resulting tractogram to {}"
                 .format(args.out_tractogram))
    sft = StatefulTractogram(streamlines, ref, space=ALWAYS_VOX_SPACE,
                             origin=ALWAYS_CORNER,
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, args.out_tractogram, bbox_valid_check=False)
