# -*- coding: utf-8 -*-
import logging
from typing import List

import numpy as np
import torch

# We could try using nan instead of zeros for non-existing previous dirs...
DEFAULT_UNEXISTING_VAL = torch.zeros((1, 3), dtype=torch.float32)


def compute_n_previous_dirs(streamlines_dirs, nb_previous_dirs,
                            unexisting_val=DEFAULT_UNEXISTING_VAL,
                            device=torch.device('cpu'), point_idx=None):
    """
    Params
    ------
    streamline_dirs: list[torch.tensor]
        A list of length nb_streamlines with the streamline direction at each
        point. Each tensor is of size [(N-1) x 3]; where N is the length of the
        streamline. For each streamline: [dir1, dir2, dir3, ...]
        ** If one streamline contains no dirs (tensor = []), by default, we
        consider it had one point (with no existing previous_dirs).
    unexisting_val: torch.tensor:
        Tensor to use as n^th previous direction when this direction does not
        exist (ex: 2nd previous direction for the first point).
        Ex: torch.zeros((1, 3))
    device: torch device
    point_idx: int
        If given, gets the n previous directions for a given point of the
        streamline. If None, returns the previous directions at each point.
        Hint: can be -1.

    Returns
    -------
    previous_dirs: list[tensor]
        A list of length nb_streamlines. Each tensor is of size
        [N, nb_previous_dir x 3]; the n previous dirs at each
        point of the streamline. Order is 1st previous dir, 2nd previous dir,
        etc., (reading the streamline backward).
        For each streamline:
             [[dirx dirx ...], ---> idx0
             [dir1 dirx ...],  ---> idx1
             [dir2 dir1 ...]]  ---> idx2
             Where dirx is a "non-existing dir" (typically, [0,0,0])
             The length of each row (...) is self.nb_previous_dirs.
    """
    if nb_previous_dirs == 0:
        return None

    unexisting_val = unexisting_val.to(device, non_blocking=True)

    if point_idx:
        prev_dirs = _get_one_n_previous_dirs(
            streamlines_dirs, nb_previous_dirs, unexisting_val, point_idx)
    else:
        prev_dirs = _get_all_n_previous_dirs(streamlines_dirs,
                                             nb_previous_dirs, unexisting_val)

    return prev_dirs


def _get_all_n_previous_dirs(streamlines_dirs, nb_previous_dirs,
                             unexisting_val):
    """
    Summary: Builds vertically:
    first prev dir    |        second prev dir      | ...

    Non-existing val:
    (0,0,0)           |         (0,0,0)             |   (0,0,0)
      -               |         (0,0,0)             |   (0,0,0)
      -               |           -                 |   (0,0,0)

    + concat other vals:
     -                |            -                |  -
     dir_1            |            -                |  -
     dir_2            |           dir_1             |  -
    """
    n_previous_dirs = [
        torch.cat([
            torch.cat(
                (unexisting_val.repeat(min(len(dirs) + 1, i), 1),
                 dirs[:max(0, len(dirs) - i + 1)]))
            for i in range(1, nb_previous_dirs + 1)], dim=1)
        for dirs in streamlines_dirs
    ]

    return n_previous_dirs


def _get_one_n_previous_dirs(streamlines_dirs, nb_previous_dirs,
                             unexisting_val, point_idx):
    # Builds horizontally.

    # Ex: if point_idx == -1:
    # i=1 -->  last dir --> dirs[point_idx-i+1]
    # But if point_idx == 5:
    # i=1 -->  dir #4 --> dirs[point_idx - i]

    n_previous_dirs = [
        torch.cat([
            dirs[point_idx - i][None, :] if (point_idx >= 0 and i <= point_idx)
            else dirs[point_idx - i + 1][None, :] if (
                point_idx < 0 and i <= len(dirs) + 1 + point_idx)
            else unexisting_val
            for i in range(1, nb_previous_dirs + 1)], dim=1)
        for dirs in streamlines_dirs
    ]
    return n_previous_dirs


def compute_directions(streamlines):
    """
    Params
    ------
    batch_streamlines: list[np.array]
            The streamlines (after data augmentation)
    """
    if isinstance(streamlines, list):
        batch_directions = [torch.diff(s, n=1, dim=0) for s in streamlines]
    else:  # Tensor:
        batch_directions = torch.diff(streamlines, n=1, dim=0)

    return batch_directions


def normalize_directions(directions):
    """
    Params
    ------
    directions: list[tensor]
    """
    if isinstance(directions, torch.Tensor):
        # Not using /= because if this is used in forward propagation, backward
        # propagation will fail.
        directions = directions / torch.linalg.norm(directions, dim=-1,
                                                    keepdim=True)
    else:
        directions = [s / torch.linalg.norm(s, dim=-1, keepdim=True)
                      for s in directions]

    return directions


def compute_angles(line_dirs, degrees=False):
    one = torch.ones(1, device=line_dirs.device)

    line_dirs /= torch.linalg.norm(line_dirs, dim=-1, keepdim=True)
    cos_angles = torch.sum(line_dirs[:-1, :] * line_dirs[1:, :], dim=1)

    # Resolve numerical instability
    cos_angles = torch.minimum(torch.maximum(-one, cos_angles), one)
    angles = torch.arccos(cos_angles)

    if degrees:
        angles = torch.rad2deg(angles)
    return angles


def compress_streamline_values(
        streamlines: List = None, dirs: List = None, values: List = None,
        compress_eps: float = 1e-3):
    """
    Parameters
    ----------
    streamlines: List[Tensors]
        Streamlines' coordinates. If None, dirs must be given.
    dirs: List[Tensors]
        Streamlines' directions, optional. Useful to skip direction computation
        if already computed elsewhere.
    values: List[Tensors]
        If set, compresses the values rather than the streamlines themselves.
    compress_eps: float
        Angle (in degrees)
    """
    if streamlines is None and dirs is None:
        raise ValueError("You must provide either streamlines or dirs.")
    elif dirs is None:
        dirs = compute_directions(streamlines)

    if values is None:
        assert streamlines is not None
        # Compress the streamline itself with our technique.
        # toDo
        raise NotImplementedError("Code not ready")

    compress_eps = np.deg2rad(compress_eps)

    compressed_mean_loss = 0.0
    compressed_n = 0
    for loss, line_dirs in zip(values, dirs):
        if len(loss) < 2:
            compressed_mean_loss = compressed_mean_loss + torch.mean(loss)
            compressed_n += len(loss)
        else:
            # 1. Compute angles
            angles = compute_angles(line_dirs)

            # 2. Compress losses
            # By definition, the starting point is different from previous
            # and has an important meaning. Separating.
            compressed_mean_loss = compressed_mean_loss + loss[0]
            compressed_n += 1

            # Then, verifying other segments
            current_loss = 0.0
            current_n = 0
            for next_loss, next_angle in zip(loss[1:], angles):
                # toDO. Find how to skip loop
                if next_angle < compress_eps:
                    current_loss = current_loss + next_loss
                    current_n += 1
                else:
                    if current_n > 0:
                        # Finish the straight segment
                        compressed_mean_loss = \
                            compressed_mean_loss + current_loss / current_n
                        compressed_n += 1

                    # Add the point following a big curve separately
                    compressed_mean_loss = compressed_mean_loss + next_loss
                    compressed_n += 1

                    # Then restart a possible segment
                    current_loss = 0.0
                    current_n = 0

    return compressed_mean_loss / compressed_n, compressed_n


def weight_value_with_angle(values: List, streamlines: List = None,
                            dirs: List = None):
    """
    Parameters
    ----------
    values: List[Tensors]
        Value to weight with angle. Ex: losses.
    streamlines: List[Tensors]
        Streamlines' coordinates. If None, dirs must be given.
    dirs: List[Tensors]
        Streamlines' directions, optional. Useful to skip direction computation
        if already computed elsewhere.
    """
    if streamlines is None and dirs is None:
        raise ValueError("You must provide either streamlines or dirs.")
    elif dirs is None:
        dirs = compute_directions(streamlines)

    zero = torch.as_tensor(0.0, device=dirs[0].device)
    for i, line_dirs in enumerate(dirs):
        angles = compute_angles(line_dirs, degrees=True)
        # Adding a zero angle for first value.
        angles = torch.hstack([zero, angles])

        # Mult choice:
        # We don't want to multiply by 0. Multiplying by angles + 1.
        # values[i] = values[i] * (angles + 1.0)
        values[i] = values[i] * (angles + 1.0)**2

        # Pow choice:
        # loss^0 = 1. loss^1 = loss. Also adding 1.
        # But if values are < 1, pow becomes smaller.
        # Our losses tend toward 0.  Adding 1 before.
        # values[i] = torch.pow(1.0 + values[i], angles + 1.0) - 1.0

    return values


def compute_triu_connectivity(
        streamlines, volume_size, downsampled_volume_size,
        binary: bool = False, to_sparse_tensor: bool = False, device=None):
    """
    Compute a connectivity matrix.

    Parameters
    ----------
    streamlines: list of np arrays or list of tensors.
        Streamlines, in vox space, corner origin.
    volume_size: list
        The 3D dimension of the reference volume.
    downsampled_volume_size:
        Either a 3D size or the size m of the m x m x m downsampled volume
        coordinates for the connectivity matrix. This means that the matrix
        will be a m^d x m^d triangular matrix. In 3D, with 20x20x20, this is an
        8000 x 8000 matrix (triangular = half of it in memory). It probably
        contains a lot of zeros with the background being included. Saved as
        sparse.
    binary: bool
        If true, return a binary matrix.
    to_sparse_tensor:
        If true, return the sparse matrix.
    device:
        If true and to_sparse_tensor, the matrix will be hosted on device.
    """
    # Getting endpoint coordinates
    #  + Fix types
    volume_size = np.asarray(volume_size)
    downsampled_volume_size = np.asarray(downsampled_volume_size)
    if isinstance(streamlines[0], list):
        start_values = [s[0] for s in streamlines]
        end_values = [s[-1] for s in streamlines]
    elif isinstance(streamlines[0], torch.Tensor):
        start_values = [s[0, :].cpu().numpy() for s in streamlines]
        end_values = [s[-1, :].cpu().numpy() for s in streamlines]
    else:  # expecting numpy arrays
        start_values = [s[0, :] for s in streamlines]
        end_values = [s[-1, :] for s in streamlines]

    assert len(downsampled_volume_size) == len(volume_size)
    nb_dims = len(downsampled_volume_size)
    nb_voxels_pre = np.prod(volume_size)
    nb_voxels_post = np.prod(downsampled_volume_size)
    logging.debug("Preparing connectivity matrix of downsampled volume: from "
                  "{} to {}. Gives a matrix of size {} x {} rather than {} "
                  "voxels)."
                  .format(volume_size, downsampled_volume_size,
                          nb_voxels_post, nb_voxels_post, nb_voxels_pre))

    # Downsampling
    mult_factor = downsampled_volume_size / volume_size
    start_values = np.clip((start_values * mult_factor).astype(int),
                           a_min=0, a_max=downsampled_volume_size - 1)
    end_values = np.clip((end_values * mult_factor).astype(int),
                         a_min=0, a_max=downsampled_volume_size - 1)

    # Blocs go from 0 to m1*m2*m3.
    start_block = np.ravel_multi_index(
        [start_values[:, d] for d in range(nb_dims)], downsampled_volume_size)
    end_block = np.ravel_multi_index(
        [end_values[:, d] for d in range(nb_dims)], downsampled_volume_size)

    total_size = np.prod(downsampled_volume_size)
    matrix = np.zeros((total_size, total_size), dtype=int)
    for s_start, s_end in zip(start_block, end_block):
        matrix[s_start, s_end] += 1

        # Either, at the end, sum lower triangular + upper triangular (except
        # diagonal), or:
        if s_end != s_start:
            matrix[s_end, s_start] += 1

    matrix = np.triu(matrix)
    assert matrix.sum() == len(streamlines)

    if binary:
        matrix = matrix.astype(bool)

    if to_sparse_tensor:
        logging.debug("Converting matrix to sparse. Contained {}% of zeros."
                      .format((1 - np.count_nonzero(matrix) / total_size) * 100))
        matrix = torch.as_tensor(matrix, device=device).to_sparse()

    return matrix
