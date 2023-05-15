# -*- coding: utf-8 -*-
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


def compress_streamline_values(streamlines: List, values: List = None,
                               compress_eps: float = 1e-3):
    """
    Parameters
    ----------
    streamlines: List[Tensors]
        Streamlines' directions.
    values: List[Tensors]
        If set, compresses the values rather than the streamlines themselves.
    compress_eps: float
        Angle (in degrees)
    """
    if values is None:
        # Compress the streamline itself with our technique.
        # toDo
        raise NotImplementedError("Code not ready")

    compress_eps = np.deg2rad(compress_eps)
    one = torch.ones(1, device=streamlines[0].device)

    dirs = compute_directions(streamlines)

    compressed_mean_loss = 0.0
    compressed_n = 0
    for loss, line_dirs in zip(values, dirs):
        if len(loss) < 2:
            compressed_mean_loss = compressed_mean_loss + torch.mean(loss)
            compressed_n += len(loss)
        else:
            # 1. Compute angles
            # Skip normalization if not required:
            line_dirs /= torch.linalg.norm(line_dirs, dim=-1, keepdim=True)
            cos_angles = torch.sum(line_dirs[:-1, :] * line_dirs[1:, :], dim=1)

            # Resolve numerical instability
            cos_angles = torch.minimum(torch.maximum(-one, cos_angles), one)
            angles = torch.arccos(cos_angles)

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
