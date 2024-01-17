#!/usr/bin/env python
# coding: utf-8
#
# nri2.py
# Computes net reclassification index for a single threshold
# which is non-standard, since NRI is defined for two thresholds,
# however, the single threshold results are easily combined to make
# the standard two-threshold NRI.
#
# Copyright 2024 André M Carrington, Ottawa Hospital Research Institute
#
# Use is subject to the GNU Public License 3.0
# The original versions of these functions were written by Lambert T Leong.
# The functions have been modified substantively by André M Carrington
#
# The original functions were:
#   track_movement
#   category_free_nri
#   found in: https://github.com/LambertLeong/AUC_NRI_IDI_python_functions/blob/main/more_metrics/more_metrics.py
#   See GNU Public License 3.0
#
# They have been modified by André Carrington to:
#   track_movement (modified to use a single threshold)
#   nri_single_threshold
#   See GNU Public License 3.0

import numpy as np
from typing import List, Tuple

def track_movement(ref: np.ndarray, new: np.ndarray, indices: List[int], thresholdb: float, threshold: float) -> Tuple[int, int]:
    """
    Track the movement (upward and downward) between two sets of predictions.

    Args:
        ref (np.ndarray):    Reference predictions.
        new (np.ndarray):    New predictions.
        indices (List[int]): List of data indices.
        threshold (float):   Threshold (score or probability) used to track movement

    Returns:
        Tuple[int, int]: Count of upward movements, count of downward movements.
    """
    up, down = 0, 0

    for i in indices:
        ref_val, new_val = ref[i], new[i]
        if   ref_val <  thresholdb and new_val >= threshold:
            up += 1
        elif ref_val >= thresholdb and new_val <  threshold:
            down += 1

    return up, down


def nri_single_threshold(y_truth: np.ndarray, y_ref: np.ndarray, y_new: np.ndarray, thresholdb: float, threshold: float) -> Tuple[float, float, float]:
    """
    Calculate Net Reclassification Improvement (NRI) for a single threshold, as NRI for events, NRI for nonevents and NRI.

    Args:
        y_truth (np.ndarray): Ground truth labels.
        y_ref (np.ndarray):   Reference predictions.
        y_new (np.ndarray):   New predictions.
        threshold (float):    A single threshold (score or probability) for which NRI is computed

    Returns:
        Tuple[float, float, float]: NRI for events, NRI for nonevents, and total NRI.
    """
    if not np.issubdtype(y_truth.dtype, np.number):
        raise TypeError("All elements of y_truth must be numerical")

    event_index                  = np.where(y_truth == 1)[0]
    nonevent_index               = np.where(y_truth == 0)[0]
    events_up, events_down       = track_movement(y_ref, y_new, event_index,    thresholdb, threshold)
    nonevents_up, nonevents_down = track_movement(y_ref, y_new, nonevent_index, thresholdb, threshold)
    nri_events                   = (events_up      / len(event_index))    - (events_down  / len(event_index))
    nri_nonevents                = (nonevents_down / len(nonevent_index)) - (nonevents_up / len(nonevent_index))

    return nri_events, nri_nonevents, nri_events + nri_nonevents