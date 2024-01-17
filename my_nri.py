#!/usr/bin/env python
# coding: utf-8
#
# my_nri.py
# Computes standard and non-standard measures for
# net reclassification index (NRI) based on two thresholds, as it was defined
# which I refer to as rule-in and rule-out thresholds
#
# Copyright 2024 André M Carrington, Ottawa Hospital Research Institute
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Andre: This code is based on my own approach/way of computing NRI, for which there is
# a proof of equivalence.  It uses confusion matrices for the reference and new
# models.  It does not need to track each instance individually.  It also, therefore
# does not have to have the labels and scores in the same order, as an advantage.
# Furthermore, it has other, potential uses as a generalization of NRI.

import numpy as np

def nri_measures(labels_ref, scores_ref, labels_new, scores_new, RI_thresholdb, RO_thresholdb, RI_threshold, RO_threshold):
    from sklearn.metrics import confusion_matrix
    if RI_threshold is None and RO_threshold is None:
        SystemError('nri: either RI_threshold, RO_threshold or both must be specified')
    if (len(labels_ref) != len(labels_new)) or (len(scores_ref) != len(scores_new)) or \
       (len(labels_ref) != len(scores_ref)):
            SystemError('nri: lengths of labels or scores do not match')
    cm_ref             = np.zeros((2,2,2))
    cm_new             = np.zeros((2,2,2))
    NRI_p              = 0
    NRI_n              = 0
    RO_RIp, RO_RIn     = None, None
    RI_RIp, RI_RIn     = None, None
    RI_NRI, RO_NRI     = None, None
    for i in range(0,2):
        if i==1:
            thresholdb = RO_thresholdb
            threshold  = RO_threshold
        else:
            thresholdb = RI_thresholdb
            threshold  = RI_threshold
        if threshold is None:
            continue
        y_ref          = scores_ref >= thresholdb  # scores greater than or equal to threshold 1, else 0
        y_new          = scores_new >= threshold  # scores greater than or equal to threshold 1, else 0

        cm_ref[i]      = confusion_matrix(labels_ref, y_ref) # rows are true, columns are predicted, index is row,col
        J,K,Q,R        = cm_ref[i].ravel()  # row 0 J,K, row 1 Q,R
        Up             = R-Q
        Un             = J-K

        cm_new[i]      = confusion_matrix(labels_new, y_new) # rows are true, columns are predicted, index is row,col
        L,M,S,T        = cm_new[i].ravel()  # row 0 L,M,  row 1 S,T
        Vp             = T-S
        Vn             = L-M

        Np             = T+S
        Nn             = L+M

        if i==1:
            RO_RIp = (Vp-Up)/(2*Np)
            RO_RIn = (Vn-Un)/(2*Nn)
            RO_NRI = RO_RIp + RO_RIn
            NRI_p  = NRI_p + RO_RIp
            NRI_n  = NRI_n + RO_RIn
            if RI_threshold is None:
                NRI = RO_NRI
        else:
            RI_RIp = (Vp-Up)/(2*Np)
            RI_RIn = (Vn-Un)/(2*Nn)
            RI_NRI = RI_RIp + RI_RIn
            NRI_p  = NRI_p + RI_RIp
            NRI_n  = NRI_n + RI_RIn
            if RO_threshold is None:
                NRI = RI_NRI
        #endif
        if RI_threshold is not None and RO_threshold is not None:
            NRI = NRI_p + NRI_n
    #endif
    return NRI_p, NRI_n, NRI, RI_NRI, RI_RIp, RI_RIn, RO_NRI, RO_RIp, RO_RIn
#enddef