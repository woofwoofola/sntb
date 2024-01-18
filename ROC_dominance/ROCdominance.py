#!/usr/bin/env python
# coding: utf-8

# ROCdominance.py

def getSuperset(fpr1_list, fpr2_list):
    fpr_list = fpr1_list.copy()
    fpr_list.extend(fpr2_list)
    return fpr_list
#enddef

def getUnique(inList):
    outList = []
    for item in inList:
        if item not in outList:
            outList.append(item)
    return outList
#enddef

def getInterpValue(fpr, fpr1_list, tpr1_list):
    last_fpr1 = -1  # first item
    last_tpr1 = -1  # first item
    for fpr1, tpr1 in zip(fpr1_list, tpr1_list):
        if last_fpr1 == -1:  # first item
            last_fpr1 = fpr1
            last_tpr1 = tpr1
            continue
        if fpr1 > fpr:
            if last_fpr1 == fpr:
                raise ValueError(
                    'getInterpValues is not intended for matching fpr values which may be non-unique in some cases.')
            fpr_proportion = (fpr - last_fpr1) / (fpr1 - last_fpr1)
            tpr_interpolated = last_tpr1 + fpr_proportion * (tpr1 - last_tpr1)
            return tpr_interpolated
        last_fpr1 = fpr1
        last_tpr1 = tpr1
    # endfor
    raise ValueError('No interpolated value found')
#enddef

def myCheckDominance(val1, val2):
    diff = val1 - val2
    if diff > 0:  # if diff positive and greater than epsilon, return 1 for the dominance vector
        return 1
    elif diff < 0:  # if diff negative and less    than -epsilon, return 2 for the dominance vector
        return 2
    else:  # else return 0 for the dominance vector
        return 0
    #endif
#enddef

def ROCdominance(fpr1_listlike, tpr1_listlike, fpr2_listlike, tpr2_listlike):
    from stripInternalROCpoints import stripInternalROCpoints
    from scipy.interpolate      import interp1d

    fpr1_list, tpr1_list = stripInternalROCpoints(fpr1_listlike, tpr1_listlike)
    print(f'fpr1: {fpr1_list}')
    print(f'tpr1: {tpr1_list}')
    fpr2_list, tpr2_list = stripInternalROCpoints(fpr2_listlike, tpr2_listlike)
    print(f'fpr2: {fpr2_list}')
    print(f'tpr2: {tpr2_list}')

    # if we traverse the ROC curve by classification score (predicted risk or threshold),
    # there are ties in threshold that would need to be resolved (but there is no clear way how to do that)
    #
    # if we traverse the ROC curve by FPR, there are ties in FPR which can be resolved by TPR
    # but we cannot use interpolation in a range of FPR unless FPR are unique (which they are not)
    #
    # it is best to remove internal points in an ROC curve to reduce ties in FPR to only the bottom and
    # top of a vertical line segment.
    #
    # to assess dominance, we will do so at each FPR value in both ROC curves
    # so we get the superset of all x (FPR) values (after omitting internal values)
    all_fpr_list = getSuperset(fpr1_list, fpr2_list)
    all_fpr_list = getUnique(all_fpr_list)
    all_fpr_list.sort()

    # if we want to assess dominance as a binary value at a point, then it does not matter whether
    # we assess that from an ROC point to a corresponding vertical point, corresponding horizontal
    # point or corresponding point along any slope.
    #
    # if we want to assess dominance as a continuous value, then we need all the points at which the curves
    # cross or meet and we can compute the areas between the curves.  But this is a separate matter.

    # for each x (FPR) value in the superset of x values
    lastPoint       = len(all_fpr_list)
    print(f'all_fpr_list: {all_fpr_list}')
    dominanceVector = []
    dominanceFPR    = []
    for index, fpr in zip(range(1, len(all_fpr_list)+1), all_fpr_list):

        # how many matching x (FPR) values exist in ROC1?
        match1 = fpr1_list.count(fpr)
        if   match1 == 0:
            # if 0 - horizontal or diagonal - interpolate the y (TPR) value
            tpr1a   = getInterpValue(fpr, fpr1_list, tpr1_list)
        elif match1 == 1:
            # if 1 - get the y (TPR) value
            index1a = fpr1_list.index(fpr)
            tpr1a   = tpr1_list[index1a]
        elif match1 == 2:
            # if 2 - get both y (TPR) values
            index1a = fpr1_list.index(fpr)
            index1b = fpr1_list.index(fpr, index1a+1)
            tpr1a   = tpr1_list[index1a]
            tpr1b   = tpr1_list[index1b]
        else:
            raise ValueError('Unexpected number of matches (more than 2) after stripping internal points.')
        #endif

        # how many matchine x (FPR) values exist in ROC2?
        match2 = fpr2_list.count(fpr)
        if   match2 == 0:
            # if 0 - horizontal or diagonal - interpolate the y (TPR) value
            tpr2a   = getInterpValue(fpr, fpr2_list, tpr2_list)
        elif match2 == 1:
            # if 1 - get the y (TPR) value
            index2a = fpr2_list.index(fpr)
            tpr2a   = tpr2_list[index2a]
        elif match2 == 2:
            # if 2 - get both y (TPR) values
            index2a = fpr2_list.index(fpr)
            index2b = fpr2_list.index(fpr, index2a+1)
            tpr2a   = tpr2_list[index2a]
            tpr2b   = tpr2_list[index2b]
        else:
            raise ValueError('Unexpected number of matches (more than 2) after stripping internal points.')
        #endif

        # treat zero matches (interpolated) like single matches
        if  match1 == 0:
            match1 = 1
        if  match2 == 0:
            match2 = 1
        # if 2 for both:
        if   match1 == 2 and match2 == 2:
            # first, compare the lower values
            dominanceVector.extend([myCheckDominance(tpr1a, tpr2a)])
            dominanceFPR.extend([fpr])
            # second, compare the higher values
            dominanceVector.extend([myCheckDominance(tpr1b, tpr2b)])
            dominanceFPR.extend([fpr])

        elif match1 == 2 and match2 == 1:
            # first, compare the lower value to the single value
            dominanceVector.extend([myCheckDominance(tpr1a, tpr2a)])
            dominanceFPR.extend([fpr])
            # second, compare the higher value to the single value
            dominanceVector.extend([myCheckDominance(tpr1b, tpr2a)])
            dominanceFPR.extend([fpr])

        elif match1 == 1 and match2 == 2:
            # first, compare the lower value to the single value
            dominanceVector.extend([myCheckDominance(tpr1a, tpr2a)])
            dominanceFPR.extend([fpr])
            # second, compare the higher value to the single value
            dominanceVector.extend([myCheckDominance(tpr1a, tpr2b)])
            dominanceFPR.extend([fpr])

        else:
            # compare the two single values
            dominanceVector.extend([myCheckDominance(tpr1a, tpr2a)])
            dominanceFPR.extend([fpr])
        #endif
    #enddef
    return dominanceVector, dominanceFPR
#enddef
