#!/usr/bin/env python
# coding: utf-8

# test1.py

def test1(first, second):
    from ROCdominance    import ROCdominance
    from TestVectors     import getTestVector
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    import numpy as np

    scores1, labels1, groups, groupAxis, description = getTestVector(first)
    scores2, labels2, groups, groupAxis, description = getTestVector(second)
    fpr1, tpr1, thr1 = roc_curve(labels1, scores1)
    fpr2, tpr2, thr2 = roc_curve(labels2, scores2)
    plt.figure()
    plt.plot(fpr1, tpr1, linestyle=':', lw=2, color='black')
    plt.scatter(fpr1, tpr1, s=40, marker='o', lw=2, facecolors='black', edgecolors='black')
    plt.plot(fpr2, tpr2, linestyle='-', lw=2, color='blue')
    plt.scatter(fpr2, tpr2, s=40, marker='o', lw=2, facecolors='blue', edgecolors='blue')
    plt.show()
    dv, fpr = ROCdominance(fpr1, tpr1, fpr2, tpr2)
    print(f'Dominance vector for {first:d} vs {second:d}: {dv}')

    formatted_fpr = '['
    for i in fpr:
        formatted_fpr = formatted_fpr + f'{i:0.2f}, '
    formatted_fpr = formatted_fpr[0:-2]  # remove last 2 characters
    formatted_fpr = formatted_fpr + ']'

    print(f'FPR       vector for {first:d} vs {second:d}: {formatted_fpr}')
    # to evaluate strict domination, remove the first and last points which tie for all ROC curves
    strictdv = dv.copy()
    strictdv = strictdv[1:len(dv)-1]
    dom  = np.unique(strictdv)
    dom0 = (dom == 0).any()
    dom1 = (dom == 1).any()
    dom2 = (dom == 2).any()
    if dom1 and not dom2:
        if dom0:
            print(f'The first ROC curve (black dotted line), #{first:d} dominates.')
        else:
            print(f'The first ROC curve (black dotted line), #{first:d} strictly dominates.')
    if dom2 and not dom1:
        if dom0:
            print(f'The second ROC curve (blue solid line), #{second:d} dominates.')
        else:
            print(f'The second ROC curve (blue solid line), #{second:d} strictly dominates.')
    if dom1 and dom2:
        print(f'Neither ROC curve dominates.')
#enddef

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-first")
    parser.add_argument("-second")
    args = parser.parse_args()
    test1(first=int(args.first), second=int(args.second))
