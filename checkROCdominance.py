#!/usr/bin/env python
# coding: utf-8
# checkROCdominance.py

def checkROCdominance(fpr1, tpr1, fpr2, tpr2):
    from ROCdominance import ROCdominance
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    plt.plot(fpr1, tpr1, linestyle=':', lw=2, color='black')
    plt.scatter(fpr1, tpr1, s=40, marker='o', lw=2, facecolors='black', edgecolors='black')
    plt.plot(fpr2, tpr2, linestyle='-', lw=2, color='blue')
    plt.scatter(fpr2, tpr2, s=40, marker='o', lw=2, facecolors='blue', edgecolors='blue')
    plt.show()
    dv, fpr = ROCdominance(fpr1, tpr1, fpr2, tpr2)
    print(f'Dominance vector: {dv}')

    formatted_fpr = '['
    for i in fpr:
        formatted_fpr = formatted_fpr + f'{i:0.2f}, '
    formatted_fpr = formatted_fpr[0:-2]  # remove last 2 characters
    formatted_fpr = formatted_fpr + ']'

    print(f'FPR       vector: {formatted_fpr}')
    # to evaluate strict domination, remove the first and last points which tie for all ROC curves
    strictdv = dv.copy()
    strictdv = strictdv[1:len(dv)-1]
    dom  = np.unique(strictdv)
    dom0 = (dom == 0).any()
    dom1 = (dom == 1).any()
    dom2 = (dom == 2).any()
    if dom1 and not dom2:
        if dom0:
            print(f'The first ROC curve (black dotted line) dominates.')
        else:
            print(f'The first ROC curve (black dotted line) strictly dominates.')
    if dom2 and not dom1:
        if dom0:
            print(f'The second ROC curve (blue solid line) dominates.')
        else:
            print(f'The second ROC curve (blue solid line) strictly dominates.')
    if dom1 and dom2:
        print(f'Neither ROC curve dominates.')
#enddef