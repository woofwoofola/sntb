#!/usr/bin/env python
# coding: utf-8

# test2.py

def test2(first, second):
    from checkROCdominance import checkROCdominance
    from TestVectors     import getTestVector
    from sklearn.metrics import roc_curve

    scores1, labels1, groups, groupAxis, description = getTestVector(first)
    scores2, labels2, groups, groupAxis, description = getTestVector(second)
    fpr1, tpr1, thr1 = roc_curve(labels1, scores1)
    fpr2, tpr2, thr2 = roc_curve(labels2, scores2)

    checkROCdominance(fpr1, tpr1, fpr2, tpr2)
#enddef

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-first")
    parser.add_argument("-second")
    args = parser.parse_args()
    test2(first=int(args.first), second=int(args.second))
