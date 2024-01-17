# analyzeApparentValidation.py

# Copyright 2021 Ottawa Hospital Research Institute
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
# Revision history:
#   Original Python version by Andre Carrington, 2021

global total_folds

import pickle
import numpy as np
import pandas as pd
import transcript
import matplotlib.pyplot as plt
import loess as ml
import scipy.stats as st

resultsv2  = True
testNum    = 32  #38
model      = 'logr'
reanalysis = ''

loadPlotBaseline  = True    # baseline logr no CRP, features converted, apparent validation, no std/center, matches 2012 paper
baselineTestNum   = 27      # baseline logr no CRP, features converted, apparent validation, no std/center, matches 2012 paper
baselineModel     = 'logr'  #
baselineIteration = 0       # logr only has 1 iteration
baselineFold      = 0       # baseline logr apparent validation is only 1 fold
baselineResultsv2 = False

showLoess  = False
showLines  = False
dotcolor   = 'orange'
curvecolor = 'orange'
iterations = 20  # except for the next clause...
if model == 'logr' or model == 'tpfn':  # these methods have only 1 iteration
    iterations    = 1
k_folds    = 1
repetition = 1
total_folds= k_folds * repetition

#wholeMeasureToOptimize = 'AUC'
#wholeMeasureToOptimize = 'rulein_not_ruleout'
wholeMeasureToOptimize = None
groupMeasureToOptimize = 'cpAUCn.2'
#groupMeasureToOptimize = None

def formatList(alist):
    # Create a format spec for each item in the input `alist`.
    # E.g., each item will be right-adjusted, field width=3.
    format_list = ['{:0.4f}' for item in alist]

    listlen = len(format_list)
    linelen = 5  # start a new line every 5 elements
    addme   = 0
    linewrap= '\n                    '
    if listlen > linelen:
        for element in range(linelen, listlen, linelen): # starting at 5 (listlen) add carriage returns
            format_list.insert(element+addme, linewrap)
            addme = addme + 1
        #endfor
    #endif
    # Now join the format specs into a single string:
    # E.g., '{:0.4f}, {:0.4f}, {:0.4f}' if the input list has 3 items.
    s = ', '.join(format_list)

    # Now unpack the input list `alist` into the format string. Done!
    return s.format(*alist)
#enddef

def plotIteration(plotThis, testNum, model):
    from sklearn.metrics import roc_curve
    from deeproc.DeepROC import DeepROC

    fileHandleLS = open(f'output/labelScore_{testNum:03d}{reanalysis}_{model}.pkl', 'rb')
    try:
        for i in range(0, plotThis+1):
            # Overwrite labels and scores, we only want the last one at iteration==plotThis
            #[labels, scores]\
            a = pickle.load(fileHandleLS)
            if len(a) == 4:
                print('warning: 4 lines in input file, used 2.')
                [labels, scores, _, __] = a
            else:
                [labels, scores] = a
            print(' ')
    except:
        print(f'pickle load for {fileHandleLS} failed on iteration {i}')
        exit(1)
    #endtry

    aDeepROC = DeepROC()

    allLabels = []
    folds     = len(labels)
    for i in range(0, folds):
        labels[i] = labels[i].tolist()  # convert (index, value) to just value, Series to list
        scores[i] = scores[i].tolist()  # convert numpy ndarray to list
        allLabels = allLabels + labels[i]
        fpr_temp, tpr_temp, threshold_temp = roc_curve(labels[i], scores[i])
        aDeepROC.set_fold(fpr=fpr_temp, tpr=tpr_temp, threshold=threshold_temp)

    P = sum(allLabels)
    N = len(allLabels) - P
    print(f'Using weighted average sample prevalence: {P/(P+N):0.2f}')
    foldsNPclassRatio  = N/P
    aDeepROC.setFoldsNPclassRatio(foldsNPclassRatio)
    groupAxis = 'TPR'
    # groups = [[0, 0.85], [0.85, 1]]
    #groups = [[0, 0.60], [0.60, 0.90], [0.90, 1]]
    groups = [[0, 0.37], [0.37, 0.90], [0.90, 1]]
    aDeepROC.setGroupsBy(groupAxis=groupAxis, groups=groups, groupByClosestInstance=False)

    costs = dict(cFP=1, cFN=5, cTP=0, cTN=0, costsAreRates=False)  # for ruling-out FN cost is high
    print(f"Assuming Cost_FN:Cost_FP is {costs['cFN']}:{costs['cFP']}")

    print('\nAnalysis by groups of predicted risk:')
    measure_dict = aDeepROC.analyze(forFolds=True)

    print('\nMean ROC Plot.')
    aDeepROC.plot_folds(plotTitle='Mean ROC Plot with folds', showOptimalROCpoints=True, costs=costs,
                        saveFileName=f'output/mean_{testNum}_{model}_i{plotThis}.png', showPlot=True)

    print('\nPlotting groups/ROIs in ROC plot.')
    numgroups = len(groups)
    for i in range(0, numgroups):
        print(f'ROC plot shown for group {i} [{groups[i][0]:0.2f}, {groups[i][1]:0.2f}]')
        fig, ax = aDeepROC.plotGroupForFolds(plotTitle=f'Deep ROC Plot for group {i+1}',
                                   groupIndex=i, foldsNPclassRatio=foldsNPclassRatio,
                                   showError=False, showThresholds=False,
                                   showOptimalROCpoints=True, costs=costs,
                                   saveFileName=f'output/group_{testNum}_{model}_i{plotThis}.png',
                                   numShowThresh=20, showPlot=False, labelThresh=False, full_fpr_tpr=True)
        # annotate plot with measures
        whichMeasures = ['AUCn_i', 'avgSens', 'avgSpec', 'avgPPV', 'avgNPV']
        # whichMeasures = ['AUCn_i', 'avgSens', 'avgSpec']
        top = False
        if   aDeepROC.groupAxis == 'FPR' and measure_dict[i]['avgSens'] < 0.5:
            top = True
        elif aDeepROC.groupAxis == 'TPR' and measure_dict[i]['avgSpec'] < 0.5:
            top = True
        aDeepROC.annotateGroup(i, measure_dict, whichMeasures, top=top)

        plt.show()
    #endfor

#enddef

def analyze(testNum, model, iterations):
    global total_folds

    # example to load results
    logfn = f'output/deepROCanalysis_{testNum:3d}{reanalysis}_{model}.txt'
    transcript.start(logfn)
    if resultsv2:
        print(f'resultsv2_{testNum:03d}_{model}:')
        fileHandle   = open(f'output/resultsv2_{testNum:03d}{reanalysis}_{model}.pkl', 'rb')
    else:
        print(f'results_{testNum:03d}_{model}:')
        fileHandle = open(f'output/results_{testNum:03d}{reanalysis}_{model}.pkl', 'rb')
    #endif
    fileHandleS  = open(f'output/settings_{testNum:03d}.pkl', 'rb')
    try:
        if resultsv2:
            measure_to_optimize, type_to_optimize, deepROC_groups, \
                groupAxis, wholeMeasures, groupMeasures, pointMeasures,\
                NRIMeasures, NRIMeasures2, ROC_points, NRI_points = pickle.load(fileHandleS)
            fileHandleS.close()
            wholeMatrix, groupMatrix, pointMatrix, NRIMatrix, NRIMatrix2, thrMatrix = \
                pickle.load(fileHandle)
            # RI_thr_low, RO_thr_low, RI_thr_high, RO_thr_high = thrMatrix[fold, :, iteration]
            fileHandle.close()
        else:
            measure_to_optimize, type_to_optimize, deepROC_groups, \
                groupAxis, wholeMeasures, groupMeasures = pickle.load(fileHandleS)
            fileHandleS.close()
            wholeMatrix, groupMatrix = pickle.load(fileHandle)
            fileHandle.close()
    except:
        print('pickle load failed')
        exit(1)
    #endtry

    is_not_nan = lambda a: a[np.invert(np.isnan(a))]

    myround = lambda a: round(10000.0*np.max(a))/100.0  # as percentage, rounded to 2nd decimal place
    if wholeMeasureToOptimize is not None:
        vector  = wholeMatrix[0,wholeMeasures.index(wholeMeasureToOptimize),:]
        j       = np.argmax(vector)
        print(f'The highest value of {wholeMeasureToOptimize} occurs at index {j}')
    elif groupMeasureToOptimize is not None:
        name_to_optimize, num_string = groupMeasureToOptimize.split('.')
        i = 0
        for m in groupMeasures:
            if name_to_optimize == m:
                opt_gmeas = i
                opt_group = int(num_string)
                break
            # endif
            i = i + 1
        # endfor
        vector  = groupMatrix[0,opt_group,opt_gmeas,:]
        j       = np.argmax(vector)
        print(f'The highest value of {groupMeasureToOptimize} occurs at index {j}')
    else:
        ValueError('wholeMeasureToOptimize or groupMeasureToOptimize are not set.')
    #endif

    # load baseline scores and labels for NRI measures
    labelScoreFileBaseline = open(f'output/labelScore_{baselineTestNum:03d}_{baselineModel}.pkl', 'rb')
    for i in range(0, baselineIteration + 1):
        iteration_a = pickle.load(labelScoreFileBaseline)
    # endfor
    labelScoreFileBaseline.close()
    labelsArray = iteration_a[0]
    scoresArray = iteration_a[1]
    baselineLabels = labelsArray[baselineFold]
    baselineScores = scoresArray[baselineFold]

    plotIteration(j, testNum, model)
    if len(deepROC_groups) == 2:
        resultsj = [wholeMatrix[0,:,j], groupMatrix[0,0,:,j], groupMatrix[0,1, :, j],
                    groupMatrix[0,2,:,j], pointMatrix[0,0,:,j], pointMatrix[0,1,:,j],
                    NRIMatrix[0,:,j], NRIMatrix2[0,:,j]]
        allnames = [wholeMeasures, groupMeasures, groupMeasures, groupMeasures,
                    pointMeasures, pointMeasures, NRIMeasures, NRIMeasures2]
    elif len(deepROC_groups) == 3:
        resultsj = [wholeMatrix[0,:,j],   groupMatrix[0,0,:,j], groupMatrix[0,1, :, j],
                    groupMatrix[0,2,:,j], groupMatrix[0,3,:,j],
                    pointMatrix[0,0,:,j], pointMatrix[0,1,:,j],
                    NRIMatrix[0,:,j], NRIMatrix2[0,:,j]]
        allnames = [wholeMeasures, groupMeasures, groupMeasures, groupMeasures, groupMeasures,
                    pointMeasures, pointMeasures, NRIMeasures, NRIMeasures2 ]
    #endif

    labels   = ['wholeMeasures', 'groupMeasures.0 (whole)', 'groupMeasures.1', 'groupMeasures.2',
                'pointMeasures.1', 'pointMeasures.2', 'NRIMeasures', 'NRIMeasures2' ]
    for vec, names, label in zip(resultsj, allnames, labels):
        print(f'\n{label}: ', end='')
        if   label == 'pointMeasures.1':
            print(f'at {ROC_points[0][0]}=={ROC_points[0][1]} SW (most southwest point)')
        elif label == 'pointMeasures.2':
            print(f'at {ROC_points[1][0]}=={ROC_points[1][1]} NE (most northeast point)')
        elif label == 'NRIMeasures' or label == 'NRIMeasures2':
            print(f'at {NRI_points[0][0]}=={NRI_points[0][1]} SW and ', end='')
            print(f'{NRI_points[1][0]}=={NRI_points[1][1]} NE')
        else:
            print('')
        #endif
        mod_ctr = 0
        for meas, name in zip(vec, names):
            if name=='avgLRp' or name=='avgLRn' or name=='avgOR':
                meas=np.nan
            int_measures = ['TP','TN','FP','FN','PP', 'PN']
            try:
                int_measures.index(name)
                t = f'{name:15s}: {int(meas):d}'
            except:
                if meas < 0:
                    t = f'{name:15s}:{meas:0.3f}'
                else:
                    if meas < 10:
                        t = f'{name:15s}: {meas:0.3f}'
                    elif meas >= 10 and meas < 100:
                        t = f'{name:15s}: {meas:0.2f}'
                    elif meas >= 100:
                        t = f'{name:15s}: {meas:0.1f}'
                    elif np.isnan(meas):
                        t = f'{name:15s}: nan'
                    #endif
                #endif
            #endtry
            if mod_ctr == 2:
                print(f'{t:25s}')
                mod_ctr = 0
            else:
                print(f'{t:25s}    ', end='')
                mod_ctr = mod_ctr + 1
            #endif
        #endfor
        print('')
    #endfor
    # print('')
    # print('\nwholeMatrix')
    # print(formatList(list(wholeMatrix[0,:,j])))
    # print('\ngroupMatrix.0')
    # print(formatList(list(groupMatrix[0,0,:,j])))
    # print('\ngroupMatrix.1')
    # print(formatList(list(groupMatrix[0,1,:,j])))
    # print('\ngroupMatrix.2')
    # print(formatList(list(groupMatrix[0,2,:,j])))
    # # print('\ngroupMatrix.3')
    # # print(formatList(list(groupMatrix[0,3,:,j])))
    # print('\npointMatrix.0')
    # print(formatList(list(pointMatrix[0,0,:,j])))
    # print('\npointMatrix.1')
    # print(formatList(list(pointMatrix[0,1,:,j])))
    # print('\nNRIMatrix')
    # print(formatList(list(NRIMatrix[0,:,j])))

    print(' ')
    transcript.stop()
#enddef

analyze(testNum, model, iterations)
