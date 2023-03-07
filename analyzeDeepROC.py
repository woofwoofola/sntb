# analyzeDeepROC.py

#
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

testNum    = 200
#model      = 'case'
model      = 'svcLin'
reanalysis = ''
showLoess  = False
showLines  = False
dotcolor   = 'orange'
curvecolor = 'orange'
iterations = 100
k_folds    = 10
repetition = 10
total_folds= k_folds * repetition

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

def simple_plot_group_results(yvals1, yvals2, j):
    # y_AUC = np.transpose(groupMatrix[:, 1:, cpAUCn_i, j])  # transpose folds x groups (y by x)
    # y_avgPPV = np.transpose(groupMatrix[:, 1:, avgPPV_i, j])
    groups = range(1, num_groups)
    # fig      = plt.figure()
    # ax       = fig.add_subplot(6, 2, 1)
    plt.subplot(121)
    plt.plot(groups, y_AUC)
    plt.subplot(122)
    plt.plot(groups, y_avgPPV)
    plt.show()
# enddef

def nice_plot_group_results(j, x, y, y_name, w, w_name, model_name, num_groups,
                            measure_to_optimize, fileNum, ybottom, ytop, dotcolor, curvecolor):
    df = pd.DataFrame({"x": x, "y": y})
    if showLoess:
        fit_df = ml.loess("x", "y", data=df, alpha=0.7, poly_degree=2)
    #endif
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    xlen = int(df.shape[0])
    xjitter = np.random.uniform(-0.04, +0.04, (xlen, ))

    if showLines:
        zz = len(df["x"])
        za = df["x"] + pd.Series(xjitter)
        zb = df["y"]
        for i in range(0, num_groups-1):
            theyStart = i*(num_groups-1)
            theyEnd   = theyStart + (num_groups-1)
            they = list(range(theyStart, theyEnd))
            ax1.plot(za[they], zb[they], color=dotcolor, marker="o", label="_nolegend_")
        #ax1.plot(za, zb, color="blue", marker="o", label="_nolegend_")
    else:
        ax1.scatter(df["x"]+pd.Series(xjitter), df["y"], color=dotcolor, marker="o", s=7, label="_nolegend_")
    #endif
    if showLoess:
        ax1.plot(fit_df['x'], fit_df['y'], color=curvecolor, linewidth=3, label=f'{y_name} loess fit')
    #endif
    if w_name == '':
        pass
    else:
        mu = w[j, 0]
        # se = w[j, 1]
        # ax1.fill([1, num_groups - 1, num_groups - 1, 1, 1],
        #          [mu + se, mu + se, mu - se, mu - se, mu + se],
        #          color='grey', alpha=0.3, linewidth=0)
        ax1.plot(list(range(1, num_groups)), [mu] * (num_groups - 1),
                 color=curvecolor, alpha=0.3, linestyle='--', linewidth=2, label=w_name)
    #endif
    plt.title(f'{y_name} results from {model_name} optimized by {measure_to_optimize}')
    plt.xlabel('Predicted Probability/Risk Groups')
    plt.ylabel(f'{y_name}')
    #floor5percent = lambda x: np.floor((x*100)/5)*5/100
    #plt.ylim(floor5percent(float(min(y))), floor5percent(float(max(y)))+0.05)
    plt.ylim(ybottom, ytop)
    #plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(f'output/groups_{fileNum:03d}_{model_name}_{y_name}_{j}.png')
# enddef

def setupPlotData(groupMatrix, measure_index, num_groups, iteration, total_folds):
    x = []
    y = []
    for i in range(0, total_folds):
        x = x + list(range(1, num_groups))
        y = y + list(groupMatrix[i, 1:, measure_index, iteration])
    # endfor
    return x, y
# enddef

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
    groups = [[0, 0.30], [0.30, 0.92], [0.92, 1]]
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
    if model == 'case':  # for test vectors
        return
    else:
        logfn = f'output/deepROCanalysis_{testNum:3d}{reanalysis}_{model}.txt'
        transcript.start(logfn)
        print(f'results_{testNum:03d}_{model}:')
        fileHandle   = open(f'output/results_{testNum:03d}{reanalysis}_{model}.pkl', 'rb')
        fileHandleS  = open(f'output/settings_{testNum:03d}.pkl', 'rb')
    #endif
    try:
        measure_to_optimize, type_to_optimize, deepROC_groups, \
        groupAxis, areaMeasures, groupMeasures = pickle.load(fileHandleS)
        fileHandleS.close()
        areaMatrix, groupMatrix                = pickle.load(fileHandle)
        fileHandle.close()
    except:
        print('pickle load failed')
        exit(1)
    #endtry

    # create indices from above
    AUC_ix      = areaMeasures.index('AUC')
    AUC_full_ix = areaMeasures.index('AUC_full')
    AUC_plain_ix= areaMeasures.index('AUC_plain')
    AUC_micro_ix= areaMeasures.index('AUC_micro')
    AUPRC_ix    = areaMeasures.index('AUPRC')

    cDelta_ix   = groupMeasures.index('cDelta')
    cpAUC_ix    = groupMeasures.index('cpAUC')
    pAUC_ix     = groupMeasures.index('pAUC')
    pAUCx_ix    = groupMeasures.index('pAUCx')

    cDeltan_ix  = groupMeasures.index('cDeltan')
    cpAUCn_ix   = groupMeasures.index('cpAUCn')
    pAUCn_ix    = groupMeasures.index('pAUCn')
    pAUCxn_ix   = groupMeasures.index('pAUCxn')

    groupMeasures = ['cDelta', 'cpAUC', 'pAUC', 'pAUCx',
                     'cDeltan', 'cpAUCn', 'pAUCn', 'pAUCxn',
                     'avgA', 'bAvgA', 'avgSens', 'avgSpec',
                     'avgPPV', 'avgNPV',
                     'avgLRp', 'avgLRn',
                     'ubAvgA', 'avgBA', 'sPA']

    avgA_ix     = groupMeasures.index('avgA')
    bAvgA_ix    = groupMeasures.index('bAvgA')
    avgSens_ix  = groupMeasures.index('avgSens')
    avgSpec_ix  = groupMeasures.index('avgSpec')

    avgPPV_ix   = groupMeasures.index('avgPPV')
    avgNPV_ix   = groupMeasures.index('avgNPV')
    avgLRp_ix   = groupMeasures.index('avgLRp')
    avgLRn_ix   = groupMeasures.index('avgLRn')

    ubAvgA_ix   = groupMeasures.index('ubAvgA')
    avgBA_ix    = groupMeasures.index('avgBA')
    sPA_ix      = groupMeasures.index('sPA')

    num_groups         = len(deepROC_groups) + 1
    num_group_measures = len(groupMeasures)
    num_area_measures  = len(areaMeasures)

    #mean_measure = [None] * total_folds
    #areaMatrix   = np.zeros(shape=[total_folds, num_area_measures, iterations])
    #                               5            x5                 x100
    #groupMatrix  = np.zeros(shape=[total_folds, num_groups, num_group_measures, iterations])
    #                               5            x6          x15                 x100 = 270k * 4B = 1.08 MB

    is_not_nan = lambda a: a[np.invert(np.isnan(a))]

    # because of nan's computing means has to be done element-wise, not with matrix operations
    # initialize variables
    computed_mean_CI_AUC     = np.zeros([iterations, 2])
    computed_mean_CI_AUPRC   = np.zeros([iterations, 2])
    computed_mean_CI_avgPPV  = np.zeros([iterations, 2])
    computed_mean_CI_avgNPV  = np.zeros([iterations, 2])
    computed_mean_CI_cpAUCni = np.zeros([iterations, 2])
    computed_mean_CI_pAUCn1  = np.zeros([iterations, 2])
    computed_mean_CI_cpAUCn2 = np.zeros([iterations, 2])
    computed_mean_CI_pAUCn2  = np.zeros([iterations, 2])

    # code from https://stackoverflow.com/questions/7568627/using-python-string-formatting-with-lists
    for i in range(0, iterations):
        print(f'{i:03d}: mean_AUC_full:   {np.mean(is_not_nan(areaMatrix[:, AUC_full_ix, i])):0.4f}')
        vector = is_not_nan(areaMatrix[:, AUC_ix, i])
        computed_mean_CI_AUC[i, 0] = np.mean(vector)
        if total_folds >= 30:
            # z = 1.96
            lb, ub = st.norm.interval(alpha=0.95, loc=np.mean(vector), scale=st.sem(vector))
        else:
            # for total_folds==10, use z = 2.262
            lb, ub = st.t.interval(alpha=0.95, df=len(vector)-1, loc=np.mean(vector), scale=st.sem(vector))
        #endif
        computed_mean_CI_AUC[i, 1] = (ub-lb)/2
        # computed_mean_CI_AUC[i, 1] = 2.262 * np.std(vector, ddof=1) / np.sqrt(len(vector))
        print(f'     mean_AUC:        {computed_mean_CI_AUC[i, 0]:0.4f}')
        print(f'     AUC:             {formatList(list(areaMatrix[:, AUC_ix, i]))}')
        print(f'     cDeltan.0:       {formatList(list(groupMatrix[:, 0, cDeltan_ix, i]))}')
        print(f'     bAvgA.0:         {formatList(list(groupMatrix[:, 0, bAvgA_ix, i]))}')
        print(f'     cpAUC.0:         {formatList(list(groupMatrix[:, 0, cpAUC_ix, i]))}')
        temp = []
        for f in range(0, total_folds):
            # compute sum of cpAUC.i, across groups, for i=1:
            temp = temp + [np.sum(is_not_nan(groupMatrix[f, 1:, cpAUC_ix, i]))]
        # show cp.AUC.sum across folds
        print(f'     cpAUC.sum:       {formatList(list(temp))}')

        print(' ')
        vector = is_not_nan(areaMatrix[:, AUPRC_ix, i])
        computed_mean_CI_AUPRC[i, 0] = np.mean(vector)
        computed_mean_CI_AUPRC[i, 1] = 2.262 * np.std(vector, ddof=1) / np.sqrt(len(vector))
        print(f'     mean_AUPRC:      {computed_mean_CI_AUPRC[i, 0]:0.4f}')
        vector = is_not_nan(groupMatrix[:, 0, avgPPV_ix, i])
        computed_mean_CI_avgPPV[i, 0] = np.mean(vector)
        computed_mean_CI_avgPPV[i, 1] = 2.262 * np.std(vector, ddof=1) / np.sqrt(len(vector))
        print(f'     mean_avgPPV.0:   {computed_mean_CI_avgPPV[i, 0]:0.4f}')
        print(f'     AUPRC:           {formatList(list(areaMatrix[:, AUPRC_ix, i]))}')
        print(f'     avgPPV.0:        {formatList(list(groupMatrix[:, 0, avgPPV_ix, i]))}')
        print(' ')
        vector = is_not_nan(groupMatrix[:, 0, avgNPV_ix, i])
        computed_mean_CI_avgNPV[i, 0] = np.mean(vector)
        computed_mean_CI_avgNPV[i, 1] = 2.262 * np.std(vector, ddof=1) / np.sqrt(len(vector))
        print(f'     mean_avgNPV.0:   {computed_mean_CI_avgNPV[i, 0]:0.4f}')
        print(f'     avgNPV.0:        {formatList(list(groupMatrix[:, 0, avgNPV_ix, i]))}')
        print(' ')

        # figure out the range for partial groups
        if deepROC_groups[0][0]==0 and deepROC_groups[0][1]==1:
            # if first group is the whole [0,q], then we will exclude it
            firstPartialGroup = 1
        else:
            firstPartialGroup = 0
        #endif
        for g in range(firstPartialGroup, num_groups):
            ROIindex = 1
            if g == ROIindex:
                vector = is_not_nan(groupMatrix[:, g, cpAUCn_ix, i])  # groupMatrix is always 0-indexed
                computed_mean_CI_cpAUCni[i, 0] = np.mean(vector)
                computed_mean_CI_cpAUCni[i, 1] = 2.262 * np.std(vector, ddof=1) / np.sqrt(len(vector))
                print(f'     mean_cpAUCn.{g}:   {computed_mean_CI_cpAUCni[i, 0]:0.4f}')
            else:
                print(f'     mean_cpAUCn.{g}:   {np.mean(is_not_nan(groupMatrix[:, g, cpAUCn_ix, i])):0.4f}')
            #endif
            print(f'     cpAUCn.{g}:        {formatList(list(groupMatrix[:, g, cpAUCn_ix, i]))}')
            print(f'     bAvgA.{g}:         {formatList(list(groupMatrix[:, g, bAvgA_ix, i]))} - not interpolated to align with group boundaries')
            print(' ')
            if g == 1:
                vector = is_not_nan(groupMatrix[:, g, pAUCn_ix, i])
                computed_mean_CI_pAUCn1[i, 0] = np.mean(vector)
                computed_mean_CI_pAUCn1[i, 1] = 2.262 * np.std(vector, ddof=1) / np.sqrt(len(vector))
                print(f'     mean_pAUCn.1:    {computed_mean_CI_pAUCn1[i, 0]:0.4f}')
            else:
                print(f'     mean_pAUCn.{g}:    {np.mean(is_not_nan(groupMatrix[:, g, pAUCn_ix, i])):0.4f}')
            #endif
            print(f'     pAUCn.{g}:         {formatList(list(groupMatrix[:, g, pAUCn_ix, i]))}')
            print(f'     avgSens.{g}:       {formatList(list(groupMatrix[:, g, avgSens_ix, i]))} - not interpolated to align with group boundaries')
            print(f'     pAUCxn.{g}:        {formatList(list(groupMatrix[:, g, pAUCxn_ix, i]))}')
            print(f'     avgSpec.{g}:       {formatList(list(groupMatrix[:, g, avgSpec_ix, i]))} - not interpolated to align with group boundaries')
            print(' ')
        #endfor
        #print(f'     group0_measFold0: {formatList(list(groupMatrix[0,0,:,i]))}')
    #endfor

    myround = lambda a: round(10000*np.max(a))/100  # as percentage, rounded to 2nd decimal place
    j    = np.argmax(computed_mean_CI_AUC[:, 0])
    print(f'Max mean_AUC     {myround(computed_mean_CI_AUC[j, 0]):0.2f} '
          f'+/- {myround(computed_mean_CI_AUC[j, 1]):0.2f} is at index {j}')
    x, y = setupPlotData(groupMatrix, cpAUCn_ix, num_groups, j, total_folds)
    nice_plot_group_results(j, x, y, 'cpAUCn', computed_mean_CI_AUC, 'AUC',
                            model, num_groups, measure_to_optimize, testNum, 0.35, 1, dotcolor, curvecolor)
    plotIteration(j, testNum, model)

    j = np.argmax(computed_mean_CI_avgPPV[:, 0])
    print(f'Max mean_avgPPV  {myround(computed_mean_CI_avgPPV[j, 0]):0.2f} '
          f'+/- {myround(computed_mean_CI_avgPPV[j, 1]):0.2f} is at index {j}')
    x, y = setupPlotData(groupMatrix, avgPPV_ix, num_groups, j, total_folds)
    # nice_plot_group_results(j, x, y, 'avgPPV', computed_mean_CI_avgPPV, 'avgPPV',
    nice_plot_group_results(j, x, y, 'avgPPV', computed_mean_CI_avgPPV, '',
                            model, num_groups, measure_to_optimize, testNum, 0, 1, dotcolor, curvecolor)

    j = np.argmax(computed_mean_CI_AUPRC[:, 0])
    print(f'Max mean_AUPRC   {myround(computed_mean_CI_AUPRC[j, 0]):0.2f} '
          f'+/- {myround(computed_mean_CI_AUPRC[j, 1]):0.2f} is at index {j}')

    j = np.argmax(computed_mean_CI_avgNPV[:, 0])
    print(f'Max mean_avgNPV  {myround(computed_mean_CI_avgNPV[j, 0]):0.2f} '
          f'+/- {myround(computed_mean_CI_avgNPV[j, 1]):0.2f} is at index {j}')
    x, y = setupPlotData(groupMatrix, avgNPV_ix, num_groups, j, total_folds)
    # nice_plot_group_results(j, x, y, 'avgNPV', computed_mean_CI_avgNPV, 'avgNPV',
    nice_plot_group_results(j, x, y, 'avgNPV', computed_mean_CI_avgNPV, '',
                            model, num_groups, measure_to_optimize, testNum, 0, 1, dotcolor, curvecolor)
    plotIteration(j, testNum, model)

    j = np.argmax(computed_mean_CI_cpAUCni[:, 0])
    print(f'Max mean_cpAUCn1 {myround(computed_mean_CI_cpAUCni[j, 0]):0.2f} '
          f'+/- {myround(computed_mean_CI_cpAUCni[j, 1]):0.2f} is at index {j}')
    x, y = setupPlotData(groupMatrix, cpAUCn_ix, num_groups, j, total_folds)
    # nice_plot_group_results(j, x, y, 'cpAUCn_', computed_mean_CI_cpAUCn1, 'cpAUCn.1',
    nice_plot_group_results(j, x, y, 'cpAUCn_', computed_mean_CI_cpAUCni, '',
                            model, num_groups, measure_to_optimize, testNum, 0.4, 1, dotcolor, curvecolor)
    plotIteration(j, testNum, model)

    j = np.argmax(computed_mean_CI_pAUCn1[:, 0])
    print(f'Max mean_pAUCn1  {myround(computed_mean_CI_pAUCn1[j, 0]):0.2f} '
          f'+/- {myround(computed_mean_CI_pAUCn1[j, 1]):0.2f} is at index {j}')
    x, y = setupPlotData(groupMatrix, pAUCn_ix, num_groups, j, total_folds)
    # nice_plot_group_results(j, x, y, 'pAUCn', computed_mean_CI_pAUCn1, 'pAUCn.1',
    nice_plot_group_results(j, x, y, 'pAUCn', computed_mean_CI_pAUCn1, '',
                            model, num_groups, measure_to_optimize, testNum, 0, 1, dotcolor, curvecolor)

    print(' ')
    transcript.stop()
#enddef

analyze(testNum, model, iterations)
