# AnalyzeDeepROCFolds.py
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
testNum           = 2
model             = 'svcOISVpos'
iterations        = 100 # except for the next clause...
if model == 'resnet1013D':  # matlab v2021a only captures the last iteration
    iterations    = 1
if model == 'logr' or model == 'tpfn':  # these methods have only 1 iteration
    iterations    = 1
iterationToMeasure= None
measureToAnalyze = 'avgSens.2'  # 0-indexed
#measureToAnalyze = 'AUCn_i.2'  # 0-indexed
totalFolds        = 100  # 10x10CV
groupAxis         = 'TPR'
#groupAxis         = 'FPR'
#groups            = [[0, 0.15], [0.15, 1], [0, 1]]
groups            = [[0, 0.65], [0.65, 0.85], [0.85, 1], [0, 1]]
#group            = [[0, 1], [0, 0.65], [0.65, 0.85], [0.85, 1]]
costs             = dict(cFP=1, cFN=5, cTP=0, cTN=0, costsAreRates=False)
doPlot            = True
showPlot          = True
doAnalyze         = True
whichMeasures     = ['AUCn_i', 'avgSens', 'avgSpec', 'avgPPV', 'avgNPV']

class AnalyzeDeepROCFolds(object):

    def __init__(self, testNum=None, model=None, iterationToMeasure=None, iterations=None, totalFolds=None,
                 groupAxis=None, groups=None, costs=None, doPlot=True, showPlot=True, doAnalyze=True,
                 whichMeasures=None, measureToAnalyze=measureToAnalyze):
        from deeproc import DeepROC
        import transcript

        self.testNum            = testNum
        self.model              = model
        self.iterations         = iterations
        self.iterationToMeasure = iterationToMeasure
        self.totalFolds         = totalFolds
        self.groupAxis          = groupAxis
        self.groups             = groups
        self.costs              = costs
        self.doPlot             = doPlot
        self.showPlot           = showPlot
        self.doAnalyze          = doAnalyze
        self.whichMeasures      = whichMeasures
        self.measureToAnalyze   = measureToAnalyze
        self.NPclassRatio       = None
        self.wholeGroup         = None
        self.bestIteration      = None
        self.iteration_a        = None
        self.labelScoreFile     = None
        self.bestMeanMeasure    = None
        self.CIplusminus        = None
        if measureToAnalyze[0:6] == 'avgLRn':
            self.smallerIsBetter = True
        else:
            self.smallerIsBetter = False

        self.findOrMakeWholeGroup()

        logfn = f'output/AnalyzeDeepROCFolds_{self.testNum:3d}_{self.model}.txt'
        transcript.start(logfn)
        self.deepROC = DeepROC.DeepROC()
        self.deepROC.setGroupsBy(groupAxis=self.groupAxis, groups=self.groups, groupByClosestInstance=False)

        if self.model == 'resnet1013D':
            self.loadMatlab(inputdir='input-ms', prefix='labelScore_', scoreVariable='scores', targetVariable='labels')
        else:
            self.load()
        #endif
        if self.doPlot:
            self.plot()
        if self.doAnalyze:
            self.analyze()
        transcript.stop()
    #endef

    def findOrMakeWholeGroup(self):
        numgroups  = len(self.groups)
        self.wholeGroup = -1
        for i in range(0, numgroups):
            if self.groups[i] == [0, 1]:
                self.wholeGroup = i
        #endfor
        if self.wholeGroup == -1:
            self.groups.extend([[0, 1]])
            self.wholeGroup = numgroups
        #endif
    #enddef

    def getMeasures(self, iteration_a, measure, setFolds=False):
        from sklearn.metrics import roc_curve
        from deeproc import DeepROC
        pos = 0
        neg = 0
        measures = []
        measure_part = measure.split(".")
        for i in range(0, self.totalFolds):
            labels = iteration_a[0]
            scores = iteration_a[1]

            temp_pos = sum(labels[i])
            pos     += temp_pos
            neg     += (len(labels[i]) - temp_pos)
            NPclassRatio = neg / pos
            #print(type(scores[i]))
            #print(type(labels[i]))
            print('', end='')

            tempROC = DeepROC.DeepROC(predicted_scores=scores[i], labels=labels[i])
            tempROC.setGroupsBy(groupAxis=self.groupAxis, groups=self.groups)
            tempROC.setNPclassRatio(NPclassRatio=NPclassRatio)
            if   len(measure_part) == 1:  # whole measure
                measure_dict = tempROC.analyze()
                print(measure_dict)
                measures.extend([measure_dict[measure]])
            elif len(measure_part) == 2:  # group measure
                groupNum     = int(measure_part[1])
                _, measure_dict = tempROC.analyzeGroup(groupNum, quiet=True)
                measures.extend([measure_dict[measure_part[0]]])
            else:
                del tempROC
                raise ValueError('Unexpected format for measure name.')
            #endif
            del tempROC

            if setFolds:
                fpr, tpr, threshold = roc_curve(labels[i], scores[i], drop_intermediate=True)
                self.deepROC.set_fold(fpr=fpr, tpr=tpr, threshold=threshold)
            #endif
        #endfor
        self.NPclassRatio = neg / pos
        if setFolds:
            self.deepROC.setFoldsNPclassRatio(self.NPclassRatio)
        else:
            self.deepROC.setNPclassRatio(self.NPclassRatio)
        print('.', end='')
        return measures, pos, neg
    #enddef

    def getAUCs(self, iteration_a, setFolds=False):
        from sklearn.metrics import roc_curve, roc_auc_score
        pos = 0
        neg = 0
        AUCs = []
        for i in range(0, self.totalFolds):
            labels = iteration_a[0]
            scores = iteration_a[1]
            fpr, tpr, threshold = roc_curve(labels[i], scores[i], drop_intermediate=True)
            AUCs.extend([roc_auc_score(labels[i], scores[i])])
            temp_pos = sum(labels[i])
            pos     += temp_pos
            neg     += (len(labels[i]) - temp_pos)
            if setFolds:
                self.deepROC.set_fold(fpr=fpr, tpr=tpr, threshold=threshold)
        #endfor
        return AUCs, pos, neg
    #enddef

    def loadMatlab(self, inputdir, prefix, scoreVariable, targetVariable):
        import scipy.io    as sio
        import numpy       as np

        if self.model == 'resnet1013D':
            self.iterationToMeasure = 0
            iteration_a = [[], []]
            for f in range(1, self.totalFolds+1):
                fileName = f'{inputdir}/{prefix}{self.testNum:03d}_{f:02d}.mat'
                try:
                    fileContent = sio.loadmat(fileName)  # handle any file not found errors naturally
                    scores = fileContent[scoreVariable]
                    labels = fileContent[targetVariable]
                except:
                    raise ValueError(f'File {fileName} is either not found or is not a matlab file')
                # endtry
                iteration_a[0].extend([labels.flatten()])
                iteration_a[1].extend([scores.flatten()])
                print('.', end='')
            # endfor
            self.measures, _, __ = self.getMeasures(iteration_a, self.measureToAnalyze, setFolds=True)
            self.bestMeanMeasure = np.mean(self.measures)
            self.CIplusminus     = 1.96 * np.std(self.measures, ddof=1) / np.sqrt(len(self.measures))
            print(f'\n{self.measureToAnalyze} at iteration {self.iterationToMeasure:03d} is ' +
                  f'{self.bestMeanMeasure:0.3f} +/-{self.CIplusminus:0.4f}')
        else:
            print('Not implemented.')
            return
        #endif
    #enddef

    def load(self):
        import pickle
        import numpy as np

        if self.iterationToMeasure is None:
            # Need to get the iteration that is best for the specified measure
            self.labelScoreFile = open(f'output/labelScore_{self.testNum:03d}_{self.model}.pkl', 'rb')
            #self.bestMeanAUC    = -1
            self.bestMeanMeasure = -1
            self.CIplusminus     = -1
            for i in range(0, self.iterations):
                iteration_a      = pickle.load(self.labelScoreFile)
                measures, _, __  = self.getMeasures(iteration_a, self.measureToAnalyze, setFolds=False)
                meanMeasure      = np.mean(measures)
                #AUCs, _, __     = self.getAUCs(iteration_a, setFolds=True)
                #meanAUC         = np.mean(AUCs)
                #if meanAUC > self.bestMeanAUC:
                    #self.bestMeanAUC    = meanAUC
                if meanMeasure > self.bestMeanMeasure:
                    self.measures        = measures
                    self.bestMeanMeasure = meanMeasure
                    self.bestIteration   = i
                    self.iteration_a     = iteration_a
                    #self.CIplusminus    = 1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(AUCs))
                    self.CIplusminus     = 1.96 * np.std(measures, ddof=1) / np.sqrt(len(measures))
                #endif
            #endfor
            print(' ')
            self.labelScoreFile.close()
            # rounding up/down is automatic with formatted print
            print(f'\nBest Mean {measureToAnalyze} for {model} is {self.bestMeanMeasure:0.3f}' +
                  f'+/-{self.CIplusminus:0.4f} at iteration {self.bestIteration:03d}.\n')
            #print(f'\nBest Mean AUC for {model} is {self.bestMeanAUC:0.3f} +/-{self.CIplusminus:0.4f} ' +
            #      f'at iteration {self.iterationToMeasure}.\n')
            not_used = self.getMeasures(iteration_a, self.measureToAnalyze, setFolds=True)  # setFolds
        else:
            # Need to get the iteration specified
            self.labelScoreFile = open(f'output/labelScore_{self.testNum:03d}_{self.model}.pkl', 'rb')
            for i in range(0, self.iterationToMeasure):
                iteration_a      = pickle.load(self.labelScoreFile)
            #endfor
            self.labelScoreFile.close()
            self.measures, _, __ = self.getMeasures(iteration_a, self.measureToAnalyze, setFolds=True)
            self.bestMeanMeasure = np.mean(self.measures)
            self.CIplusminus     = 1.96 * np.std(self.measures, ddof=1) / np.sqrt(len(self.measures))
            print(f'\n{self.measureToAnalyze} at iteration {self.iterationToMeasure:03d} is ' +
                  f'{self.bestMeanMeasure:0.3f} +/-{self.CIplusminus:0.4f}')
        #endif
    #enddef

    def plot(self):
        if self.iterationToMeasure is None:
            iText = self.bestIteration
        else:
            iText = self.iterationToMeasure
        title = f'Mean ROC plot for {self.model} in test #{self.testNum} iter #{iText:03d}'
        saveFileName = f'output/mean_{self.testNum}_{self.model}_i{iText:03d}.png'
        self.deepROC.plot_folds(title, showOptimalROCpoints=True, costs=self.costs,
                                saveFileName=saveFileName, showPlot=self.showPlot, showLegend=False)
    #enddef

    def analyze(self):
        import numpy as np
        import matplotlib.pyplot as plt

        #print(f'AUCs: ', end='')
        #for auc in self.AUCs:
        #    print(f'{auc:0.3f}, ', end='')
        #print(f'\nMean AUC {np.mean(self.AUCs):0.4f}')

        print(f'\n{self.measureToAnalyze} in each fold: ', end='')
        for measure in self.measures:
            print(f'{measure:0.3f}, ', end='')
        print('')

        self.deepROC.setGroupsBy(groupAxis=self.groupAxis, groups=self.groups, groupByClosestInstance=False)
        measure_dict = self.deepROC.analyze(forFolds=True, verbose=True)

        numgroups = len(groups)
        for z in range(0, numgroups):
            print(measure_dict[z])
            print('')

        if self.doPlot:
            numgroups = len(groups)
            print('\nPlotting groups/ROIs in ROC plot.')
            if self.iterationToMeasure is None:
                iText = self.bestIteration
            else:
                iText = self.iterationToMeasure
            saveFileName = f'output/group_{self.testNum}_{self.model}_i{iText:03d}.png'
            for i in range(0, numgroups):
                if self.showPlot:
                    shown_saved = 'shown and saved'
                else:
                    shown_saved = 'saved'
                print(f'ROC plot {shown_saved} for group {i} [{groups[i][0]:0.2f}, {groups[i][1]:0.2f}]')
                title = f'Deep ROC: group {i+1} for {self.model} in test #{self.testNum} iter #{iText:03d}'
                fig, ax = self.deepROC.plotGroupForFolds(plotTitle=title, groupIndex=i,
                                                         foldsNPclassRatio=self.NPclassRatio,
                                                         showError=False, showThresholds=False,
                                                         showOptimalROCpoints=True, costs=self.costs,
                                                         saveFileName=None, numShowThresh=20, showPlot=False,
                                                         labelThresh=False, full_fpr_tpr=True)
                top = False
                if self.deepROC.groupAxis == 'FPR' and measure_dict[i]['avgSens'] < 0.5:
                    top = True
                elif self.deepROC.groupAxis == 'TPR' and measure_dict[i]['avgSpec'] < 0.5:
                    top = True
                self.deepROC.annotateGroup(i, measure_dict, self.whichMeasures, top=top)

                if saveFileName is not None:
                    temp = saveFileName[0:-4] + f'_{i+1}.png'
                    fig.savefig(temp)
                # endif

                if self.showPlot:
                    plt.show()
            #endfor
        #endif
    #enddef

#endclass

a = AnalyzeDeepROCFolds(testNum=testNum, model=model, iterationToMeasure=iterationToMeasure, iterations=iterations,
                        totalFolds=totalFolds, groupAxis=groupAxis, groups=groups, costs=costs, doPlot=doPlot,
                        showPlot=showPlot, doAnalyze=doAnalyze, whichMeasures=whichMeasures,
                        measureToAnalyze=measureToAnalyze)
