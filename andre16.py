#!/usr/bin/env python
# coding: utf-8

# andre15.py


###########################################
# Program settings

import sys
import signal
import warnings
import os
import progress_transcript
import acLogging as log
import deepROC2   as ac
import pickle
import time
from   sklearn.impute  import KNNImputer
from   sklearn.metrics import roc_auc_score
from   sklearn.metrics import average_precision_score

global tic, toc, work_time, sleep_time, quiet, resultFile, labelScoreFile, bayes_trial, MAX_TRIALS
global globalP, globalN

#########################################################################
##  IMPORTANT for good performance on servers / not burning out laptops
##
laptop_mode = True
work_time   = 50  # seconds of work
sleep_time  = 20  # seconds of sleep
tic, toc    = 0, 0
##
#########################################################################

#if not sys.warnoptions:
if True:
   warnings.simplefilter("ignore")
   os.environ["PYTHONWARNINGS"]="ignore"
#endif

# Load and print general parameters
fnprefix       = 'output/log_andre_' 
fnsuffix       = '.txt'
logfn, testNum = log.findNextFileNumber(fnprefix, fnsuffix)

# capture standard out to logfile
progress_transcript.start(logfn)

def signal_term_handler(signal, frame):
    progress_transcript.stop() # write buffer to logfile
    sys.exit(0)
#enddef
signal.signal(signal.SIGTERM, signal_term_handler)  # 15: kill -15 pid (soft kill)
signal.signal(signal.SIGINT,  signal_term_handler)  # 2:  Ctrl-C

def getCategoricalIndices(vars, catvars):
    indices = []
    for catvar in catvars:
        try:
            ix = vars.index(catvar)
            indices = indices + [ix]
        except:
            pass
        #endtry
    #endfor
    return indices
#enddef

dataset_name   = 'TB'
data_fn        = 'input-data/TB.csv'
data_header    = True
outcome_fn     = ''
outcome_header = False
quiet          = True
id_col0        = False
converted      = False

# TB variable selection
var5    = ['gender', 'sweats', 'hgbbase', 'temp', 'bmi']
var8    = ['gender', 'sweats', 'hgbbase', 'temp', 'bmi', 'age', 'cough2wks.3', 'wtloss']
#var8    = ['gender', 'sweats', 'hgbbase', 'temp', 'bmi', 'age', 'cough2wks', 'wtloss']
base_categories = ['gender', 'sweats', 'cough2wks.3', 'wtloss']
more_categories = ['Received antibiotic', '1.1a Pulmonary', '1.2a Pleural exudate',
                   '1.2b Pericardial effusion', '1.3a Periph Lymph Nodes',
                   '1.3b Mediastinal lymph nodes', '1.3c Intra-abdom lymph nodes', '1.4 Constitutional',
                   'appetloss', 'feverchills','fatigue', 'chestpain', 'hemoptysis', 'sob', 'lymphnodes',
                   'abdomswel', 'spucough', 'arv', 'priorabx', 'priortbdiag', 'thrush', 'hairyleuko',
                   'shingles', 'ppe', 'necknodes ', ' axilnodes', 'periphnodes', 'abnabdoex']
more_continuous = ['kpsbase', 'heartrate', 'systolicbp', 'diastolicbp']  # removed temp
more_categories_SDH_need_imputation = [' language']
more_categories_need_imputation = ['1.2c Ascitic Exudate', 'groinodes']
more_continuous_need_imputation = ['resprate']
all_continuous = ['hgbbase', 'temp', 'bmi', 'age'] + more_continuous + more_continuous_need_imputation
andre_test     = var5 + ['cough2wks.3'] + more_continuous + more_continuous_need_imputation
allvars = var8 + more_categories + more_continuous + more_continuous_need_imputation + more_categories_need_imputation
varCRP1 = ['crpbase']
varCRP2 = ['screening.CRP_xULN']
varCRP3 = ['No.CRP=1;_no.RTT=2']

outcome = ['Final lab coding: 1=culture.positive; 2=histo+AFB; 9=culture.negative/other']

varFilter = ['hivfinaldx']

#vars         = var5
vars         = var5 + varCRP2
#vars         = allvars + varCRP2
cat_features = getCategoricalIndices(vars, base_categories + more_categories + more_categories_need_imputation +
                                     more_categories_SDH_need_imputation + varCRP3)

main_size = 218   # complete-case-analysis
# main_size = 228     # imputed
print(f'main_size:            {main_size} = training + validation')

# dataset_name   = 'German_Breast_cancer_Study_Group'
# data_fn        = 'input-data/gbsg_X.csv'
# data_header    = True
# outcome_fn     = 'input-data/gbsg_z.csv'
# outcome_header = False
# quiet          = True
# id_col0        = False
#main_size       = 1000
#print(f'main_size:            {main_size} = training + validation')

#dataset_name   = 'Diabetes130'
#data_header    = False
#data_fn        = 'input-data/diabetic_preprocessed1c.csv'
#data_fn       = '../data/public_diabetic_preprocessed1c.csv'
#data_fn       = '/Users/andrecarrington/Data/UCI/hepatitis/hepatitismi.csv'
# id_col0        = False
#main_size      = 100000
#print(f'main_size:            {main_size} = training + validation')
#holdout_size         = 8000 # not currently used
#print(f'holdout_size:         {holdout_size}')  # not currently used


print(f'{logfn}')
print(f'testNum:              {testNum}')
print(f'dataset_name:         {dataset_name}')
print(f'data_fn:              {data_fn}')
print(f'data_header:          {data_header}')
print(f'outcome_fn:           {outcome_fn}')
print(f'outcome_header:       {outcome_header}')
print(f'id_col0:              {id_col0}')
print(f'variables:            {vars}')
print(f'cat_features:         {cat_features}')
print(f'converted:            {converted}')

print(f'laptop_mode:          {laptop_mode}')
print(f'work_time:            {work_time}')
print(f'sleep_time:           {sleep_time}')
print(f'quiet:                {quiet}')

holdout               = False
holdout_random_state  = 18
split_random_state    = 11
smote_random_state    = 30
#split_random_state    = 51
#smote_random_state    = 12
print(f'holdout:              {holdout}')
print(f'holdout_random_state: {holdout_random_state}')
print(f'split_random_state:   {split_random_state}')
print(f'smote_random_state:   {smote_random_state}')

standardize           = True
# quantile_range        = (25, 75)  # standardize to ±1, from IQR, 2sd, 3sd: (25,75), (5,95), (0.3,99.7)
quantile_range        = (5, 95)  # standardize to ±1, from IQR, 2sd, 3sd: (25,75), (5,95), (0.3,99.7)
with_centering        = True
impute                = 'none'      # kNN, none
k                     = 5          # 1, 3, 5

smote_training        = False
smote_neighbors       = 5                 # 5 based on Blagus and Lusa
print(f'standardize:          {standardize}')
print(f'quantile_range:       {quantile_range}')
print(f'with_centering:       {with_centering}')
if impute=='kNN' or impute=='KNN':
    print(f'imputation:           {k}{impute[1:]}')
else:
    print(f'imputation:           {impute}')
print(f'smote_training:       {smote_training}')
print(f'smote_neighbors:      {smote_neighbors}')

measure_to_optimize   = 'AUC'   # e.g., AUC, cpAUC.1 (1st group)
type_to_optimize      = 'area'  # area or group (future: point)
#measure_to_optimize   = 'cpAUCn.2'   # e.g., AUC, cpAUC.1 (1st group)
#type_to_optimize      = 'group'  # area or group (future: point)
#measure_to_optimize   = 'AUPRC'   # e.g., AUC, cpAUC.1 (1st group)
#type_to_optimize      = 'area'  # area or group (future: point)
#measure_to_optimize   = 'pAUCn.1'   # e.g., AUC, cpAUC.1 (1st group)
#type_to_optimize      = 'group'  # area or group (future: point)
#measure_to_optimize   = 'cpAUCn.1'   # e.g., AUC, cpAUC.1 (1st group)
#type_to_optimize      = 'group'  # area or group (future: point)

# note: whole ROC is automatically included as group 0
deepROC_groups        = [[0.0, 0.85], [0.85, 1.0] ]
# deepROC_groups      = [[0.0, 0.167], [0.167, 0.333], [0.333, 0.5],
#                        [0.5, 0.667], [0.667, 0.833], [0.833, 1.0] ]

groupAxis             = 'TPR'   # FPR, TPR or any of: Predicted Risk, Probability or Score
areaMeasures          = ['AUC', 'AUC_full', 'AUC_plain', 'AUC_micro', 'AUPRC']
groupMeasures         = ['cDelta',  'cpAUC',  'pAUC',    'pAUCx',
                         'cDeltan', 'cpAUCn', 'pAUCn',   'pAUCxn',
                         'avgA',    'bAvgA',  'avgSens', 'avgSpec',
                                              'avgPPV',  'avgNPV',
                                              'avgLRp',  'avgLRn',
                         'ubAvgA',  'avgBA',  'sPA']
num_group_measures    = len(groupMeasures)
num_area_measures     = len(areaMeasures)
#pointMeasures        = ['Acc', 'BA',   'Sens', 'Spec',
#                               'BPV',  'PPV',  'NPV'  ]


print(f'measure_to_optimize:  {measure_to_optimize}')
print(f'type_to_optimize:     {type_to_optimize}')
print(f'deepROC_groups:       {deepROC_groups}')
print(f'groupAxis:            {groupAxis}')

cv_option             = 'CV'   # CV, Stratified CV, Bootstrap
k_folds               = 10
repetition            = 10      # >1 results in RepeatedCV, RepeatedStratifiedCV
total_folds           = repetition * k_folds
print(f'cv_option:            {cv_option}')
print(f'k_folds:              {k_folds}')
print(f'repetition:           {repetition}')
print(f'total_folds:          {total_folds}')

hyperparameter_search_api  = 'Hyperopt'  # hyperopt; future: GPyOpt, Optima, etc
hyperparameter_search_mode = 'random'    # hyperopt: random, tree, adaptive_tree
#hyperparameter_search_mode = 'adaptive_tree'    # hyperopt: random, tree, adaptive_tree

print(f'hyperparameter_search_api:     {hyperparameter_search_api}')
print(f'hyperparameter_search_mode:    {hyperparameter_search_mode}')

# which_methods  = ['logr', 'plogr1', 'plogrE2', 'svcMSig', 'rf0', 'svcOISVgc']
# which_methods  = ['plogr2', 'xgb1', 'svcLin', 'svcRBF', 'nnReLU']
# which_methods  = ['knn','gb1','gb2','nsvcOISVgc','svcSig','svcSigN']
#which_methods    = ['logr', 'plogrE2', 'plogr2', 'plogr1', 'svcMSig']
# which_methods    = [ 'rf1', 'svcSigN', 'svcOISVgc']
# which_methods  = ['tpfn']
# which_methods    = ['svcMSig', 'rf1', 'svcOISVgc', 'svcSigN']
# which_methods    = ['logr', 'plogrE2', 'plogr2', 'plogr1']
# which_methods    = ['logr', 'plogrE2', 'plogr1']
# which_methods    = ['svcMSig', 'cb']
# which_methods    = ['tpfn']
which_methods    = ['logr', 'svcOISVpos']
#which_methods    = ['svcLin', 'logr', 'svcMSig', 'plogrE2', 'plogr2', 'svcOISVgc']
#which_methods    = ['logr', 'plogr1', 'svcSigN', 'tpfn', 'rf1', 'svcPwr1']
#which_methods    = ['logr', 'svcRBF', 'xgb1', 'cb']
#which_methods    = ['logr', 'plogr1', 'svcMSig', 'rf1', 'svcOISVgc', 'svcSigN', 'tpfn', 'svcPwr1', 'xgb1', 'cb']
# which_methods    = ['logr', 'plogr1', 'svcMSig', 'rf1', 'svcOISVgc', 'svcSigN', 'tpfn', 'svcPwr1', 'xgb1', 'cb']
#which_methods    = ['logr', 'svcSigN', 'tpfn', 'svcPwr1', 'xgb1', 'cb']
#                    'plogr1', 'svcMSig', 'svcOISVgc', 'svcSigN', 'tpfn', 'svcPwr1', 'xgb1', 'cb']
# which_methods    = ['svcLin', 'plogrE2', 'svcRBF', 'nnReLU']
# which_methods  = ['nsvcOISVgc','nsvcRBF','svcSigN']
# which_methods=['knn','nb','dt','logr',                      # classic stats
#                'plogr1','plogr2','plogrE1','plogrE2',       # penalized logistic regression
#                'rf0','rf1','rf2','gb1','gb2','gb3','gb4',   # tree ensembles
#                'xgb1','xgb2','xgb3',
#                'lsvc_t','lsvc_w','svcLin','svcRBF','svcSig',# classic SVM
#                'svcSigN','svcMSig','svcOISVgc','svcOSig',   # SVM with custom kernels
#                'nsvcRBF','nsvcOISVgc',                      # nu-SVM with classic and custom kernels
#                'nnReLU','nnReLU0','nnTanh']                 # neural networks (shallow)
print(f'which_methods:        {which_methods}')

# num trials for functions with 0,1,2,3,4,5 dimensions in hyperparameter search space
trials_dim     = [1,100,100,100,100,100]           # near optimal, reasonable but not best
#trials_dim     = [1,200,200,200,200,200,200,200]  # adequate
#trials_dim     = [1,3,3,3,3,3]                    # fast for prototype testing
#trials_dim     = [1,20,20,20,20,20]               # fast for prototype testing
#trials_dim    = [1,200,300,600,800,1000]  # better
print(f'trials_dim:           {trials_dim}  (points in hyperparameter seach by model dof')

#######
# Convert measure_to_optimize into indices
i = 0
if type_to_optimize == 'area':
    for m in areaMeasures:
        if measure_to_optimize == m:
            opt_index  = i
            opt_index2 = False
        #endif
        i = i + 1
    #endfor
elif type_to_optimize == 'group':
    name_to_optimize, num_string = measure_to_optimize.split('.')
    for m in groupMeasures:
        if name_to_optimize == m:
            opt_index  = i
            opt_index2 = int(num_string)
        #endif
        i = i + 1
    #endfor
#endif

###########################################
# A. Exploratory data analysis - done separately

###########################################
# B. Examine and report on data, feature extraction/engineering - done separately

###########################################
# C. Make train/validation sets and a (final) holdout test set, general preprocessing

# Load data
import pandas as pd
import numpy  as np
if outcome_fn == '':  # no outcome_fn
    if data_header:
        mydata_df = pd.read_csv(data_fn,    header=0)
    else:
        mydata_df = pd.read_csv(data_fn,    header=None)
    p         = mydata_df.shape[1]          # number of features
    if dataset_name == 'TB':
        X_df = mydata_df
        y_s  = pd.Series(np.zeros(shape=[mydata_df.shape[0], ]))
    else:
        if id_col0:
            X_df      = mydata_df.iloc[:, 1:(p-2)]  # features (_df=dataframe)
        else:
            X_df      = mydata_df.iloc[:, 0:(p-2)]  # features (_df=dataframe)
        y_s       = mydata_df.iloc[:, p-1]      # outcome is last column (_s=series)
    #endif
else:
    if data_header:
        X_df      = pd.read_csv(data_fn,    header=0)
    else:
        X_df      = pd.read_csv(data_fn,    header=None)
    if outcome_header:
        y_s       = pd.read_csv(outcome_fn, header=0).squeeze()
    else:
        y_s       = pd.read_csv(outcome_fn, header=None).squeeze()
#endif

# For TB apply data filter (only use hiv pos cases) and handle missingness
if dataset_name == 'TB':

    # filter on hiv positive (vars and outcome filtered together):
    X_df_temp = X_df[vars+varFilter+outcome].copy()
    filter    = X_df_temp[varFilter] == 2
    filter    = filter.squeeze()
    indices   = filter[filter].index.values
    X_df_temp = X_df_temp.iloc[indices, :]
    X_df_temp.drop(columns=varFilter, inplace=True)

    if impute == 'kNN' or impute == 'KNN':
        y_s       = X_df_temp[outcome].squeeze()
        X_df_temp = X_df_temp[vars]
        X_df_temp.replace('', np.nan, inplace=True)
        imputer   = KNNImputer(n_neighbors=k, weights="uniform")
        X_nd      = imputer.fit_transform(X_df_temp)
        X_df      = pd.DataFrame(X_nd, columns=X_df_temp.columns)
    elif impute == 'none' or impute == 'None':
        # complete case analysis (vars and outcome filtered together):
        X_df_temp.replace('', np.nan, inplace=True)
        X_df_temp.dropna(inplace=True)
        y_s       = X_df_temp[outcome].squeeze()
        X_df      = X_df_temp[vars]
        #X_df.drop(columns=varFilter, inplace=True)
    #endif
#endif
print(f'Shape of data after imputation: {X_df.shape}')

# Convert data continuous variables to binary variables
if dataset_name == 'TB':
    #temp_df = X_df[vars].copy()
    #temp_s  = X_df[outcome].squeeze().copy()
    temp_df = X_df.copy()
    temp_s  = y_s.copy()
    del X_df
    if converted:
        for var in vars:
            if var   == 'gender':
                temp_df['gender']    = temp_df['gender'].map({1: 1, 2: 0})     # 1 male; 2 female
            elif var == 'sweats':
                temp_df['sweats']    = temp_df['sweats'].map({1: 1, 2: 0})     # 1 yes; 2 no
            elif var == 'cough2wks':
                temp_df['cough2wks'] = temp_df['cough2wks'].map({1: 1, 2: 0})  # 1 yes; 2 no
            elif var == 'wtloss':
                temp_df['wtloss']    = temp_df['wtloss'].map({1: 1, 2: 0})     # 1 yes; 2 no
            elif var == 'hgbbase':
                temp_df['hgbbase']   = temp_df['hgbbase'] <= 12
                temp_df['hgbbase']   = temp_df['hgbbase'].map({True: 1, False: 0})
            elif var == 'temp':
                temp_df['temp']      = temp_df['temp'] > 38
                temp_df['temp']      = temp_df['temp'].map({True: 1, False: 0})
            elif var == 'bmi':
                temp_df['bmi']       = temp_df['bmi'] < 18.5
                temp_df['bmi']       = temp_df['bmi'].map({True: 1, False: 0})
            #endfor
        #endif
    #endfor
    X_df = temp_df
    y_s  = temp_s.map({1: 1, 2: 0, 9: 0})  # 1 TB (culture positive), else negative
#endif

# Convert data continuous variables to binary variables
if dataset_name == 'TB':
    usingCatboost = False
    for model in which_methods:
        if model == 'cb':
            usingCatboost = True
    #endfor
    if usingCatboost:
        for index in cat_features:
            var = vars[index]
            if X_df[var].dtype == np.dtype('float64'):
                X_df[var] = X_df[var].astype(int)
            #endif
        #endifor
    #endif
#endif

n = X_df.shape[0]              # number of instances
p = X_df.shape[1]              # number of features
print(f'Data dimensions (instances,features): {n},{p}')

N = sum(y_s == 0)              # number of actual negatives (assume lower value)
P = sum(y_s == 1)              # number of actual positives (assume higher value)
globalP = P
globalN = N
print(f'Outcome positive to negative class ratio: {P}:{N}')

##########
# C.i. Make holdout test set (~10%)
if holdout:
    from sklearn.model_selection import train_test_split
    print('Making holdout test set')
    X_main_df, X_test_df, y_main_s, y_test_s = train_test_split(X_df, y_s, train_size=main_size,
                                                                random_state=holdout_random_state)
    print(f'{X_main_df[0].shape}')
else:
    X_main_df = X_df
    y_main_s  = y_s
#endif

##########
# C.ii Make training/validation sets
#    # option 1: plain cross-validation
#    # option 2: stratified cross validation
#    # option 3: boostraps without replacement (SVM not intended for duplicates)
#    #   option I: repeated CV, e.g. 3 x 10CV = 30 folds

# Fold option 1
if cv_option == 'CV':
    if repetition > 1:
        print('Making training/validation sets with repeatedKFold (CV).')
        from sklearn.model_selection import RepeatedKFold
        kf = RepeatedKFold(n_splits=k_folds, n_repeats=repetition, random_state=split_random_state)
        #kf.get_n_splits(X_main_df)
    else:
        print('Making training/validation sets with KFold (CV)')
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_folds, random_state=split_random_state, shuffle=True)
        #kf.get_n_splits(X_main_df)
    #endif

    X_train_df = [None] * total_folds
    y_train_s  = [None] * total_folds
    X_val_df   = [None] * total_folds
    y_val_s    = [None] * total_folds

    i = 0
    for train_index, val_index in kf.split(X_main_df):
        X_train_df[i] = X_main_df.iloc[train_index]
        y_train_s[i]  = y_main_s.iloc[train_index]
        X_val_df[i]   = X_main_df.iloc[val_index]
        y_val_s[i]    = y_main_s.iloc[val_index]
        i = i + 1
    #endfor
    with open(f'output/kf_andre{testNum:03d}.pkl', 'wb') as output:
        pickle.dump( kf, output, pickle.HIGHEST_PROTOCOL)
    #endwith
    del kf

elif cv_option == 'Stratified CV':
    if repetition > 1:
        print('Making training/validation sets with StratifiedKFold (CV)')
        # StratifiedKFold
    else:
        print('Making training/validation sets with StratifiedKFold (CV)')
        # RepeatedStratifiedKFold
    #endif

elif cv_option == 'Bootstrap':
    print('Making training/validation sets with resample (Bootstrap)')
    # Code here adapted from:
    # https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
    from sklearn.utils import resample
    for i in range(0, total_folds):
        # lookup n_samples parameter
        X_train_df[i] = resample(X_main_df, replace=True, n_samples=4, random_state=split_random_state)
        X_val_df[i]   = [x for x in X_main_df if x not in X_train_df[i]]  # out of bag observations
        y_val_s[i]    = [y for y in y_main_s  if y not in y_train_s[i] ]  # out of bag observations
    #endfor
#endif

##########
# C.iii. Apply centering and standardization 
#    # to each training set/fold and capture the parameters
#    # to each validation set/fold using the parameters from the corresponding training 
#    #   set/fold should be done *prior* to any SMOTE since SMOTE rebalances/changes 
#    #   training but not validation and validation will need parameters from realistic,
#    #   non-SMOTE data
if standardize:
    from sklearn.preprocessing import RobustScaler
    X_train_pure_df = X_train_df.copy()
    X_val_pure_df   = X_val_df.copy()
    if with_centering:
        print('Centering and standardizing data')
    else:
        print('Standardizing data')
    #endif
    for i in range(0, total_folds):
        rs         = RobustScaler(quantile_range=quantile_range,
                                  with_centering=with_centering).fit(X_train_df[i])
        X_train_df[i] = rs.transform(X_train_df[i])
        X_val_df[i]   = rs.transform(X_val_df[i])
    #endfor
    del rs
#endif

##########
# C.iv. Apply SMOTE to each training set/fold (BUT NOT validation--that would be testing
#      on an unrealistic distribution that does not reflect reality, i.e. it would be
#      cheating)

#    # not just as indices into one data set, but as 10 training + 10 validation 
#    #   sets/folds stored separately, so that we can apply SMOTE on training 
#    #   separate from NO SMOTE on validation
if smote_training:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(sampling_strategy='auto', k_neighbors=smote_neighbors,
               random_state=smote_random_state)
    for i in range(0, total_folds):
        print(f'Applying SMOTE to training fold {i}')
        X_train_df[i], y_train_s[i] = sm.fit_resample(X_train_df[i], y_train_s[i])
        # X_val and y_val untouched
    #endfor
    del sm
#endif

##########
# Show the data and target dimensions
# text = ['Validation data', 'Training data', 'Validation target', 'Training target']
# vars = [ X_val_df,         X_train_df,      y_val_s,             y_train_s]
text = ['Training data', 'Validation data' ]
vars = [ X_train_df,      X_val_df,        ]
for i in range(0, len(vars)):
    print(f'{text[i]} dimensions:')
    var = vars[i]
    for j in range(0, total_folds):
        print(f'{var[j].shape}')
    #enddef
#endfor

##########
# What is the largest number of negatives (horizontal ROC axis) in the folds?
# We will use this value to interpolate later when showing the mean ROC
max_negatives = int(0)
for j in range(0, total_folds):
    num_negatives = int( len(y_val_s[j]) - sum(y_val_s[j]) )
    if num_negatives > max_negatives:
        max_negatives = num_negatives
    #endif
#endfor
# print(f'max_negatives: {max_negatives}')

###########################################
# D. Define/make hyperparameter sets for models/kernels (hyperparameter search optimization)
#    # option 1: Bayesian search optimization using random, tree or adaptive_tree
#    # option 2: Random + preselected search
#    # future:   Particle swarm optimization
#    # future:   Genetic algorithm search
#    # store the hyperparameter sets (later it helps to plot performance in the 
#    #   hyperparameter space)

# execute the following files as inline code (many functions are specified in these)
# exec(open('./survival_analysis_models.py'   ).read())
# exec(open('./regression_models.py'          ).read())
exec(open(  './classification_models.py'      ).read())
exec(open(  './hyperparameter_imports.py'     ).read())

import warnings
warnings.filterwarnings('ignore')

if hyperparameter_search_mode == 'random':
    hyperparameter_search_algo = rand.suggest     # random search

if hyperparameter_search_mode == 'tree':
    hyperparameter_search_algo = tpe.suggest      # tree of parzen estimators: search

if hyperparameter_search_mode == 'adaptive_tree':
    hyperparameter_search_algo = tpe.rand.suggest # adaptive tree of parzen estimators: search
#               = atpe.suggest     # adaptive tree of parzen estimators: search

exec(open('./hyperparameter_bayesian_priors.py').read())

##########
# Build an array of classifiers and associated hyperparameter search space information
# cl is a classifier list (a list of dictionaries, 1 per classifier)
cl = np.empty(0)

def addcl(cl, dict1a, dict1b):
    # take a copy of dict1a
    # update it with dict1b if that exists
    # append cl with the combined dict1
    dict1 = dict1a.copy()
    if len(dict1b)>0:
        dict1.update(dict1b)
    return np.append(cl,dict1)
#enddef

##########
#    # Define the hyperparameter space for each classifier
exec(open('./hyperparameter_search_space.py').read())


###########################################
# E. Define/make the abstract models/kernels (not yet populated with hyperparameters)

##########
# Define functions to build models/kernels
def get_cldict(name):
    for dictx in cl:
        if dictx['name']==name:
            return dictx
    return dict()
#enddef

def arrange_kernel_and_params(params, kernel):
    all_params = dict()
    if type(kernel)!=str:      # if a custom kernel
        if len(params)>1:      # get kernel hyperparams: all items except C
            kernel_params = dict()
            for key in params:
                if key!='C' and key!='nu':   # all items except C
                    kernel_params.update({key:params[key]})
                #endif
            #endfor
            complete_kernel = kernel(**kernel_params)
        else:
            complete_kernel = kernel()
        #endif
        # calls re custom kernel
        all_params.update({'kernel':complete_kernel})
    else:                       # built-in kernel
        all_params.update({'kernel':kernel})
    #endif
    #print(f'kernel_params: {kernel_params}')
    #print(' ')
    return all_params
#enddef

def arrange_classifier_params(params, fixed, fixed2):

    if len(fixed2)>0:

        # the following is only for SVC
        if "kernel" in fixed2:
            kernel = fixed2['kernel']
            all_params = arrange_kernel_and_params(params, kernel)

            # a kernel implies SVC, add SVC params from params
            if 'C' in params:
                all_params.update({'C':params['C']})
            elif 'nu' in params:
                all_params.update({'nu':params['nu']})
            #endif

            if len(fixed)>0:
                all_params.update(fixed)
            #endif

        else:  # no kernel specified (fixed2 not empty), add all the params

            # did I also want to add params here ?
            all_params = dict()

            if len(fixed)>0:
                all_params.update(fixed)
            #endif
            all_params.update(fixed2)
        #endif

    else:  # no kernel specified, fixed2 empty, add all the params
        all_params = dict()
        if len(params)>0:
            all_params.update(params)
        #endif
        if len(fixed)>0:
            all_params.update(fixed)
        #endif
    #endif

    #print(f'params: {params}')
    #print(' ')
    #print(f'all_params: {all_params}')
    #print(' ')
    return all_params
#enddef


###########################################
# F. Optimize on training data and predict on validation data

##########
# Setup helper functions
def check_for_laptop_nap():
    global tic, toc, work_time, sleep_time
    toc  = time.perf_counter()
    if (toc - tic) > work_time:
        print('napping!')
        time.sleep(sleep_time)
        tic  = time.perf_counter()
        print('awake!  ')
    #endif
#enddef

##########
# Setup advanced performance measures
def doAdvancedMeasures(scores, labels, groupAxis, deepROC_groups, testNum):
    global quiet, globalP, globalN
    costs = dict(cFP=1, cFN=1, cTP=0, cTN=0, rates=False)
    results, EQresults = ac.deepROC(costs=costs,     showPlot=False,  showData=False, showError=False,
                                    globalP=globalP, globalN=globalN, scores=scores,   labels=labels,  posclass=1,
                                    testNum=testNum, pAUCranges=deepROC_groups,   rangeAxis=groupAxis,
                                    useCloseRangePoint=False, sanityCheckWholeAUC=True, quiet=quiet)
    return results, EQresults
#enddef


##########
# main code to train and validate each model with hyperparameter search

# start the timer, independent of any classifier
tic  = time.perf_counter()

i            = 0
bayes_trials = np.empty(0)
bayes_trial  = np.empty(0) # just something to delete

objective=[]
for name in which_methods: # for each model

    cldict       = get_cldict(name)
    space_dim    = len(cldict['space'])
    MAX_TRIALS   = trials_dim[space_dim]
    num_groups   = len(deepROC_groups) + 1  # automatically include (add) whole ROC as group 0
    areaMatrix   = np.zeros(shape=[total_folds, num_area_measures, MAX_TRIALS])
    #                              30           x5                 x100
    groupMatrix  = np.zeros(shape=[total_folds, num_groups, num_group_measures, MAX_TRIALS])
    #                              30           x6          x15                 x100 = 270k * 4B = 1.08 MB

    print(' ')
    print(name) # classifier name as header for progress bar and results
    print(f"hyperparameter search space: {cldict['space']}")
    print(f'space_dim: {space_dim}')

    if hyperparameter_search_api == 'Hyperopt':
        del bayes_trial
        bayes_trial=Trials()
    #endif

    ##########
    # define objective function used in hyperparameter search iterations
    del objective
    def objective(params, n_folds=10):  # define the optimization objective for each model
        global resultFile, labelScoreFile, bayes_trial, MAX_TRIALS
        import warnings
        warnings.filterwarnings('ignore')

        abstract_classifier = cldict['classifier']
        fixed               = cldict['fixed']
        fixed2              = cldict['fixed2']
        #measure_name       = cldict['measure']
        loss_sign           = -1
        #print(f'fixed: {fixed}')
        #print(f'fixed2: {fixed2}')
        #print(' ')

        # Make the concrete model/kernel with the hyperparameters
        all_params     = arrange_classifier_params(params,fixed,fixed2)
        #print(' ')
        #progress_transcript.stop() # write buffer to logfile
        #sys.exit(0)
        #raise SystemExit(0)        # alternate exit
        #print(f'all_params: {all_params}')
        classifier     = [abstract_classifier(**all_params)] * total_folds

        # the following approach allows indexing, instead of initializing to null and appending
        y_val_scores   = [None] * total_folds
        #y_train_scores = [None] * total_folds

        # Fit and predict
        # Can run the 10 folds in parallel...
        for fold in range(0, total_folds):

            if laptop_mode:
                check_for_laptop_nap()
            #endif

            # train/fit (optimize) classifier
            if cldict['name'] == 'cb':
                classifier[fold].fit(X_train_pure_df[fold], y_train_s[fold])
            else:
                classifier[fold].fit(X_train_df[fold], y_train_s[fold])
            #print(f'class order: {classifier[fold].classes_}   ')
            if cldict['name'] == 'cb':
                y_val_scores[fold]    = classifier[fold].predict_proba(X_val_pure_df[fold])[:, 1]
            else:
                y_val_scores[fold]    = classifier[fold].predict_proba(X_val_df[fold])[:, 1]
            #y_train_scores[fold] = classifier[fold].predict_proba(X_train_df[fold])[:, 1]

            # Measure performance (common measures) for each hyperparameter set
            trial_num = len(bayes_trial)-1
            logTestNum                      = f'{testNum}-{name}-{fold}'
            results, EQresults              = doAdvancedMeasures(y_val_scores[fold], y_val_s[fold],
                                                                 groupAxis, deepROC_groups, logTestNum)

            for group in range(0, num_groups):
                # note: these are validation results, not training results
                if group == 0:
                    areaMatrix[fold, :, trial_num]     = np.array([results[group][m] for m in areaMeasures])
                #endif
                groupMatrix[fold, group, :, trial_num] = np.array([results[group][m] for m in groupMeasures])
            #endfor
        #endfor

        # the following lambda function allows us to handle np.nan when computing mean, sum, etc with results
        is_not_nan = lambda a: a[np.invert(np.isnan(a))]

        if   type_to_optimize == 'area':
            mean_measure   = np.mean(is_not_nan(areaMatrix[:, opt_index, trial_num]))         # mean across folds
            # mean_measure = np.mean(is_not_nan(areaMatrix[:, opt_index, trial_num]), axis=0) # mean across folds
        elif type_to_optimize == 'group':
            mean_measure   = np.mean(is_not_nan(groupMatrix[:, opt_index2, opt_index, trial_num]))        # mean across folds
            # mean_measure = np.mean(is_not_nan(groupMatrix[:, opt_index2, opt_index, trial_num], axis=0) # mean across folds
        #endif

        # every iteration, store (append) the validation labels, scores
        pickle.dump([y_val_s, y_val_scores], labelScoreFile, pickle.HIGHEST_PROTOCOL)
        # in future: could store the training labels, scores; and the learned classifier too
        # pickle.dump( [y_train_s, y_train_scores], labelScoreFile, pickle.HIGHEST_PROTOCOL)
        # pickle.dump(classifier, labelScoreFile, pickle.HIGHEST_PROTOCOL)

        # every 10 iterations, store (over-write) the validation performance measures
        if np.mod(trial_num+1, 10) == 0 or (trial_num+1) ==  MAX_TRIALS:
            resultFile = open(f'output/results_{testNum:03d}_{name}.pkl', 'wb')
            pickle.dump([areaMatrix, groupMatrix], resultFile, pickle.HIGHEST_PROTOCOL)
            # in future: could create and store matrices for training results too
            resultFile.close()
        #endif

        if mean_measure == np.nan:
            return 0  # -100 is best, 0 is worst
        else:
            return loss_sign * mean_measure  # return a loss that will be minimized
        #endif
    #enddef

    ##########
    # hyperparameter search with the optimization objective defined above
    labelScoreFile = open(f'output/labelScore_{testNum:03d}_{name}.pkl', 'wb')

    settingsFile   = open(f'output/settings_{testNum:03d}.pkl', 'wb')
    # Store settings
    pickle.dump([measure_to_optimize, type_to_optimize, deepROC_groups,
                 groupAxis, areaMeasures, groupMeasures], settingsFile, pickle.HIGHEST_PROTOCOL)
    settingsFile.close()

    try:
        if space_dim>0: # no search required if there is no hyperparameter space

            if hyperparameter_search_api == 'Hyperopt':
                #print(' '); print(' '); print(' ')
                print('ACTION:START_PROGRESS_BAR')
                # Do training optimization and test prediction
                best = fmin(objective, cldict['space'], algo=hyperparameter_search_algo,
                            max_evals=MAX_TRIALS, trials=bayes_trial)
                print('ACTION:STOP_PROGRESS_BAR')
                print(best)
                bayes_trials=np.append(bayes_trials, bayes_trial) # store results of every iteration
            #endif
        else:
            best = objective({})
            print(best)
        #endif
    finally:
        labelScoreFile.close()
#endfor

progress_transcript.stop()
