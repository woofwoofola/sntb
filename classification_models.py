# classification_models.py
# Copyright 2020 André Carrington, Ottawa Hospital Research Institute
# Use is subject to the Apache 2.0 License
# Written by André Carrington
#
# define classification models (abstract models, without hyperparameters)
#
#    models
#    kernels
#    data helper functions
#    cross validation functions
#    for future development:
#    old or other:

#  models
from sklearn.neighbors      import KNeighborsClassifier        as knn
from sklearn.naive_bayes    import GaussianNB                  as nb
from sklearn.linear_model   import LogisticRegression          as logr
from sklearn.tree           import DecisionTreeClassifier      as dt
from sklearn.ensemble       import RandomForestClassifier      as rf
from sklearn.ensemble       import GradientBoostingClassifier  as gb
from sklearn.svm            import LinearSVC                   as lsvc
from sklearn.svm            import SVC                         as svc
from sklearn.svm            import NuSVC                       as nsvc
from xgboost                import XGBRFClassifier             as xgb
from sklearn.neural_network import MLPClassifier               as nn
from catboost               import CatBoostClassifier          as cb
from tabpfn                 import TabPFNClassifier            as tpfn

#    kernels
from acCustomKernels        import MSig_kernel,   MSig0_kernel
from acCustomKernels        import OISVgc_kernel, OISVneg_kernel, OISVpos_kernel
from acCustomKernels        import OSig_kernel
from acCustomKernels        import SigN_kernel
from acCustomKernels        import Pwr1_kernel,   Log1_kernel

#    data helper functions
from xgboost                import DMatrix

#    for future development:
#from catboost               import Pool                        as cbpool
#from fukuml                 import KernelLogisticRegression    as klogr
#from fukuml                 import KernelRidgeClassifier       as krc
#from sklearn.naive_bayes    import ComplementNB                as cnb

#    old or other:
#from xgboost                import XGBClassifier               as xgb
#from sklearn.naive_bayes    import CategoricalNB               as cnb
